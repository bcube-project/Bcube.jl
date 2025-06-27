module BcubeGPU
# temporary / dev imports
# using Cthulhu
# using CUDA # just for debug, to be commented once solved
using InteractiveUtils
using BenchmarkTools

# standard imports
using Bcube
using StaticArrays
using KernelAbstractions
using GPUArrays # just to access the type AbstractGPUArray for dispatch of `inner_faces`
using Adapt
using SparseArrays
import Bcube:
    AbstractCellDomain,
    AbstractFaceDomain,
    AllFaceDomain,
    CellInfo,
    CellPoint,
    Connectivity,
    DofHandler,
    DomainIterator,
    FaceInfo,
    FacePoint,
    FaceSidePair,
    LazyMapOver,
    Mesh,
    MeshConnectivity,
    NullOperator,
    PhysicalDomain,
    ReferenceDomain,
    Side⁻,
    Side⁺,
    SingleFESpace,
    SingleFieldFEFunction,
    boundary_faces,
    boundary_nodes,
    connectivities,
    entities,
    get_args,
    get_cellinfo_n,
    get_cellinfo_p,
    get_cell_shape_functions,
    get_cell_side_n,
    get_cell_side_p,
    get_coords,
    get_dof,
    get_dirichlet_boundary_tags,
    get_element_index,
    get_element_type,
    get_function_space,
    get_mapping,
    get_metadata,
    get_ncomponents,
    get_quadrature,
    get_shape_functions,
    idof_by_face_with_bounds,
    indices,
    integrate_on_ref_element,
    is_continuous,
    nfaces,
    nlayers,
    shape,
    shape_functions,
    _get_dhl,
    _get_index,
    _scalar_shape_functions

const WORKGROUP_SIZE = 32

include(joinpath(@__DIR__, "temp.jl"))
include(joinpath(@__DIR__, "assemble_linear.jl"))
include(joinpath(@__DIR__, "assemble_bilinear.jl"))

"""
Structure representing a sparse Matrix with non-empty rows, meaning that each
row contains at least on element.

Rq : we can actually remove this non-empty constraint easily

`offset` starts at 0 : `offset[1] = 0`
"""
struct DenseRowsSparseCols{O, D}
    offset::O # 1:n
    values::D # values[offset[i]+1:offset[i]+offset[i+1]]
end

function get_n_elts(x::DenseRowsSparseCols, i)
    l = length(x.offset)
    n = x.offset[i]
    m = (i < l) ? x.offset[i + 1] : length(x.values)
    return m - n
end

get_elt(x::DenseRowsSparseCols, i, j) = x.values[x.offset[i] + j]

"""
This structure actually represents two "sparse" matrices. The first sparse matrix is (idof, ielt), the
second is (idof, iloc of idof in ielt).
"""
struct ReverseDofHandler{A}
    dof2elt::A
    dof2iloc::A
end

get_n_elts(rdhl::ReverseDofHandler, idof) = get_n_elts(rdhl.dof2elt, idof)
get_ielt(rdhl::ReverseDofHandler, idof, j) = get_elt(rdhl.dof2elt, idof, j)
get_iloc(rdhl::ReverseDofHandler, idof, j) = get_elt(rdhl.dof2iloc, idof, j)

function ReverseDofHandler(dof_to_elts)
    ndofs = length(dof_to_elts)
    offset = zeros(Int, ndofs)
    nelts = zeros(Int, ndofs)
    n = sum(length.(dof_to_elts))
    ielts = zeros(Int, n)
    iloc = zeros(Int, n)
    curr = 1
    for (idof, x) in enumerate(dof_to_elts)
        nelts[idof] = length(x)
        for (ielt, _iloc) in x
            ielts[curr] = ielt
            iloc[curr] = _iloc
            curr += 1
        end
        (idof > 1) && (offset[idof] = offset[idof - 1] + nelts[idof - 1])
    end
    dof2elt = DenseRowsSparseCols(offset, ielts)
    dof2iloc = DenseRowsSparseCols(offset, iloc)
    return ReverseDofHandler{typeof(dof2elt)}(dof2elt, dof2iloc)
end

"""
Build the dof -> cell ReverseDofHandler

Build the connectivity <global dof index> -> <cells surrounding this dof, local index of this dof
in the cells (all components merged)>
"""
function ReverseDofHandler(domain::AbstractCellDomain, U)
    dof_to_cells = [Tuple{Int, Int}[] for _ in 1:get_ndofs(U)]
    dhl = _get_dhl(U)
    ncomps = get_ncomponents(U)
    for cInfo in DomainIterator(domain)
        icell = get_element_index(cInfo)
        kloc = 1
        for icomp in 1:ncomps
            for iloc in 1:get_ndofs(dhl, icell, icomp)
                idof = get_dof(dhl, icell, icomp, iloc)
                push!(dof_to_cells[idof], (icell, kloc))
                kloc += 1
            end
        end
    end
    return ReverseDofHandler(dof_to_cells)
end

"""
Build the dof -> face ReverseDofHandler

Build the connectivity <global dof index> -> <faces (side) surrounding this dof, local index of this dof in the attached faces>

Note that for interior faces, a dof may belong (in FEM) to two "face side", i.e two cells.
For a given face number, dofs lying on the negative side will have a minus sign.
"""
function ReverseDofHandler(domain::AbstractFaceDomain, U)
    dof_to_faces = [Tuple{Int, Int}[] for _ in 1:get_ndofs(U)]
    dhl = _get_dhl(U)
    fs = get_function_space(U)

    for (iface_l, fInfo) in enumerate(DomainIterator(domain))
        iface_g = get_element_index(fInfo)

        # Negative side
        cInfo_n = get_cellinfo_n(fInfo)
        cshape_n = shape(get_element_type(cInfo_n))
        icell_n = get_element_index(cInfo_n)
        cside_n = get_cell_side_n(fInfo)
        for iloc in 1:get_ndofs(dhl, icell_n)
            idof = get_dof(dhl, icell_n, 1, iloc) # comp = 1
            push!(dof_to_faces[idof], (iface_l, -iloc)) # -iloc because negative side
        end
        # for iloc in idof_by_face_with_bounds(fs, cshape_n)[cside_n]
        #     idof = get_dof(dhl, icell_n, 1, iloc) # comp = 1
        #     push!(dof_to_faces[idof], (iface, -iloc)) # -iloc because negative side
        # end

        # Positive side
        cInfo_p = get_cellinfo_p(fInfo)
        icell_p = get_element_index(cInfo_p)
        (icell_n == icell_p) && continue # no positive side (boundary face)
        cshape_p = shape(get_element_type(cInfo_p))
        cside_p = get_cell_side_p(fInfo)
        for iloc in 1:get_ndofs(dhl, icell_p)
            idof = get_dof(dhl, icell_p, 1, iloc) # comp = 1
            push!(dof_to_faces[idof], (iface_l, iloc)) # +iloc because positive side
        end
        # for iloc in idof_by_face_with_bounds(fs, cshape_p)[cside_p]
        #     idof = get_dof(dhl, icell_p, 1, iloc) # comp = 1
        #     push!(dof_to_faces[idof], (iface, iloc)) # +iloc because positive side
        # end
    end

    return ReverseDofHandler(dof_to_faces)
end

#>>>>>>>> Adapt some structures
Adapt.@adapt_structure Connectivity

function Adapt.adapt_structure(to, conn::MeshConnectivity{C, F, T, B}) where {C, F, T, B}
    layers = adapt(to, nlayers(conn))
    ind = adapt(to, indices(conn))
    MeshConnectivity{typeof(ind), F, T, B, typeof(layers)}(layers, ind)
end

function Adapt.adapt_structure(to, mesh::Mesh)
    nodes_gpu = adapt(to, get_nodes(mesh))
    entities_gpu = adapt(to, entities(mesh))
    connectivities_gpu = adapt(to, connectivities(mesh))
    bc_nodes_gpu = adapt(to, boundary_nodes(mesh))
    bc_faces_gpu = adapt(to, boundary_faces(mesh))
    metadata_gpu = adapt(to, get_metadata(mesh))

    Mesh{
        topodim(mesh),
        spacedim(mesh),
        typeof(nodes_gpu),
        typeof(entities_gpu),
        typeof(connectivities_gpu),
        typeof(bc_nodes_gpu),
        typeof(bc_faces_gpu),
        typeof(metadata_gpu),
    }(
        nodes_gpu,
        entities_gpu,
        connectivities_gpu,
        bc_nodes_gpu,
        bc_faces_gpu,
        metadata_gpu,
    )
end

Adapt.@adapt_structure CellDomain
Adapt.@adapt_structure InteriorFaceDomain

Adapt.@adapt_structure Measure

Adapt.@adapt_structure DofHandler

function Adapt.adapt_structure(to, feSpace::SingleFESpace{S, FS}) where {S, FS}
    dhl = adapt(to, _get_dhl(feSpace))
    tags = adapt(to, get_dirichlet_boundary_tags(feSpace))
    SingleFESpace{S, FS, typeof(dhl), typeof(tags)}(
        get_function_space(feSpace),
        dhl,
        is_continuous(feSpace),
        tags,
    )
end

Adapt.@adapt_structure TestFESpace
Adapt.@adapt_structure TrialFESpace
Adapt.@adapt_structure SingleFieldFEFunction

Adapt.@adapt_structure DenseRowsSparseCols
Adapt.@adapt_structure ReverseDofHandler
#<<<<<<<< Adapt some structures

@kernel function inner_faces_kernel!(n_neighbors, @Const(f2c))
    iface = @index(Global)
    n_neighbors[iface] = length(f2c[iface])
end

function Bcube.inner_faces(mesh::Mesh{T, S, N}) where {T, S, N <: AbstractGPUArray}
    # TODO : recall why we can't just use `n_neighbors = AK.map(length, f2c)` ?
    # (maybe I haven't tried)
    f2c = indices(connectivities(mesh, :f2c))
    backend = get_backend(get_nodes(mesh))
    n_neighbors = KernelAbstractions.zeros(backend, Int, nfaces(mesh))
    inner_faces_kernel!(backend, WORKGROUP_SIZE)(
        n_neighbors,
        f2c;
        ndrange = size(n_neighbors),
    )
    return findall(n_neighbors .> 1)
end

struct MyShapeFunction{S, FE, I} <: Bcube.AbstractLazy where {S, FE, I}
    feSpace::FE
    iloc::I
end

function MyShapeFunction(feSpace, iloc)
    MyShapeFunction{get_ncomponents(feSpace), typeof(feSpace), typeof(iloc)}(feSpace, iloc)
end

# This is in Bcube but only for an AbstractLazyOperator and for simplicity I chose to descend from AbstractLazy
(lOp::MyShapeFunction)(x::Vararg{Any, N}) where {N} = Bcube.materialize(lOp, x...)

Bcube.materialize(f::MyShapeFunction, ::Bcube.CellInfo) = f

function Bcube.materialize(f::MyShapeFunction{1}, cPoint::Bcube.CellPoint)
    cInfo = Bcube.get_cellinfo(cPoint)
    cType = Bcube.get_element_type(cInfo)
    cShape = Bcube.shape(cType)
    fs = get_function_space(f.feSpace)
    ξ = get_coords(cPoint)
    return _scalar_shape_functions(fs, cShape, ξ)[f.iloc]
end

function Bcube.materialize(f::MyShapeFunction{N}, cPoint::Bcube.CellPoint) where {N}
    cInfo = Bcube.get_cellinfo(cPoint)
    cType = Bcube.get_element_type(cInfo)
    cShape = Bcube.shape(cType)
    fs = get_function_space(f.feSpace)
    ξ = get_coords(cPoint)
    nc = Val(get_ncomponents(f.feSpace))
    return shape_functions(fs, nc, cShape, ξ)[f.iloc, :]
end

function Bcube.materialize(f::MyShapeFunction, ::Side⁻{Nothing, <:Tuple{FaceInfo}})
    return f
end
function Bcube.materialize(f::MyShapeFunction, ::Side⁺{Nothing, <:Tuple{FaceInfo}})
    return f
end
function Bcube.materialize(f::MyShapeFunction, side::Side⁻{Nothing, <:Tuple{FacePoint}})
    # return NullOperator()
    (f.iloc > 0) && return NullOperator()
    cPoint = side_n(first(get_args(side)))
    Bcube.materialize(MyShapeFunction(f.feSpace, -f.iloc), cPoint)
end
function Bcube.materialize(f::MyShapeFunction, side::Side⁺{Nothing, <:Tuple{FacePoint}})
    # return NullOperator()
    (f.iloc < 0) && return NullOperator()
    cPoint = side_p(first(get_args(side)))
    Bcube.materialize(f, cPoint)
end

export ReverseDofHandler
export test_arg # to be deleted

end