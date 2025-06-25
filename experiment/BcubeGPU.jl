module BcubeGPU
# using Cthulhu
# using CUDA # just for debug, to be commented once solved
using Bcube
using StaticArrays
using KernelAbstractions
using GPUArrays # just to access the type AbstractGPUArray for dispatch of `inner_faces`
using Adapt
using SparseArrays
using InteractiveUtils
# using BcubeGmsh # TMP
include(joinpath(@__DIR__, "misc.jl"))
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
    get_metadata,
    get_quadrature,
    idof_by_face_with_bounds,
    indices,
    integrate_on_ref_element,
    is_continuous,
    nfaces,
    nlayers,
    shape,
    _get_dhl,
    _get_index,
    _scalar_shape_functions

const WORKGROUP_SIZE = 32

"""
Structure representing a sparse Matrix with non-empty rows, meaning that each
row contains at least on element.

`offset` starts at 0 : `offset[1] = 0`
"""
struct DenseRowsSparseCols{O, D}
    offset::O # 1:n
    values::D # values[offset[i]+1:offset[i]+offset[i+1]]
end

function get_n_elts(x::DenseRowsSparseCols, i)
    n = x.offset[i]
    m = (i < l) ? offset[i + 1] : l
    return m - n + 1
end

get_elt(x::DenseRowsSparseCols, i, j) = x.values[offset[i] + j]

"""
This structure actually represents two "sparse" matrices. The first sparse matrix is (idof, ielt), the
second is (idof, iloc).
"""
struct ReverseDofHandler{A, B, C, D}
    offset::A # 1:ndofs
    nelts::B # 1:ndofs number of elts owning each dof
    ielts::C # ielts[offset[idof]:offset[idof]+nelts[idof]] are elements surrounding idof
    iloc::D # iloc[offset[idof]:offset[idof]+nelts[idof]] are local indices of the dof in the corresponding element
end

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
    return ReverseDofHandler(offset, nelts, ielts, iloc)
end

"""
Build the dof -> cell ReverseDofHandler

Build the connectivity <global dof index> -> <cells surrounding this dof, local index of this dof
in the cells>
"""
function ReverseDofHandler(domain::AbstractCellDomain, U)
    dof_to_cells = [Tuple{Int, Int}[] for _ in 1:get_ndofs(U)]
    dhl = _get_dhl(U)
    for cInfo in DomainIterator(domain)
        icell = get_element_index(cInfo)
        for iloc in 1:get_ndofs(dhl, icell)
            idof = get_dof(dhl, icell, 1, iloc) # comp = 1
            push!(dof_to_cells[idof], (icell, iloc))
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

struct MyShapeFunction{FE, I} <: Bcube.AbstractLazy where {FE, I}
    feSpace::FE
    iloc::I
end

Bcube.materialize(f::MyShapeFunction, ::Bcube.CellInfo) = f

function Bcube.materialize(f::MyShapeFunction, cPoint::Bcube.CellPoint)
    cInfo = Bcube.get_cellinfo(cPoint)
    cType = Bcube.get_element_type(cInfo)
    cShape = Bcube.shape(cType)
    fs = get_function_space(f.feSpace)
    ξ = get_coords(cPoint)
    return _scalar_shape_functions(fs, cShape, ξ)[f.iloc]
end

function Bcube.materialize(f::MyShapeFunction, side::Side⁻{Nothing, <:Tuple{FaceInfo}})
    # error("passage materialize side_n(MyShapeFunction, finfo)")
    return f
end
function Bcube.materialize(f::MyShapeFunction, side::Side⁺{Nothing, <:Tuple{FaceInfo}})
    # error("passage materialize side_p(MyShapeFunction, finfo)")
    return f
end
function Bcube.materialize(f::MyShapeFunction, side::Side⁻{Nothing, <:Tuple{FacePoint}})
    # return NullOperator()
    (f.iloc > 0) && return 0.0
    cPoint = side_n(first(get_args(side)))
    Bcube.materialize(MyShapeFunction(f.feSpace, -f.iloc), cPoint)
end
function Bcube.materialize(f::MyShapeFunction, side::Side⁺{Nothing, <:Tuple{FacePoint}})
    # return NullOperator()
    (f.iloc < 0) && return 0.0
    cPoint = side_p(first(get_args(side)))
    Bcube.materialize(f, cPoint)
end

custom_get_shape_function(::CellInfo, V, iloc) = MyShapeFunction(V, iloc)
function custom_get_shape_function(fInfo::FaceInfo, V, iloc)
    φ = MyShapeFunction(V, iloc)
    return φ

    # cInfo = (iloc < 0) ? get_cellinfo_n(fInfo) : get_cellinfo_p(fInfo)
    # cshape = shape(get_element_type(cInfo))
    # φ = get_cell_shape_functions(V, cshape)[abs(iloc)]

    # fsp = if (iloc > 0)
    #     FaceSidePair(NullOperator(), LazyMapOver(φ))
    # else
    #     FaceSidePair(LazyMapOver(φ), NullOperator())
    # end
    # fsp = if (iloc > 0)
    #     FaceSidePair(LazyMapOver(NullOperator()), LazyMapOver(φ))
    # else
    #     FaceSidePair(LazyMapOver(φ), LazyMapOver(NullOperator()))
    # end
    # fsp = (iloc > 0) ? FaceSidePair(NullOperator(), φ) : FaceSidePair(φ, NullOperator())
    # return fsp
    # return LazyMapOver((fsp,))
end

function assemble_linear_elemental!(idof, b, f, domain, V, quadrature, rdhl)
    offset = rdhl.offset[idof]
    for i in 1:rdhl.nelts[idof]
        ielt = rdhl.ielts[offset + i]
        iloc = rdhl.iloc[offset + i]
        eltInfo = _get_index(domain, ielt)

        φ = MyShapeFunction(V, iloc)
        fᵥ = Bcube.materialize(f(φ), eltInfo)
        value = integrate_on_ref_element(fᵥ, eltInfo, quadrature)
        b[idof] += value
    end
end

"""
In this version, the parallelization is only performed on the rows of the matrix (A[i,:]) not
the elements (A[i,j])
"""
function assemble_bilinear_elemental_v2!(
    idof,
    _I,
    _J,
    _V,
    f,
    domain,
    U,
    V,
    quadrature,
    rdhl_U,
    rhdl_V,
)
    offset_V = rdhl_V.offset[idof]
    dhl_U = _get_dhl(U)

    # Loop on elements "surrounding" idof
    for i in 1:rdhl.nelts[idof]
        ielt = rdhl_V.ielts[offset_V + i]
        iloc = rdhl.iloc[offset + i]
        eltInfo = _get_index(domain, ielt)
        φi = MyShapeFunction(V, iloc)

        # Loop on dofs of U in this cell
        # Warning : this is only valid for CellDomain assembly!
        for (jloc, jdof) in enumerate(get_dof(dhl_U, get_element_index(eltInfo), 1)) # Warning icomp=1
            φj = MyShapeFunction(U, jloc)
            fᵤᵥ = Bcube.materialize(f(φj, φi), eltInfo)
            value = integrate_on_ref_element(fᵤᵥ, eltInfo, quadrature)
            # TODO : store in _I, _J, _V
        end
    end
end

"""
In this version, the parallelization is performed on the elements (A[i,j]).
This function must not be called for all idof ∈ V, jdof ∈ U, but only for
(idof,jdof) sharing at least on element
"""
function assemble_bilinear_elemental_v1!(
    idof,
    jdof,
    _I,
    _J,
    _V,
    f,
    domain,
    U,
    V,
    quadrature,
    rdhl_U,
    rhdl_V,
)
    offset_V = rdhl_V.offset[idof]
    # Loop on elements "surrounding" idof
    for i in 1:rdhl.nelts[idof]
        ielt = rdhl_V.ielts[offset_V + i]
    end
end

@kernel function assemble_linear_kernel!(
    b,
    @Const(f),
    @Const(domain),
    @Const(V),
    @Const(quadrature),
    @Const(rdhl)
)
    # Here  `I` is a global index of a dof
    I = @index(Global)

    assemble_linear_elemental!(I, b, f, domain, V, quadrature, rdhl)
end

function kernabs_assemble_linear!(backend, y, f, V, measure, rdhl)
    quadrature = get_quadrature(measure) # not sure if it's needed here
    domain = get_domain(measure)

    # @code_warntype test_gpu_assemble_kernel!(1, y, f, domain, V, quadrature, rdhl)
    # error("dbg")

    assemble_linear_kernel!(backend, WORKGROUP_SIZE)(
        y,
        f,
        domain,
        V,
        quadrature,
        rdhl;
        ndrange = size(y),
    )
end

function run_linear_cell_continuous(backend)
    # Mesh and domains
    mesh_cpu = rectangle_mesh(2, 3)
    mesh = adapt(backend, mesh_cpu)
    test_arg(backend, mesh)
    println("mesh on GPU!")

    Ω_cpu = CellDomain(mesh_cpu)
    Ω = CellDomain(mesh)
    test_arg(backend, Ω)
    println("Ω on GPU!")

    dΩ = Measure(Ω, 1)
    test_arg(backend, dΩ)
    println("dΩ on GPU!")

    # Build TrialFESpace and TestFESpace
    # The TrialFESpace must be first built on the CPU for now because the
    # underlying DofHandler constructor uses scalar indexing
    g(x, t) = 3x[1]
    h(x, t) = 5x[1] + 2

    U_cpu = TrialFESpace(
        FunctionSpace(:Lagrange, 1),
        mesh_cpu,
        Dict("xmin" => 5.0, "xmax" => g, "ymin" => h),
    )
    U = adapt(backend, U_cpu)
    test_arg(backend, U)
    println("U on GPU!")

    V = TestFESpace(U)
    test_arg(backend, V)
    println("V on GPU!")

    # Build ReverseDofHandler
    rdhl_cpu = ReverseDofHandler(Ω_cpu, U_cpu)
    rdhl = adapt(backend, rdhl_cpu)
    test_arg(backend, rdhl)
    println("rdhl on GPU!")

    # Build FEFunction
    u = FEFunction(U, KernelAbstractions.ones(backend, Float64, get_ndofs(U)))
    test_arg(backend, u)
    println("u on GPU!")

    # Define linear form and assemble
    f(φ) = u * φ
    # f(φ) = PhysicalFunction(x -> 1.0) * φ
    y = KernelAbstractions.zeros(backend, Float64, get_ndofs(U))
    kernabs_assemble_linear!(backend, y, f, V, dΩ, rdhl)
    display(y)

    # Compare with CPU result
    u_cpu = FEFunction(U_cpu, ones(get_ndofs(U)))
    f_cpu(φ) = ∫(u_cpu * φ)Measure(CellDomain(mesh_cpu), 1)
    println("Result on CPU:")
    display(assemble_linear(f_cpu, TestFESpace(U_cpu)))

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end

function run(backend)
    # Mesh and domains
    mesh_cpu = rectangle_mesh(3, 2)
    mesh = adapt(backend, mesh_cpu)
    test_arg(backend, mesh)
    println("mesh on GPU!")

    Γ_cpu = InteriorFaceDomain(mesh_cpu)
    Γ = InteriorFaceDomain(mesh)
    test_arg(backend, Γ)
    println("Γ on GPU!")

    dΓ = Measure(Γ, 1)
    test_arg(backend, dΓ)
    println("dΓ on GPU!")

    # Build TrialFESpace and TestFESpace
    # The TrialFESpace must be first built on the CPU for now because the
    # underlying DofHandler constructor uses scalar indexing
    g(x, t) = 3x[1]
    h(x, t) = 5x[1] + 2

    U_cpu = TrialFESpace(
        FunctionSpace(:Lagrange, 1),
        mesh_cpu,
        :discontinuous,
        Dict("xmin" => 5.0, "xmax" => g, "ymin" => h),
    )
    U = adapt(backend, U_cpu)
    test_arg(backend, U)
    println("U on GPU!")

    V = TestFESpace(U)
    test_arg(backend, V)
    println("V on GPU!")

    # Build ReverseDofHandler
    rdhl_cpu = ReverseDofHandler(Γ_cpu, U_cpu)
    rdhl = adapt(backend, rdhl_cpu)
    test_arg(backend, rdhl)
    println("rdhl on GPU!")

    # Build FEFunction
    u = FEFunction(U, KernelAbstractions.ones(backend, Float64, get_ndofs(U)))
    test_arg(backend, u)
    println("u on GPU!")

    # Define linear form and assemble
    f(φ) = side_n(u) * jump(φ)
    y = KernelAbstractions.zeros(backend, Float64, get_ndofs(U))
    kernabs_assemble_linear!(backend, y, f, V, dΓ, rdhl)
    display(y)

    # Compare with CPU result
    u_cpu = FEFunction(U_cpu, ones(get_ndofs(U)))
    f_cpu(φ) = ∫(side_n(u_cpu) * jump(φ))Measure(InteriorFaceDomain(mesh_cpu), 1)
    println("Result on CPU:")
    display(assemble_linear(f_cpu, TestFESpace(U_cpu)))

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end

end