module BcubeGPU
# using Cthulhu
using Bcube
using StaticArrays
using KernelAbstractions
using Adapt
using SparseArrays
import Bcube:
    Connectivity,
    DofHandler,
    Mesh,
    MeshConnectivity,
    SingleFESpace,
    SingleFieldFEFunction,
    boundary_faces,
    boundary_nodes,
    connectivities,
    entities,
    get_coords,
    get_dof,
    get_dirichlet_boundary_tags,
    get_function_space,
    get_metadata,
    get_quadrature,
    indices,
    integrate_on_ref_element,
    is_continuous,
    nlayers,
    _get_dhl,
    _get_index,
    _scalar_shape_functions

const WORKGROUP_SIZE = 32

struct ReverseDofHandler{A, B, C, D}
    offset::A # 1:ndofs
    ncells::B # 1:ndofs number of cells owning each dof
    icells::C # icells[offset[idof]:offset[idof]+ncells[idof]] are cells surrounding idof
    iloc::D # iloc[offset[idof]:offset[idof]+ncells[idof]] are local indices of the dof in the corresponding cell
end

function ReverseDofHandler(mesh, U)
    dof_to_cells = build_dof_to_cells(mesh, U)
    ndofs = get_ndofs(U)
    offset = zeros(Int, ndofs)
    ncells = zeros(Int, ndofs)
    n = sum(length.(dof_to_cells))
    icells = zeros(Int, n)
    iloc = zeros(Int, n)
    curr = 1
    for (idof, x) in enumerate(dof_to_cells)
        ncells[idof] = length(x)
        for (icell, _iloc) in x
            icells[curr] = icell
            iloc[curr] = _iloc
            curr += 1
        end
        (idof > 1) && (offset[idof] = offset[idof - 1] + ncells[idof - 1])
    end
    return ReverseDofHandler(offset, ncells, icells, iloc)
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

"""
Build the connectivity <global dof index> -> <cells surrounding this dof, local index of this dof
in the cells>
"""
function build_dof_to_cells(mesh, U)
    dof_to_cells = [Tuple{Int, Int}[] for _ in 1:get_ndofs(U)]
    dhl = _get_dhl(U)
    # dof_to_cells = [Int[] for _ in 1:get_ndofs(U)]
    for icell in 1:ncells(mesh)
        # foreach(idof -> push!(dof_to_cells[idof], icell), Bcube.get_dofs(U, icell))
        for iloc in 1:get_ndofs(dhl, icell)
            idof = get_dof(dhl, icell, 1, iloc) # comp = 1
            push!(dof_to_cells[idof], (icell, iloc))
        end
    end
    return dof_to_cells
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

@kernel function gpu_assemble_kernel!(
    b,
    @Const(f),
    @Const(domain),
    @Const(V),
    @Const(quadrature),
    @Const(rdhl)
)
    # Here  `I` is a global index of a dof
    I = @index(Global)

    offset = rdhl.offset[I]
    for i in 1:rdhl.ncells[I]
        icell = rdhl.icells[offset + i]
        iloc = rdhl.iloc[offset + i]
        eltInfo = _get_index(domain, icell)

        φ = MyShapeFunction(V, iloc)
        fᵥ = Bcube.materialize(f(φ), eltInfo)
        value = integrate_on_ref_element(fᵥ, eltInfo, quadrature)
        b[I] += value
    end
end

function gpu_assemble!(backend, y, f, V, measure, rdhl)
    quadrature = get_quadrature(measure) # not sure if it's needed here
    domain = get_domain(measure)
    gpu_assemble_kernel!(backend, WORKGROUP_SIZE)(
        y,
        f,
        domain,
        V,
        quadrature,
        rdhl;
        ndrange = size(y),
    )
end

@kernel function test_arg_kernel(x, @Const(arg))
    I = @index(Global)
    x[I] += 1
end

function test_arg(backend, arg)
    x = KernelAbstractions.zeros(backend, Float32, 10)
    test_arg_kernel(backend, WORKGROUP_SIZE)(x, arg; ndrange = size(x))
end

function run(backend)
    # Mesh and domains
    mesh_cpu = rectangle_mesh(2, 3)
    mesh = adapt(backend, mesh_cpu)
    test_arg(backend, mesh)
    println("mesh on GPU!")

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

    # Build ReverseDofHandler : it could be part of the (direct)DofHandler
    rdhl_cpu = ReverseDofHandler(mesh, U_cpu)
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
    gpu_assemble!(backend, y, f, V, dΩ, rdhl)
    display(y)

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end

end