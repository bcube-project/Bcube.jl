module BcubeGPU
# using Cthulhu
using Bcube
using StaticArrays
using KernelAbstractions
using Adapt
using SparseArrays

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

function Adapt.adapt_structure(
    to,
    conn::Bcube.MeshConnectivity{C, F, T, B},
) where {C, F, T, B}
    layers = adapt(to, Bcube.nlayers(conn))
    ind = adapt(to, Bcube.indices(conn))
    Bcube.MeshConnectivity{typeof(ind), F, T, B, typeof(layers)}(layers, ind)
end

function Adapt.adapt_structure(to, feSpace::Bcube.SingleFESpace{S, FS}) where {S, FS}
    dhl = adapt(to, Bcube._get_dhl(feSpace))
    tags = adapt(to, Bcube.get_dirichlet_boundary_tags(feSpace))
    Bcube.SingleFESpace{S, FS, typeof(dhl), typeof(tags)}(
        Bcube.get_function_space(feSpace),
        dhl,
        Bcube.is_continuous(feSpace),
        tags,
    )
end

Adapt.@adapt_structure Bcube.Connectivity
Adapt.@adapt_structure Bcube.TestFESpace
Adapt.@adapt_structure Bcube.TrialFESpace
Adapt.@adapt_structure Bcube.SingleFieldFEFunction
Adapt.@adapt_structure Bcube.DofHandler
Adapt.@adapt_structure ReverseDofHandler
#<<<<<<<< Adapt some structures

"""
Build the connectivity <global dof index> -> <cells surrounding this dof, local index of this dof
in the cells>
"""
function build_dof_to_cells(mesh, U)
    dof_to_cells = [Tuple{Int, Int}[] for _ in 1:get_ndofs(U)]
    dhl = Bcube._get_dhl(U)
    # dof_to_cells = [Int[] for _ in 1:get_ndofs(U)]
    for icell in 1:ncells(mesh)
        # foreach(idof -> push!(dof_to_cells[idof], icell), Bcube.get_dofs(U, icell))
        for iloc in 1:get_ndofs(dhl, icell)
            idof = Bcube.get_dof(dhl, icell, 1, iloc) # comp = 1
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
    fs = Bcube.get_function_space(f.feSpace)
    ξ = Bcube.get_coords(cPoint)
    return Bcube._scalar_shape_functions(fs, cShape, ξ)[f.iloc]
end

@kernel function gpu_assemble_kernel!(
    b,
    @Const(f),
    @Const(elts),
    @Const(V),
    @Const(quadrature),
    @Const(rdhl)
)
    I = @index(Global)

    offset = rdhl.offset[I]
    for i in 1:rdhl.ncells[I]
        icell = rdhl.icells[offset + i]
        iloc = rdhl.iloc[offset + i]
        eltInfo = elts[icell]

        φ = MyShapeFunction(V, iloc)
        fᵥ = Bcube.materialize(f(φ), eltInfo)
        value = Bcube.integrate_on_ref_element(fᵥ, eltInfo, quadrature)
        b[I] += value
    end
end

function gpu_assemble!(backend, y, f, elts, V, measure, rdhl)
    quadrature = Bcube.get_quadrature(measure)
    gpu_assemble_kernel!(backend, WORKGROUP_SIZE)(
        y,
        f,
        elts,
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

struct Toto{S}
    a::S
end

function run(backend)
    mesh = rectangle_mesh(2, 3)
    mesh_gpu = adapt(backend, mesh)
    test_arg(backend, adapt(backend, mesh_gpu.nodes)) # OK
    test_arg(backend, adapt(backend, mesh_gpu.entities)) # OK
    connectivities = mesh_gpu.connectivities
    c2n = connectivities[:c2n]
    @show typeof(c2n)
    c2n_indices = c2n.indices
    test_arg(backend, adapt(backend, c2n_indices)) # OK
    @show typeof(c2n_indices)
    test_arg(backend, adapt(backend, c2n))
    error("dbg")

    Ω = CellDomain(mesh)
    dΩ = Measure(Ω, 1)

    cells_cpu = map(Bcube.DomainIterator(Ω)) do cellInfo
        icell = Bcube.cellindex(cellInfo)
        ctype = Bcube.celltype(cellInfo)
        nodes = SA[Bcube.nodes(cellInfo)...]
        c2n = SA[Bcube.get_nodes_index(cellInfo)...]

        Bcube.CellInfo(icell, ctype, nodes, c2n)
    end
    cells = adapt(backend, cells_cpu)

    g(x, t) = 3x[1]
    h(x, t) = 5x[1] + 2

    U_cpu = TrialFESpace(
        FunctionSpace(:Lagrange, 1),
        mesh,
        Dict("xmin" => 5.0, "xmax" => g, "ymin" => h),
    )
    V_cpu = TestFESpace(U_cpu)
    rdhl_cpu = ReverseDofHandler(mesh, U_cpu)

    U = adapt(backend, U_cpu)
    V = adapt(backend, V_cpu)
    rdhl = adapt(backend, rdhl_cpu)

    @show typeof(U)
    @show typeof(V)
    u = FEFunction(U, KernelAbstractions.ones(backend, Float64, get_ndofs(U)))
    f(φ) = u * φ
    # f(φ) = PhysicalFunction(x -> 1.0) * φ
    y = KernelAbstractions.zeros(backend, Float64, get_ndofs(U))
    gpu_assemble!(backend, y, f, cells, V, dΩ, rdhl)
    display(y)
    # test_arg(backend, Bcube.get_quadrature(dΩ))

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end

end