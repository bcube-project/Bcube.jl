module BcubeGPU
# using Cthulhu
using Bcube
using StaticArrays
using KernelAbstractions
using Adapt

const WORKGROUP_SIZE = 32

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

function my_line_mesh(n; xmin = 0.0, xmax = 1.0, order = 1, names = ("xmin", "xmax"))
    l = xmax - xmin # line length
    nelts = n - 1 # Number of cells

    Δx = l / (n - 1)

    # Nodes
    nodes = [Node(Float32.([xmin + (i - 1) * Δx])) for i in 1:n]

    # Cell type is constant
    celltypes = [Bcube.Bar2_t() for ielt in 1:nelts]

    # Cell -> nodes connectivity
    cell2node = zeros(Int, 2 * nelts)
    for ielt in 1:nelts
        cell2node[2 * ielt - 1] = ielt
        cell2node[2 * ielt]     = ielt + 1
    end

    # Number of nodes of each cell : always 2
    cell2nnodes = 2 * ones(Int, nelts)

    # Boundaries
    bc_names, bc_nodes = Bcube.one_line_bnd(1, n, names)

    # Mesh
    return Bcube.Mesh(
        nodes,
        celltypes,
        Bcube.Connectivity(cell2nnodes, cell2node);
        bc_names,
        bc_nodes,
    )
end

@kernel function my_kernel(res, @Const(g), @Const(cells), @Const(quadrature))
    icell = @index(Global)
    eltInfo = cells[icell]
    res[icell] = Bcube.integrate_on_ref_element(g, eltInfo, quadrature)
end

function cuda_kernel!(res, g, cells, quadrature)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for icell in index:stride:length(res)
        eltInfo = cells[icell]
        res[icell] = Bcube.integrate_on_ref_element(g, eltInfo, quadrature)
    end
    return nothing
end

function run_my_kernel(backend, res, g, cells, quadrature)
    my_kernel(backend, WORKGROUP_SIZE)(res, g, cells, quadrature; ndrange = length(cells))
    KernelAbstractions.synchronize(backend)
    return res
end

function run(backend)
    mesh = my_line_mesh(3)

    Ω = CellDomain(mesh)

    cells_cpu = map(Bcube.DomainIterator(Ω)) do cellInfo
        icell = Bcube.cellindex(cellInfo)
        ctype = Bcube.celltype(cellInfo)
        nodes = SA[Bcube.nodes(cellInfo)...]
        c2n = SA[Bcube.get_nodes_index(cellInfo)...]

        Bcube.CellInfo(icell, ctype, nodes, c2n)
    end
    cells = adapt(backend, cells_cpu)

    quadrature = Quadrature(1)

    res = KernelAbstractions.zeros(backend, Float32, ncells(mesh))

    g = PhysicalFunction(x -> 1.0)
    run_my_kernel(backend, res, g, cells, quadrature)
    display(res)

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end

end