module gpu1
using Cthulhu
using Bcube
using StaticArrays
using CUDA
using KernelAbstractions
using Adapt

const WORKGROUP_SIZE = 32

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

function run()
    mesh = line_mesh(3)

    backend_cuda = get_backend(CUDA.ones(ncells(mesh)))
    backend_cpu = CPU()

    # backend = backend_cpu
    backend = backend_cuda

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

run()
end