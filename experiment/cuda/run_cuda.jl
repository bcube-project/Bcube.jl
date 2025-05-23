module RunCuda
using CUDA
using KernelAbstractions
include(joinpath(@__DIR__, "..", "BcubeGPU.jl"))
using .BcubeGPU

function run()
    backend = get_backend(CUDA.ones(ncells(mesh)))
    BcubeGPU.run(backend)
end
end