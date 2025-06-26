module RunCuda
using CUDA
using KernelAbstractions
include(joinpath(@__DIR__, "..", "BcubeGPU.jl"))
using .BcubeGPU

function run()
    backend = get_backend(CUDA.ones(2))
    BcubeGPU.run(backend)
end

run()
end