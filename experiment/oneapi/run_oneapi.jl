module RunOneAPI
using oneAPI
using KernelAbstractions
include(joinpath(@__DIR__, "..", "BcubeGPU.jl"))
using .BcubeGPU

function run()
    backend = get_backend(oneAPI.ones(Float32, 2))
    BcubeGPU.run(backend)
end

run()
end