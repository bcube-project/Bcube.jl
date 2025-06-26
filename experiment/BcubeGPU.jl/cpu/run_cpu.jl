module RunCPU
using KernelAbstractions
using BcubeGPU
include(joinpath(@__DIR__, "..", "test", "helper.jl"))

function run()
    backend = CPU()
    run_helper(backend)
end

run()
end