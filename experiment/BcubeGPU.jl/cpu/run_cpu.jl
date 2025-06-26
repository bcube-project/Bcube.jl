module RunCPU
using KernelAbstractions
using BcubeGPU
include(joinpath(@__DIR__, "..", "test", "helper.jl"))
using .Helper

function run()
    backend = CPU()
    Helper.run_helper(backend)
    # Helper.run_tests(backend)
end

run()
end