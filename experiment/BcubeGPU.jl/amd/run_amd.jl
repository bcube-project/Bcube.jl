module RunAmd
using AMDGPU
using KernelAbstractions
using BcubeGPU
include(joinpath(@__DIR__, "..", "test", "helper.jl"))
using .Helper

function run()
    backend = get_backend(AMDGPU.ones(2))
    Helper.run_helper(backend)
    # Helper.run_tests(backend)
end

run()
end