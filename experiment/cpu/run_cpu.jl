module RunCPU
using KernelAbstractions
include(joinpath(@__DIR__, "..", "BcubeGPU.jl"))
using .BcubeGPU

function run()
    backend = CPU()
    BcubeGPU.run(backend)
end
end