using Test
using BcubeGPU
using KernelAbstractions
using DelimitedFiles
using LinearAlgebra
using BcubeGmsh
using Bcube
using Adapt
using SparseArrays

include(joinpath(@__DIR__, "utils.jl"))

const backend = CPU()
const dir = @__DIR__

const filepaths = list_examples_files(dir)

@testset "BcubeGPU.jl" begin
    for file in filepaths
        # Include the file
        include(joinpath(dir, file))

        # Extract base name (e.g., file.jl -> file)
        base_name = first(splitext(file))

        # Create function name symbol (e.g., run-file)
        func_name = Symbol("run_", base_name)

        # Check if the function exists in Main and is callable
        if isdefined(Main, func_name) && isa(getfield(Main, func_name), Function)
            println("Running function: ", func_name)
            f = getfield(Main, func_name)
            @testset "$base_name" begin
                x = f(backend)
                @test x.res_cpu == x.res_from_gpu
            end
        else
            @warn "Function $(func_name) not found or not callable"
        end
    end
end
