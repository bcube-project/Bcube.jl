module Helper
using BcubeGPU
using KernelAbstractions
using DelimitedFiles
using LinearAlgebra
using BcubeGmsh
using Bcube
using Adapt
using SparseArrays

# this file is a helper to select the example(s) we want to run by
# exposing a single "run" function. It avoids modifying the
# "run_cuda.jl", "run_cpu.jl", etc every time.

include(joinpath(@__DIR__, "utils.jl"))
const filepaths = list_examples_files(@__DIR__)
for file in filepaths
    full_path = joinpath(@__DIR__, file)
    println("Including file $(full_path)")
    include(full_path)
end

function run_helper(backend)
    # Linear examples
    # run_linear_cell_continuous(backend)
    # run_linear_cell_continuous_vector(backend)
    run_linear_face_discontinuous(backend)

    # Bilinear examples
    # run_bilinear_cell_continuous(backend)
    # run_bilinear_cell_continuous_vector(backend)
end

function run_tests(backend)
    for file in filepaths

        # Extract base name (e.g., file.jl -> file)
        base_name = first(splitext(file))

        # Create function name symbol (e.g., run-file)
        func_name = Symbol("run_", base_name)

        f = getfield(Helper, func_name)
        x = f(backend)
        success = x.res_cpu == x.res_from_gpu
        if success
            @info "$(base_name) test passed"
        else
            @warn "$(base_name) test failed"
        end
    end
end
end