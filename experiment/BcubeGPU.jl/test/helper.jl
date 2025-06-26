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

const excluded_files = ("runtests.jl", "helper.jl")

for file in readdir(edir)
    full_path = joinpath(folder_path, file)

    # Skip if not a .jl file
    endswith(file, ".jl") || continue

    # Skip excluded files
    file in excluded_files && continue

    # Include the file
    include(full_path)
end

function run_helper(backend)
    # Linear examples
    run_linear_cell_continuous(backend)
    run_linear_cell_continuous_vector(backend)

    # Bilinear examples
    run_bilinear_cell_continuous(backend)
    run_bilinear_cell_continuous_vector(backend)
end