const excluded_files = ("runtests.jl", "helper.jl", "utils.jl")

function list_examples_files(dir)
    filepaths = String[]
    for file in readdir(dir)
        endswith(file, ".jl") || continue
        file in excluded_files && continue
        push!(filepaths, file)
    end
    return filepaths
end