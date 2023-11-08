push!(LOAD_PATH, "../src/")

using Bcube
using Documenter
using Literate

# Alias for `Literate.markdown`
function gen_markdown(src, name, dir)
    Literate.markdown(joinpath(src, name), dir; documenter = false, execute = false)
end

"""
Build a markdown file with just the content of the julia file in it.
"""
function julia_to_markdown(src_dir, target_dir, filename, title)
    open(joinpath(target_dir, split(filename, ".")[1] * ".md"), "w") do io
        println(io, "# " * title)
        println(io, "```julia")
        f = open(joinpath(src_dir, filename), "r")
        lines = readlines(f)
        close(f)
        map(line -> println(io, line), lines)
        println(io, "```")
    end
end

# Generate tutorials
# `documenter = false` to avoid Documenter to execute cells
tutorial_names =
    ["helmholtz", "heat_equation", "linear_transport", "phase_field_supercooled"]
tutorial_src = joinpath(@__DIR__, "..", "tutorial")
tutorial_dir = joinpath(@__DIR__, "src", "tutorial")
Sys.rm(tutorial_dir; recursive = true, force = true)
map(filename -> gen_markdown(tutorial_src, "$(filename).jl", tutorial_dir), tutorial_names)

# Generate "commented" examples
# `documenter = false` to avoid Documenter to execute cells
example_src = joinpath(@__DIR__, "..", "example")
example_dir = joinpath(@__DIR__, "src", "example")
Sys.rm(example_dir; recursive = true, force = true)
mkdir(example_dir)
# gen_markdown(example_src, "euler_naca_steady.jl", example_dir)
# gen_markdown(example_src, "covo.jl", example_dir)
# gen_markdown(example_src, "linear_elasticity.jl", example_dir)

# Generate "uncommented" examples
julia_to_markdown(
    example_src,
    example_dir,
    "euler_naca_steady.jl",
    "Euler equations on a NACA0012",
)
julia_to_markdown(example_src, example_dir, "covo.jl", "Euler equations - covo")
julia_to_markdown(example_src, example_dir, "linear_elasticity.jl", "Linear elasticity")
julia_to_markdown(
    example_src,
    example_dir,
    "linear_thermoelasticity.jl",
    "Linear thermo-elasticity",
)

makedocs(;
    modules = [Bcube],
    authors = "Ghislain Blanchard, Lokman Bennani and Maxime Bouyges",
    sitename = "Bcube",
    clean = true,
    doctest = false,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://bcube-project.github.io/Bcube.jl",
        assets = String[],
    ),
    checkdocs = :none,
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["tutorial/$(filename).md" for filename in tutorial_names],
        "Advanced examples" => Any[
            "example/covo.md",
            "example/euler_naca_steady.md",
            "example/linear_elasticity.md",
            "example/linear_thermoelasticity.md",
        ],
        "Manual" => Any[
            "manual/geometry.md",
            "manual/integration.md",
            "manual/cellfunction.md",
            "manual/function_space.md",
            "manual/operator.md",
        ],
        "How to..." => "howto/howto.md",
        "API Reference" => Any[
            "api/mesh/mesh.md",
            "api/mesh/gmsh_utils.md",
            "api/mesh/mesh_generator.md",
            "api/interpolation/shape.md",
            "api/interpolation/function_space.md",
            "api/interpolation/spaces.md",
            "api/interpolation/fespace.md",
            "api/mapping/mapping.md",
            "api/integration/integration.md",
            # "api/operator/operator.md",
            "api/dof/dof.md",
            "api/output/vtk.md",
        ],
    ],
)

deploydocs(; repo = "github.com/bcube-project/Bcube.jl.git", push_preview = true)
