push!(LOAD_PATH, "../src/")

using Bcube
using Documenter
using Literate

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
        "Manual" => Any[
            "manual/conventions.md",
            "manual/geometry.md",
            "manual/integration.md",
            "manual/cellfunction.md",
            "manual/function_space.md",
            "manual/operator.md",
        ],
        "API Reference" => Any[
            "api/mesh/mesh.md",
            "api/mesh/gmsh_utils.md",
            "api/mesh/mesh_generator.md",
            "api/interpolation/shape.md",
            "api/interpolation/function_space.md",
            "api/interpolation/fespace.md",
            "api/mapping/mapping.md",
            "api/integration/integration.md",
            # "api/operator/operator.md",
            "api/dof/dof.md",
            "api/output/vtk.md",
        ],
        "How to... (FAQ)" => "howto/howto.md",
    ],
)

deploydocs(; repo = "github.com/bcube-project/Bcube.jl.git", push_preview = true)
