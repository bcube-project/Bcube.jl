using Bcube
using Test
using StaticArrays
using LinearAlgebra
using DelimitedFiles
using SHA: sha1

# from :
# https://discourse.julialang.org/t/what-general-purpose-commands-do-you-usually-end-up-adding-to-your-projects/4889
@generated function compare_struct(x, y)
    if !isempty(fieldnames(x)) && x == y
        mapreduce(n -> :(x.$n == y.$n), (a, b) -> :($a && $b), fieldnames(x))
    else
        :(x == y)
    end
end

"""
Custom way to "include" a file to print infos.
"""
function custom_include(path)
    filename = split(path, "/")[end]
    print("Running test file " * filename * "...")
    include(path)
    println("done.")
end

function Base.isapprox(A::Tuple{SVector{N, R}}, B::Tuple{SVector{N, R}}) where {N, R}
    all(map(isapprox, A, B))
end
function isapprox_arrays(a::AbstractArray, b::AbstractArray; rtol::Real = eps())
    function g(x, y)
        if abs(y) < 10rtol
            isapprox(x, y; rtol = 0, atol = eps())
        else
            isapprox(x, y; rtol = rtol)
        end
    end
    success = all(map(g, a, b))
    (success == false) && (@show a, b)
    return success
end

# This dir will be removed at the end of the tests
tempdir = mktempdir()

@testset "Bcube.jl" begin
    custom_include("./test_utils.jl")
    custom_include("./mesh/test_entity.jl")
    custom_include("./mesh/test_connectivity.jl")
    custom_include("./mesh/test_mesh.jl")
    custom_include("./mesh/test_mesh_generator.jl")
    custom_include("./mesh/test_gmsh.jl")
    custom_include("./mesh/test_domain.jl")
    custom_include("./mapping/test_mapping.jl")
    custom_include("./mapping/test_ref2loc.jl")
    custom_include("./interpolation/test_shape.jl")
    custom_include("./interpolation/test_lagrange.jl")
    custom_include("./interpolation/test_taylor.jl")
    # custom_include("./interpolation/test_projection.jl") # TODO: update with new API
    custom_include("./integration/test_integration.jl")
    custom_include("./dof/test_dofhandler.jl")
    # custom_include("./dof/test_variable.jl")  #TODO : update with new API
    custom_include("./interpolation/test_shapefunctions.jl")
    # custom_include("./interpolation/test_limiter.jl")
    custom_include("./interpolation/test_cellfunction.jl")
    custom_include("./dof/test_assembler.jl")
    custom_include("./operator/test_algebra.jl")
    custom_include("./dof/test_meshdata.jl")
    custom_include("./writers/test_vtk.jl")
end
