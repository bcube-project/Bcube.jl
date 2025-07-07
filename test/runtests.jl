using Bcube
using Test
using StaticArrays
using LinearAlgebra
using DelimitedFiles
using ForwardDiff
using SparseArrays
using WriteVTK

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

import Bcube:
    boundary_faces,
    boundary_nodes,
    by,
    cells,
    cellindex,
    CellInfo,
    CellPoint,
    celltype,
    cell_side,
    center,
    change_domain,
    compute,
    connectivities,
    connectivities_indices,
    Connectivity,
    connectivity_cell2cell_by_faces,
    connectivity_cell2cell_by_nodes,
    Cube,
    get_dof,
    DofHandler,
    DomainIterator,
    edges2nodes,
    FaceInfo,
    FacePoint,
    faces,
    faces2nodes,
    face_area,
    face_shapes,
    from,
    f2n_from_c2n,
    get_coords,
    get_dofs,
    get_fespace,
    get_mapping,
    get_nodes,
    has_cells,
    has_edges,
    has_entities,
    has_faces,
    has_nodes,
    has_vertices,
    indices,
    inner_faces,
    integrate_on_ref_element,
    interpolate,
    inverse_connectivity,
    Line,
    mapping,
    mapping_det_jacobian,
    mapping_face,
    mapping_inv,
    mapping_jacobian,
    mapping_jacobian_inv,
    materialize,
    maxsize,
    max_ndofs,
    Mesh,
    minsize,
    myrand,
    nedges,
    nfaces,
    nlayers,
    nodes,
    normal,
    normals,
    nvertices,
    n_entities,
    oriented_cell_side,
    outer_faces,
    Prism,
    PhysicalDomain,
    ReferenceDomain,
    shape,
    shape_functions,
    spacedim,
    Square,
    Tetra,
    to,
    topodim,
    Triangle,
    ∂λξ_∂ξ,
    ∂λξ_∂x,
    Bar2_t,
    Tri3_t,
    Quad4_t,
    Quad9_t,
    foreach_element

# This dir will be removed at the end of the tests
tempdir = mktempdir()

@testset "Bcube.jl" begin
    custom_include("./test_utils.jl")
    custom_include("./lazyop/test_lazyop.jl")
    custom_include("./mesh/test_entity.jl")
    custom_include("./mesh/test_connectivity.jl")
    custom_include("./mesh/test_transformation.jl")
    custom_include("./mesh/test_mesh.jl")
    custom_include("./mesh/test_mesh_generator.jl")
    custom_include("./mesh/test_domain.jl")
    custom_include("./mapping/test_mapping.jl")
    custom_include("./mapping/test_ref2phys.jl")
    custom_include("./interpolation/test_shape.jl")
    custom_include("./interpolation/test_lagrange.jl")
    custom_include("./interpolation/test_taylor.jl")
    custom_include("./fespace/test_dofhandler.jl")
    custom_include("./fespace/test_fespace.jl")
    custom_include("./fespace/test_fefunction.jl")
    custom_include("./interpolation/test_projection.jl")
    custom_include("./integration/test_integration.jl")
    # custom_include("./dof/test_variable.jl")  #TODO : update with new API
    custom_include("./interpolation/test_shapefunctions.jl")
    # custom_include("./interpolation/test_limiter.jl")
    custom_include("./interpolation/test_cellfunction.jl")
    custom_include("./dof/test_assembler.jl")
    custom_include("./dof/test_dirichlet.jl")
    custom_include("./operator/test_algebra.jl")
    custom_include("./dof/test_meshdata.jl")

    @testset "Issues" begin
        custom_include("./issues/issue_112.jl")
        custom_include("./issues/issue_130.jl")
        custom_include("./issues/issue_101.jl")
    end
end
