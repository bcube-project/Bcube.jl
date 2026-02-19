# tests for the minimal Gmsh 2.2 writer that was requested by the
# user.  We verify that a mesh written with `write_file` can be read back
# using the existing reader and that basic information (nodes, cells,
# boundary names) is preserved.

using Test
using Bcube

@testset "gmsh22 writer" begin
    # simple mesh without any boundary conditions
    mesh = rectangle_mesh(3, 2)
    outfile = mktemp()[1] * ".msh"
    write_file(outfile, mesh)
    mesh2 = read_mesh(outfile)

    @test nnodes(mesh2) == nnodes(mesh)
    @test ncells(mesh2) == ncells(mesh)

    # now a mesh with explicit boundary names/nodes
    mesh = basic_mesh()
    @test nboundaries(mesh) == 1
    outfile2 = mktemp()[1] * ".msh"
    write_file(outfile2, mesh)
    mesh2 = read_mesh(outfile2)
    @test nnodes(mesh2) == nnodes(mesh)
    @test ncells(mesh2) == ncells(mesh)

    # boundary names should be preserved (the reader uses physical names
    # to create bc_nodes).  we only had one boundary called "BORDER" in
    # `basic_mesh`.
    @test length(boundary_names(mesh2)) == 1
    @test first(boundary_names(mesh2)) == :BORDER
end
