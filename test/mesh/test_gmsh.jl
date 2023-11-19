const mesh_dir = string(@__DIR__, "/../../input/mesh/")

@testset "gmsh - domainLine_o1" begin
    mesh = read_msh(mesh_dir * "domainLine_o1.msh")

    @test topodim(mesh) === 1
    @test spacedim(mesh) === 1

    @test nnodes(mesh) === 11
    @test ncells(mesh) === 10

    @test nodes(mesh) == [Node_t() for i in 1:nnodes(mesh)]
    @test cells(mesh) == [Bar2_t() for i in 1:ncells(mesh)]
end

@testset "gmsh - domainSquare_tri" begin
    mesh = read_msh(mesh_dir * "domainSquare_tri.msh")

    @test topodim(mesh) === 2
    @test spacedim(mesh) === 2

    @test has_cells(mesh) === true
    @test has_nodes(mesh) === true
    @test has_nodes(mesh) === has_entities(mesh, :node)
    @test has_vertices(mesh) === has_entities(mesh, :vertex)
    @test has_edges(mesh) === has_entities(mesh, :edge)
    @test has_faces(mesh) === has_entities(mesh, :face)
    @test has_cells(mesh) === has_entities(mesh, :cell)

    @test has_cells(mesh) === true
    @test has_nodes(mesh) === true

    @test nnodes(mesh) === 20
    @test ncells(mesh) === 26
    @test nnodes(mesh) === n_entities(mesh, :node)
    @test nvertices(mesh) === n_entities(mesh, :vertex)
    @test nedges(mesh) === n_entities(mesh, :edge)
    @test nfaces(mesh) === n_entities(mesh, :face)
    @test ncells(mesh) === n_entities(mesh, :cell)

    @test nodes(mesh) == [Node_t() for i in 1:nnodes(mesh)]
    @test cells(mesh) == [Tri3_t() for i in 1:ncells(mesh)]

    @test boundary_tag(mesh, "South") == 1 &&
          boundary_tag(mesh, "East") == 2 &&
          boundary_tag(mesh, "North") == 3 &&
          boundary_tag(mesh, "West") == 4

    @test boundary_names(mesh)[1] == "South" &&
          boundary_names(mesh)[2] == "East" &&
          boundary_names(mesh)[3] == "North" &&
          boundary_names(mesh)[4] == "West"

    ref_nodes_south = Set([1, 5, 6, 2])
    ref_nodes_east = Set([3, 8, 7, 2])
    ref_nodes_north = Set([4, 10, 9, 3])
    ref_nodes_west = Set([4, 12, 11, 1])
    @test Set(absolute_indices(mesh, :node)[boundary_nodes(mesh, 1)]) == ref_nodes_south
    @test Set(absolute_indices(mesh, :node)[boundary_nodes(mesh, 2)]) == ref_nodes_east
    @test Set(absolute_indices(mesh, :node)[boundary_nodes(mesh, 3)]) == ref_nodes_north
    @test Set(absolute_indices(mesh, :node)[boundary_nodes(mesh, 4)]) == ref_nodes_west

    for tag in 1:4
        _nodes = [
            i for face in boundary_faces(mesh, tag) for
            i in connectivities_indices(mesh, :f2n)[face]
        ]
        if tag == 1
            (@test Set(absolute_indices(mesh, :node)[_nodes]) == ref_nodes_south)
        else
            nothing
        end
        if tag == 2
            (@test Set(absolute_indices(mesh, :node)[_nodes]) == ref_nodes_east)
        else
            nothing
        end
        if tag == 3
            (@test Set(absolute_indices(mesh, :node)[_nodes]) == ref_nodes_north)
        else
            nothing
        end
        if tag == 4
            (@test Set(absolute_indices(mesh, :node)[_nodes]) == ref_nodes_west)
        else
            nothing
        end
    end

    # for (tag,name) in boundary_names(mesh)
    #     @show tag,name
    #     for face in boundary_faces(mesh)[tag]
    #         #@show connectivities_indices(mesh,:f2n)[face]
    #         @show [coordinates(mesh,i) for i in connectivities_indices(mesh,:f2n)[face]]
    #     end
    #     @show [coordinates(mesh,i) for i in boundary_nodes(mesh,tag)]
    # end

    @testset "gmsh - autocompute space dim" begin
        mesh = read_msh(mesh_dir * "domainLine_o1.msh")
        @test spacedim(mesh) === 1

        mesh = read_msh(mesh_dir * "domainLine_o1.msh", 3) # without autocompute
        @test spacedim(mesh) === 3

        mesh = read_msh(mesh_dir * "domainSquare_tri.msh")
        @test spacedim(mesh) === 2

        mesh = read_msh(mesh_dir * "domainSquare_tri.msh", 3) # without autocompute
        @test spacedim(mesh) === 3

        path = joinpath(tempdir, "mesh.msh")
        Bcube.gen_sphere_mesh(path)
        mesh = read_msh(path)
        @test spacedim(mesh) === 3
    end
end

@testset "gmsh - generators" begin
    basename = "gmsh_line_mesh"
    path = joinpath(tempdir, basename * ".msh")

    n_partitions = 3
    gen_line_mesh(
        path;
        nx = 11,
        n_partitions = n_partitions,
        split_files = true,
        create_ghosts = true,
    )

    for i in 1:n_partitions
        fname = basename * "_$i.msh"
        @test fname2sum[fname] == bytes2hex(open(sha1, joinpath(tempdir, fname)))
    end
end
