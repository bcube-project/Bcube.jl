@testset "mesh generator" begin
    # These are more "functionnal tests" than "unit test" : we check that no error is committed

    m = basic_mesh()
    @test nnodes(m) == 7
    @test ncells(m) == 3

    m = one_cell_mesh(:line)
    @test nnodes(m) == 2
    @test ncells(m) == 1

    m = one_cell_mesh(:line; order = 2)
    @test nnodes(m) == 3
    @test ncells(m) == 1

    n = 10
    m = line_mesh(n)
    @test nnodes(m) == n
    @test ncells(m) == n - 1

    m = line_mesh(n; xmin = 1.0, xmax = 3.0, order = 2)
    @test nnodes(m) == 2n - 1
    @test ncells(m) == n - 1
    @test all(ct -> ct isa Bar3_t, cells(m))
    x(i) = get_coords(get_nodes(m, i))[1]
    c2n = connectivities_indices(m, :c2n)
    # vertices then middle node, consecutive cells
    @test all(i -> x(c2n[i][1]) < x(c2n[i][3]) < x(c2n[i][2]), 1:ncells(m))
    @test all(i -> x(c2n[i][3]) ≈ (x(c2n[i][1]) + x(c2n[i][2])) / 2, 1:ncells(m))
    @test all(i -> c2n[i + 1][1] == c2n[i][2], 1:(ncells(m) - 1))
    @test x.(boundary_nodes(m, Bcube.boundary_tag(m, "xmin"))) == [1.0]
    @test x.(boundary_nodes(m, Bcube.boundary_tag(m, "xmax"))) ≈ [3.0]
    @test sum(compute(∫(PhysicalFunction(_ -> 1.0))Measure(CellDomain(m), 2))) ≈ 2.0

    nx = 10
    ny = 10
    m = rectangle_mesh(nx, ny)
    @test nnodes(m) == nx * ny
    @test ncells(m) == (nx - 1) * (ny - 1)

    nx = 3
    ny = 4
    nz = 5
    mesh = hexa_mesh(nx, ny, nz)
    @test nnodes(mesh) == nx * ny * nz
    @test ncells(mesh) == (nx - 1) * (ny - 1) * (nz - 1)
    @test length(boundary_faces(mesh, "xmin")) == (ny - 1) * (nz - 1)
    @test length(boundary_faces(mesh, "ymin")) == (nx - 1) * (nz - 1)

    # Not working (maybe the structure comparison is too hard,
    # or maybe it does not compare object properties but objects themselves)
    #@test ncube_mesh([n]) == line_mesh(n)
    #@test ncube_mesh([nx, ny]) == rectangle_mesh(nx, ny)
end
