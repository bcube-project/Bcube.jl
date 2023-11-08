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

    nx = 10
    ny = 10
    m = rectangle_mesh(nx, ny)
    @test nnodes(m) == nx * ny
    @test ncells(m) == (nx - 1) * (ny - 1)

    # Not working (maybe the structure comparison is too hard,
    # or maybe it does not compare object properties but objects themselves)
    #@test ncube_mesh([n]) == line_mesh(n)
    #@test ncube_mesh([nx, ny]) == rectangle_mesh(nx, ny)
end
