@testset "entity types" begin

    # Bar2
    bar = Bar2_t()
    @test nnodes(bar) === 2
    @test nodes(bar) === (1, 2)
    @test nedges(bar) === 2
    @test edges2nodes(bar) === ((1,), (2,))
    @test edges2nodes(bar) === faces2nodes(bar)
    @test nfaces(bar) === nedges(bar)
    a = @SVector [10, 20]
    @test f2n_from_c2n(bar, a) == ([10], [20])
    @test f2n_from_c2n(bar, a) == f2n_from_c2n(Bar2_t, a)
    @test cell_side(bar, [10, 20], [20]) === 2
    @test cell_side(bar, [10, 20], [20]) === cell_side(bar, [10, 20], [20])

    # Tri3
    tri = Tri3_t()
    @test nnodes(tri) === 3
    @test nodes(tri) === (1, 2, 3)
    @test nedges(tri) === 3
    @test edges2nodes(tri) === ((1, 2), (2, 3), (3, 1))
    @test edges2nodes(tri) === faces2nodes(tri)
    @test nfaces(tri) === nedges(tri)
    a = @SVector [10, 20, 30]
    @test f2n_from_c2n(tri, a) == ([10, 20], [20, 30], [30, 10])
    @test f2n_from_c2n(tri, a) == f2n_from_c2n(Tri3_t, a)
    @test cell_side(tri, [10, 20, 30], [20, 30]) === 2
    @test cell_side(tri, [10, 20, 30], [20, 30]) === cell_side(tri, [10, 20, 30], [30, 20])
    @test oriented_cell_side(tri, [10, 20, 30], [30, 10]) === 3
    @test oriented_cell_side(tri, [10, 20, 30], [10, 30]) === -3
    @test oriented_cell_side(tri, [10, 20, 30], [20, 30]) === 2
    @test oriented_cell_side(tri, [10, 20, 30], [30, 20]) === -2

    # Quad4
    quad = Quad4_t()
    @test nnodes(quad) === 4
    @test nodes(quad) === (1, 2, 3, 4)
    @test nedges(quad) === 4
    @test edges2nodes(quad) === ((1, 2), (2, 3), (3, 4), (4, 1))
    @test oriented_cell_side(quad, [10, 20, 30, 40], [40, 10]) === 4
    @test oriented_cell_side(quad, [10, 20, 30, 40], [10, 40]) === -4
    @test oriented_cell_side(quad, [20, 10, 30, 40], [30, 10]) === -2
end
