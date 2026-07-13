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

    # Bar3
    bar = Bar3_t()
    @test topodim(bar) === 1
    @test nnodes(bar) === 3
    @test nodes(bar) === (1, 2, 3)
    @test nedges(bar) === 2
    @test edges2nodes(bar) === ((1,), (2,))
    @test nfaces(bar) === nedges(bar)
    @test faces2nodes(bar) === edges2nodes(bar)
    @test edgetypes(bar, 1) === Node_t()

    # Tri6
    tri = Tri6_t()
    @test topodim(tri) === 2
    @test nnodes(tri) === 6
    @test nodes(tri) === (1, 2, 3, 4, 5, 6)
    @test nedges(tri) === 3
    @test edges2nodes(tri) === ((1, 2, 4), (2, 3, 5), (3, 1, 6))
    @test nfaces(tri) === nedges(tri)
    @test faces2nodes(tri) === edges2nodes(tri)
    @test facetypes(tri, 1) === Bar3_t()
    @test edgetypes(tri, 1) === Bar3_t()

    # Quad9
    quad = Quad9_t()
    @test topodim(quad) === 2
    @test nnodes(quad) === 9
    @test nodes(quad) === (1, 2, 3, 4, 5, 6, 7, 8, 9)
    @test nedges(quad) === 4
    @test edges2nodes(quad) === ((1, 2, 5), (2, 3, 6), (3, 4, 7), (4, 1, 8))
    @test nfaces(quad) === nedges(quad)
    @test faces2nodes(quad) === edges2nodes(quad)
    @test facetypes(quad, 1) === Bar3_t()

    # Tetra4
    tetra = Tetra4_t()
    @test topodim(tetra) === 3
    @test nnodes(tetra) === 4
    @test nodes(tetra) === (1, 2, 3, 4)
    @test nedges(tetra) === 6
    @test edges2nodes(tetra) === ((1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4))
    @test nfaces(tetra) === 4
    @test faces2nodes(tetra) === ((1, 3, 2), (1, 2, 4), (2, 3, 4), (3, 1, 4))
    @test facetypes(tetra, 1) === Tri3_t()
    @test edgetypes(tetra, 1) === Bar2_t()
    a = @SVector [10, 20, 30, 40]
    @test f2n_from_c2n(tetra, a) == ([10, 30, 20], [10, 20, 40], [20, 30, 40], [30, 10, 40])
    @test f2n_from_c2n(tetra, a) == f2n_from_c2n(Tetra4_t, a)

    # Hexa8
    hex = Hexa8_t()
    @test topodim(hex) === 3
    @test nnodes(hex) === 8
    @test nodes(hex) === (1, 2, 3, 4, 5, 6, 7, 8)
    @test nedges(hex) === 12
    @test nfaces(hex) === 6
    @test faces2nodes(hex) === (
        (1, 4, 3, 2),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 4, 8, 7),
        (1, 5, 8, 4),
        (5, 6, 7, 8),
    )
    @test facetypes(hex, 1) === Quad4_t()
    @test edgetypes(hex, 1) === Bar2_t()

    # Penta6
    prism = Penta6_t()
    @test topodim(prism) === 3
    @test nnodes(prism) === 6
    @test nodes(prism) === (1, 2, 3, 4, 5, 6)
    @test nedges(prism) === 9
    @test nfaces(prism) === 5
    @test faces2nodes(prism) ===
          ((1, 2, 5, 4), (2, 3, 6, 5), (3, 1, 4, 6), (1, 3, 2), (4, 5, 6))
    @test facetypes(prism, 1) === Quad4_t()
    @test facetypes(prism, 4) === Tri3_t()
    @test edgetypes(prism, 1) === Bar2_t()

    # Pyra5
    pyr = Pyra5_t()
    @test topodim(pyr) === 3
    @test nnodes(pyr) === 5
    @test nodes(pyr) === (1, 2, 3, 4, 5)
    @test nedges(pyr) === 8
    @test nfaces(pyr) === 5
    @test faces2nodes(pyr) === ((1, 4, 3, 2), (1, 2, 5), (2, 3, 5), (3, 4, 5), (4, 1, 5))
    @test facetypes(pyr, 1) === Quad4_t()
    @test facetypes(pyr, 2) === Tri3_t()
end
