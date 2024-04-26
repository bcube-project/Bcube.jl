@testset "mesh" begin
    m = basic_mesh()

    @test topodim(m) === 2
    @test spacedim(m) === 2
    @test topodim(m, :cell) === 2
    @test topodim(m, :face) === 1
    @test topodim(m, :edge) === 1
    @test topodim(m, :node) === 0
    @test topodim(m, :vertex) === 0

    @test get_nodes(m)[6] == get_nodes(m, 6)
    @test get_nodes(m, 7) == Node([2.0, 0.0])

    @test connectivities_indices(m, :c2n) == indices(connectivities(m, :c2n))

    #test iterator
    c2n = indices(connectivities(m, :c2n))
    c = cells(m)
    for (i, (_c, _c2n)) in enumerate(zip(c, c2n))
        i == 1 ? (@test (i, _c, _c2n) == (1, Quad4_t(), [1, 2, 6, 5])) : nothing
        i == 2 ? (@test (i, _c, _c2n) == (2, Quad4_t(), [2, 3, 7, 6])) : nothing
        i == 3 ? (@test (i, _c, _c2n) == (3, Tri3_t(), [3, 4, 7])) : nothing
    end

    #test get_coords
    @test get_coords.(get_nodes(m, c2n[2])) ==
          [[1.0, 1.0], [2.0, 1.0], [2.0, 0.0], [1.0, 0.0]]

    # test cell->face connectivities
    c2f = indices(connectivities(m, :c2f))
    @test from(connectivities(m, :c2f)) === :cell
    @test to(connectivities(m, :c2f)) === :face
    @test by(connectivities(m, :c2f)) === nothing
    @test nlayers(connectivities(m, :c2f)) === nothing
    @test c2f[1] == [1, 2, 3, 4] && c2f[2] == [5, 6, 7, 2] && c2f[3] == [8, 9, 6]

    # test face->cell connectivities
    f2c = indices(connectivities(m, :f2c))
    @test from(connectivities(m, :f2c)) === :face
    @test to(connectivities(m, :f2c)) === :cell
    @test by(connectivities(m, :f2c)) === nothing
    @test nlayers(connectivities(m, :f2c)) === nothing
    @test f2c[1] == [1] &&
          f2c[2] == [1, 2] &&
          f2c[3] == [1] &&
          f2c[4] == [1] &&
          f2c[5] == [2] &&
          f2c[6] == [2, 3] &&
          f2c[7] == [2] &&
          f2c[8] == [3] &&
          f2c[9] == [3]

    # test face->node connectivities
    f2n = indices(connectivities(m, :f2n))
    @test from(connectivities(m, :f2n)) === :face
    @test to(connectivities(m, :f2n)) === :node
    @test by(connectivities(m, :f2n)) === nothing
    @test nlayers(connectivities(m, :f2n)) === nothing

    @test f2n[1] == [1, 2] &&
          f2n[2] == [2, 6] &&
          f2n[3] == [6, 5] &&
          f2n[4] == [5, 1] &&
          f2n[5] == [2, 3] &&
          f2n[6] == [3, 7] &&
          f2n[7] == [7, 6] &&
          f2n[8] == [3, 4] &&
          f2n[9] == [4, 7]

    f = faces(m)
    @test all([typeof(x) === Bar2_t for x in f])

    # test face orientation
    @test oriented_cell_side(c[1], c2n[1], f2n[2]) === 2   # low-level interface
    @test oriented_cell_side(c[2], c2n[2], f2n[2]) === -4
    @test oriented_cell_side(m, 1, 2) === 2                # high-level interface
    @test oriented_cell_side(m, 1, 2) === 2
    @test oriented_cell_side(m, 1, 9) === 0

    # Inner and outer faces
    @test inner_faces(m) == [2, 6]
    @test outer_faces(m) == [1, 3, 4, 5, 7, 8, 9]

    c2c = connectivity_cell2cell_by_faces(m)
    @test c2c[1] == [2] && c2c[2] == [1, 3] && c2c[3] == [2]

    # Test connectivity cell2cell by nodes
    # Checked using `Set` because order is not relevant
    mesh = rectangle_mesh(4, 4)
    c2c = connectivity_cell2cell_by_nodes(mesh)
    @test Set(c2c[1]) == Set([4, 2, 5])
    @test Set(c2c[2]) == Set([1, 4, 5, 3, 6])
    @test Set(c2c[3]) == Set([6, 2, 5])
    @test Set(c2c[4]) == Set([1, 2, 5, 7, 8])
    @test Set(c2c[5]) == Set([1, 2, 4, 6, 8, 9, 3, 7])
    @test Set(c2c[6]) == Set([9, 3, 5, 8, 2])
    @test Set(c2c[7]) == Set([4, 8, 5])
    @test Set(c2c[8]) == Set([5, 6, 9, 7, 4])
    @test Set(c2c[9]) == Set([6, 5, 8])
end
