@testset "mapping" begin

    # bmxam: For Jacobians we should have two type of tests : one using 'rand' checking analytical formulae,
    # and another one using "known" cells (for instance Square of side 2) and checking "expected" value:
    # "I known the area of a Square of side 2 is 4". I don't feel like writing the second type of tests right now
    @testset "one cell mesh" begin
        icell = 1
        tol = 1e-15

        # Line order 1
        xmin = -(1.0 + rand())
        xmax = 1.0 + rand()
        mesh = one_cell_mesh(:line; xmin, xmax, order = 1)
        c2n = connectivities_indices(mesh, :c2n)
        ctype = cells(mesh)[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        @test isapprox_arrays(mapping(ctype, cnodes, [-1.0]), [xmin]; rtol = tol)
        @test isapprox_arrays(mapping(ctype, cnodes, [1.0]), [xmax]; rtol = tol)
        @test isapprox_arrays(
            mapping_jacobian(cnodes, ctype, rand(1)),
            @SVector[(xmax - xmin) / 2.0];
            rtol = tol,
        )
        @test isapprox(
            mapping_det_jacobian(cnodes, ctype, rand(1)),
            (xmax - xmin) / 2.0,
            rtol = tol,
        )
        @test isapprox_arrays(mapping_inv(cnodes, ctype, xmin), SA[-1.0], rtol = tol)
        @test isapprox_arrays(mapping_inv(cnodes, ctype, xmax), SA[1.0], rtol = tol)

        # Line order 2 -> no mapping yet, uncomment when mapping is available
        #mesh = scale(one_cell_mesh(:line; order = 2), rand())
        #c2n = connectivities_indices(mesh,:c2n)
        #ct = cells(mesh)[icell]
        #n = get_nodes(mesh, c2n[icell])
        #@test mapping(n, ct, [-1.]) == coords(n[1])
        #@test mapping(n, ct, [ 1.]) == coords(n[2])
        #@test mapping(n, ct, [ 0.]) == coords(n[3])

        # Triangle order 1
        xmin, ymin = -rand(2)
        xmax, ymax = rand(2)
        mesh = one_cell_mesh(:triangle; xmin, xmax, ymin, ymax, order = 1)
        c2n = connectivities_indices(mesh, :c2n)
        ctype = cells(mesh)[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        x1 = [xmin, ymin]
        x2 = [xmax, ymin]
        x3 = [xmin, ymax]
        @test isapprox(mapping(ctype, cnodes, [0.0, 0.0]), x1, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [1.0, 0.0]), x2, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [0.0, 1.0]), x3, rtol = eps())
        @test isapprox_arrays(
            mapping_jacobian(cnodes, ctype, rand(2)),
            SA[
                (xmax-xmin) 0.0
                0.0 (ymax-ymin)
            ],
        )
        @test isapprox(
            mapping_det_jacobian(cnodes, ctype, rand(2)),
            (xmax - xmin) * (ymax - ymin),
            rtol = eps(),
        )
        @test isapprox_arrays(mapping_inv(cnodes, ctype, x1), SA[0.0, 0.0]; rtol = tol)
        @test isapprox_arrays(mapping_inv(cnodes, ctype, x2), SA[1.0, 0.0]; rtol = tol)
        @test isapprox_arrays(mapping_inv(cnodes, ctype, x3), SA[0.0, 1.0]; rtol = tol)

        # Triangle order 2
        xmin, ymin = -rand(2)
        xmax, ymax = rand(2)
        mesh = one_cell_mesh(:triangle; xmin, xmax, ymin, ymax, order = 2)
        c2n = connectivities_indices(mesh, :c2n)
        ctype = cells(mesh)[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        x1 = [xmin, ymin]
        x2 = [xmax, ymin]
        x3 = [xmin, ymax]
        x4 = (x1 + x2) / 2
        x5 = (x2 + x3) / 2
        x6 = (x3 + x1) / 2
        @test isapprox(mapping(ctype, cnodes, [0.0, 0.0]), x1, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [1.0, 0.0]), x2, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [0.0, 1.0]), x3, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [0.5, 0.0]), x4, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [0.5, 0.5]), x5, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [0.0, 0.5]), x6, rtol = eps())
        @test isapprox(
            mapping_det_jacobian(cnodes, ctype, rand(2)),
            (xmax - xmin) * (ymax - ymin),
        )

        # Quad order 1
        xmin, ymin = -rand(2)
        xmax, ymax = rand(2)
        mesh = one_cell_mesh(:quad; xmin, xmax, ymin, ymax, order = 1)
        c2n = connectivities_indices(mesh, :c2n)
        ctype = cells(mesh)[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        x1 = [xmin, ymin]
        x2 = [xmax, ymin]
        x3 = [xmax, ymax]
        x4 = [xmin, ymax]
        @test isapprox(mapping(ctype, cnodes, [-1.0, -1.0]), x1, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [1.0, -1.0]), x2, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [1.0, 1.0]), x3, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [-1.0, 1.0]), x4, rtol = eps())
        @test isapprox(
            mapping_jacobian(cnodes, ctype, rand(2)),
            SA[
                (xmax-xmin) 0.0
                0.0 (ymax-ymin)
            ] / 2.0,
        )
        @test isapprox(
            mapping_det_jacobian(cnodes, ctype, rand(2)),
            (xmax - xmin) * (ymax - ymin) / 4.0,
            rtol = tol,
        )
        @test isapprox_arrays(mapping_inv(cnodes, ctype, x1), SA[-1.0, -1.0]; rtol = tol)
        @test isapprox_arrays(mapping_inv(cnodes, ctype, x2), SA[1.0, -1.0]; rtol = tol)
        @test isapprox_arrays(mapping_inv(cnodes, ctype, x3), SA[1.0, 1.0]; rtol = tol)
        @test isapprox_arrays(mapping_inv(cnodes, ctype, x4), SA[-1.0, 1.0]; rtol = tol)

        θ = π / 5
        s = 3
        t = SA[-1, 2]
        R(θ) = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]
        mesh = one_cell_mesh(:quad)
        mesh = transform(mesh, x -> R(θ) * (s .* x .+ t)) # scale, translate and rotate
        c = CellInfo(mesh, 1)
        cnodes = nodes(c)
        ctype = Bcube.celltype(c)
        @test all(mapping_jacobian_inv(cnodes, ctype, SA[0.0, 0.0]) .≈ R(-θ) ./ s)

        # Quad order 2
        xmin, ymin = -rand(2)
        xmax, ymax = rand(2)
        mesh = one_cell_mesh(:quad; xmin, xmax, ymin, ymax, order = 2)
        c2n = connectivities_indices(mesh, :c2n)
        ctype = cells(mesh)[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        x1 = [xmin, ymin]
        x2 = [xmax, ymin]
        x3 = [xmax, ymax]
        x4 = [xmin, ymax]
        x5 = (x1 + x2) / 2
        x6 = (x2 + x3) / 2
        x7 = (x3 + x4) / 2
        x8 = (x4 + x1) / 2
        x9 = [(xmin + xmax) / 2, (ymin + ymax) / 2]
        @test isapprox(mapping(ctype, cnodes, [-1.0, -1.0]), x1, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [1.0, -1.0]), x2, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [1.0, 1.0]), x3, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [-1.0, 1.0]), x4, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [0.0, -1.0]), x5, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [1.0, 0.0]), x6, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [0.0, 1.0]), x7, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [-1.0, 0.0]), x8, rtol = eps())
        @test isapprox(mapping(ctype, cnodes, [0.0, 0.0]), x9, rtol = eps())
        @test isapprox(
            mapping_det_jacobian(cnodes, ctype, rand(2)),
            (xmax - xmin) * (ymax - ymin) / 4.0,
            rtol = 1000eps(),
        )

        # Quad order 3
        xmin, ymin = -rand(2)
        xmax, ymax = rand(2)
        mesh = one_cell_mesh(:quad; xmin, xmax, ymin, ymax, order = 3)
        c2n = connectivities_indices(mesh, :c2n)
        ctype = cells(mesh)[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        tol = 1e-15
        @test isapprox(mapping(ctype, cnodes, [-1.0, -1.0]), cnodes[1].x, rtol = tol)
        @test isapprox(mapping(ctype, cnodes, [1.0, -1.0]), cnodes[2].x, rtol = tol)
        @test isapprox(mapping(ctype, cnodes, [1.0, 1.0]), cnodes[3].x, rtol = tol)
        @test isapprox(mapping(ctype, cnodes, [-1.0, 1.0]), cnodes[4].x, rtol = tol)
        @test isapprox(mapping(ctype, cnodes, [-1.0 / 3.0, -1.0]), cnodes[5].x, rtol = tol)
        @test isapprox(mapping(ctype, cnodes, [1.0 / 3.0, -1.0]), cnodes[6].x, rtol = tol)
        @test isapprox(mapping(ctype, cnodes, [1.0, -1.0 / 3.0]), cnodes[7].x, rtol = tol)
        @test isapprox(mapping(ctype, cnodes, [1.0, 1.0 / 3.0]), cnodes[8].x, rtol = tol)
        @test isapprox(mapping(ctype, cnodes, [1.0 / 3, 1.0]), cnodes[9].x, rtol = tol)
        @test isapprox(mapping(ctype, cnodes, [-1.0 / 3, 1.0]), cnodes[10].x, rtol = tol)
        @test isapprox(mapping(ctype, cnodes, [-1.0, 1.0 / 3.0]), cnodes[11].x, rtol = tol)
        @test isapprox(mapping(ctype, cnodes, [-1.0, -1.0 / 3.0]), cnodes[12].x, rtol = tol)
        @test isapprox(
            mapping(ctype, cnodes, [-1.0 / 3.0, -1.0 / 3.0]),
            cnodes[13].x,
            rtol = tol,
        )
        @test isapprox(
            mapping(ctype, cnodes, [1.0 / 3.0, -1.0 / 3.0]),
            cnodes[14].x,
            rtol = tol,
        )
        @test isapprox(
            mapping(ctype, cnodes, [1.0 / 3.0, 1.0 / 3.0]),
            cnodes[15].x,
            rtol = tol,
        )
        @test isapprox(
            mapping(ctype, cnodes, [-1.0 / 3.0, 1.0 / 3.0]),
            cnodes[16].x,
            rtol = tol,
        )
        @test isapprox(
            mapping_det_jacobian(cnodes, ctype, rand(2)),
            (xmax - xmin) * (ymax - ymin) / 4.0,
            rtol = 1000eps(),
        )

        # Hexa order 1
        mesh = one_cell_mesh(
            :hexa;
            xmin = 1.0,
            xmax = 2.0,
            ymin = 1.0,
            ymax = 2.0,
            zmin = 1.0,
            zmax = 2.0,
        )
        c2n = connectivities_indices(mesh, :c2n)
        icell = 1
        ctype = cells(mesh)[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        F = mapping(ctype, cnodes)
        @test isapprox_arrays(F([-1.0, -1.0, -1.0]), [1.0, 1.0, 1.0])
        @test isapprox_arrays(F([1.0, -1.0, -1.0]), [2.0, 1.0, 1.0])
        @test isapprox_arrays(F([1.0, 1.0, -1.0]), [2.0, 2.0, 1.0])
        @test isapprox_arrays(F([-1.0, 1.0, -1.0]), [1.0, 2.0, 1.0])
        @test isapprox_arrays(F([-1.0, -1.0, 1.0]), [1.0, 1.0, 2.0])
        @test isapprox_arrays(F([1.0, -1.0, 1.0]), [2.0, 1.0, 2.0])
        @test isapprox_arrays(F([1.0, 1.0, 1.0]), [2.0, 2.0, 2.0])
        @test isapprox_arrays(F([-1.0, 1.0, 1.0]), [1.0, 2.0, 2.0])

        # Hexa order 2
        # Very trivial test for now. To be improved.
        mesh = one_cell_mesh(
            :hexa;
            xmin = 1.0,
            xmax = 2.0,
            ymin = 1.0,
            ymax = 2.0,
            zmin = 1.0,
            zmax = 2.0,
            order = 2,
        )
        c2n = connectivities_indices(mesh, :c2n)
        icell = 1
        ctype = cells(mesh)[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        F = mapping(ctype, cnodes)
        @test isapprox_arrays(F([-1.0, -1.0, -1.0]), [1.0, 1.0, 1.0])
        @test isapprox_arrays(F([1.0, -1.0, -1.0]), [2.0, 1.0, 1.0])
        @test isapprox_arrays(F([1.0, 1.0, -1.0]), [2.0, 2.0, 1.0])
        @test isapprox_arrays(F([-1.0, 1.0, -1.0]), [1.0, 2.0, 1.0])
        @test isapprox_arrays(F([-1.0, -1.0, 1.0]), [1.0, 1.0, 2.0])
        @test isapprox_arrays(F([1.0, -1.0, 1.0]), [2.0, 1.0, 2.0])
        @test isapprox_arrays(F([1.0, 1.0, 1.0]), [2.0, 2.0, 2.0])
        @test isapprox_arrays(F([-1.0, 1.0, 1.0]), [1.0, 2.0, 2.0])

        # Penta6
        mesh = one_cell_mesh(
            :penta;
            xmin = 1.0,
            xmax = 2.0,
            ymin = 1.0,
            ymax = 2.0,
            zmin = 1.0,
            zmax = 2.0,
        )
        c2n = connectivities_indices(mesh, :c2n)
        icell = 1
        ctype = cells(mesh)[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        F = mapping(ctype, cnodes)
        @test isapprox_arrays(F([0.0, 0.0, -1.0]), [1.0, 1.0, 1.0])
        @test isapprox_arrays(F([1.0, 0.0, -1.0]), [2.0, 1.0, 1.0])
        @test isapprox_arrays(F([0.0, 1.0, -1.0]), [1.0, 2.0, 1.0])
        @test isapprox_arrays(F([0.0, 0.0, 1.0]), [1.0, 1.0, 2.0])
        @test isapprox_arrays(F([1.0, 0.0, 1.0]), [2.0, 1.0, 2.0])
        @test isapprox_arrays(F([0.0, 1.0, 1.0]), [1.0, 2.0, 2.0])
    end

    @testset "basic mesh" begin
        # Create mesh
        mesh = basic_mesh()

        # Get cell -> node connectivity from mesh
        c2n = connectivities_indices(mesh, :c2n)

        # Test mapping on quad '2'
        icell = 2
        cnodes = get_nodes(mesh, c2n[icell])
        ctype = cells(mesh)[icell]
        center = sum(y -> coords(y), cnodes) / length(cnodes)
        @test isapprox_arrays(mapping(ctype, cnodes, [-1.0, -1.0]), coords(cnodes[1]))
        @test isapprox_arrays(mapping(ctype, cnodes, [1.0, -1.0]), coords(cnodes[2]))
        @test isapprox_arrays(mapping(ctype, cnodes, [1.0, 1.0]), coords(cnodes[3]))
        @test isapprox_arrays(mapping(ctype, cnodes, [-1.0, 1.0]), coords(cnodes[4]))
        @test isapprox_arrays(mapping(ctype, cnodes, [0.0, 0.0]), center)
        @test isapprox(mapping_det_jacobian(cnodes, ctype, rand(2)), 0.25, rtol = eps())

        @test isapprox_arrays(mapping_inv(cnodes, ctype, coords(cnodes[1])), [-1.0, -1.0])
        @test isapprox_arrays(mapping_inv(cnodes, ctype, coords(cnodes[2])), [1.0, -1.0])
        @test isapprox_arrays(mapping_inv(cnodes, ctype, coords(cnodes[3])), [1.0, 1.0])
        @test isapprox_arrays(mapping_inv(cnodes, ctype, coords(cnodes[4])), [-1.0, 1.0])
        @test isapprox_arrays(mapping_inv(cnodes, ctype, center), [0.0, 0.0])
        x = coords(cnodes[1])
        @test isapprox(
            mapping(ctype, cnodes, mapping_inv(cnodes, ctype, x)),
            x,
            rtol = eps(eltype(x)),
        )

        # Test mapping on triangle '3'
        icell = 3
        cnodes = get_nodes(mesh, c2n[icell])
        ctype = cells(mesh)[icell]
        center = sum(y -> coords(y), cnodes) / length(cnodes)
        @test isapprox_arrays(mapping(ctype, cnodes, [0.0, 0.0]), coords(cnodes[1]))
        @test isapprox_arrays(mapping(ctype, cnodes, [1.0, 0.0]), coords(cnodes[2]))
        @test isapprox_arrays(mapping(ctype, cnodes, [0.0, 1.0]), coords(cnodes[3]))
        @test isapprox(
            mapping(ctype, cnodes, [1 / 3, 1 / 3]),
            center,
            rtol = eps(eltype(center)),
        )
        @test isapprox(mapping_det_jacobian(cnodes, ctype, 0.0), 1.0, rtol = eps())

        @test isapprox_arrays(mapping_inv(cnodes, ctype, coords(cnodes[1])), [0.0, 0.0])
        @test isapprox_arrays(mapping_inv(cnodes, ctype, coords(cnodes[2])), [1.0, 0.0])
        @test isapprox_arrays(mapping_inv(cnodes, ctype, coords(cnodes[3])), [0.0, 1.0])
        @test isapprox(
            mapping_inv(cnodes, ctype, center),
            [1 / 3, 1 / 3],
            rtol = 10 * eps(eltype(x)),
        )
        x = coords(cnodes[1])
        @test isapprox(
            mapping(ctype, cnodes, mapping_inv(cnodes, ctype, x)),
            x,
            rtol = eps(eltype(x)),
        )
    end

    function _check_face_parametrization(mesh)
        c2n = connectivities_indices(mesh, :c2n)
        f2n = connectivities_indices(mesh, :f2n)
        f2c = connectivities_indices(mesh, :f2c)

        cellTypes = cells(mesh)
        faceTypes = faces(mesh)

        for kface in inner_faces(mesh)

            # Face nodes, type and shape
            ftype = faceTypes[kface]
            fshape = shape(ftype)
            fnodes = get_nodes(mesh, f2n[kface])
            Fface = mapping(ftype, fnodes)

            # Neighbor cell i
            i = f2c[kface][1]
            xᵢ = get_nodes(mesh, c2n[i])
            ctᵢ = cellTypes[i]
            shapeᵢ = shape(ctᵢ)
            Fᵢ = mapping(ctᵢ, xᵢ)
            sideᵢ = cell_side(ctᵢ, c2n[i], f2n[kface])
            fpᵢ = mapping_face(shapeᵢ, sideᵢ) # mapping face-ref -> cell_i-ref

            # Neighbor cell j
            j = f2c[kface][2]
            xⱼ = get_nodes(mesh, c2n[j])
            ctⱼ = cellTypes[j]
            shapeⱼ = shape(ctⱼ)
            Fⱼ = mapping(ctⱼ, xⱼ)
            sideⱼ = cell_side(ctⱼ, c2n[j], f2n[kface])
            # This part is a bit tricky : we want the face parametrization (face-ref -> cell-ref) on
            # side `j`. For this, we need to know the permutation between the vertices of `kface` and the
            # vertices of the `sideⱼ`-th face of cell `j`. However all the information we have for entities,
            # namely `fnodes` and `faces2nodes(ctⱼ, sideⱼ)` refer to the nodes, not the vertices. So we need
            # to retrieve the number of vertices of the face and then restrict the arrays to these vertices.
            # (by the way, we use that the vertices appears necessarily in first)
            # We could simplify the expressions below by introducing the notion of "vertex" in Entity, for
            # instance with `nvertices` and `faces2vertices`.
            #
            # Some additional explanations:
            # c2n[j] = global index of the nodes of cell j
            # faces2nodes(ctⱼ, sideⱼ) = local index of the nodes of face sideⱼ of cell j
            # c2n[j][faces2nodes(ctⱼ, sideⱼ)] = global index of the nodes of face sideⱼ of cell j
            # f2n[kface] = global index of the nodes of face `kface`
            # indexin(a, b) = location of elements of `a` in `b`
            nv = length(faces2nodes(shapeⱼ, sideⱼ)) # number of vertices of the face
            iglob_vertices_of_face_of_cell_j =
                [c2n[j][faces2nodes(ctⱼ, sideⱼ)[l]] for l in 1:nv]
            g2l = indexin(f2n[kface][1:nv], iglob_vertices_of_face_of_cell_j)
            fpⱼ = mapping_face(shapeⱼ, sideⱼ, g2l)

            # High-level API
            faceInfo = Bcube.FaceInfo(mesh, kface)

            for (i, fnode) in enumerate(fnodes)
                ξface = coords(fshape)[i]
                @test isapprox(Fface(ξface), fnode.x)
                @test isapprox(Fᵢ(fpᵢ(ξface)), fnode.x)
                @test isapprox(Fⱼ(fpⱼ(ξface)), fnode.x)

                # High-level API tests
                fpt_ref = Bcube.FacePoint(ξface, faceInfo, Bcube.ReferenceDomain())
                fpt_phy = Bcube.change_domain(fpt_ref, Bcube.PhysicalDomain())

                cpt_ref_i = side_n(fpt_ref)
                cpt_ref_j = side_p(fpt_ref)

                cpt_phy_i = Bcube.change_domain(cpt_ref_i, Bcube.PhysicalDomain())
                cpt_phy_j = Bcube.change_domain(cpt_ref_j, Bcube.PhysicalDomain())

                @test isapprox(Bcube.get_coord(fpt_phy), fnode.x)
                @test isapprox(Bcube.get_coord(cpt_ref_i), fpᵢ(ξface))
                @test isapprox(Bcube.get_coord(cpt_ref_j), fpⱼ(ξface))
                @test isapprox(Bcube.get_coord(cpt_phy_i), Fᵢ(fpᵢ(ξface)))
                @test isapprox(Bcube.get_coord(cpt_phy_j), Fⱼ(fpⱼ(ξface)))
            end
        end
    end

    @testset "face parametrization" begin
        # Here we test the face parametrization

        # Two Quad9 side-by-side
        path = joinpath(tempdir, "mesh.msh")
        gen_rectangle_mesh(path, :quad; nx = 3, ny = 2, order = 2)
        mesh = read_msh(path)
        _check_face_parametrization(mesh)

        # For two Hexa8 side-by-side
        path = joinpath(tempdir, "mesh.msh")
        gen_hexa_mesh(path, :hexa; n = [3, 2, 2])
        mesh = read_msh(path)
        _check_face_parametrization(mesh)
    end
end
