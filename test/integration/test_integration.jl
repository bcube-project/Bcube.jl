function test_divergence2d(mesh, u, degree)

    # Connectivity
    c2n = connectivities_indices(mesh, :c2n)
    cnodes = get_nodes(mesh, c2n[1])
    ct = cells(mesh)[1]

    return sum(
        integrate_n((n, x) -> u(x) ⋅ n, iside, cnodes, ct, Quadrature(degree)) for
        iside in 1:nfaces(ct)
    )
end

@testset "Integration" begin
    @testset "Line - volumic" begin
        # Mesh with only one line of degree 1 : [0, 1]
        mesh = line_mesh(2; xmin = 0.0, xmax = 1.0)

        # Connectivity
        c2n = connectivities_indices(mesh, :c2n)
        n = get_nodes(mesh, c2n[1])

        # Test length of line
        @test all(
            integrate(x -> 1.0, n, cells(mesh)[1], Quadrature(Val(degree))) ≈ 1.0 for
            degree in 1:5
        )
    end

    @testset "Triangle - volumic" begin
        # Mesh with only one triangle of degree 1 : [-1, -1], [1, -1], [-1, 1]
        mesh = Mesh(
            [Node([-1.0, -1.0]), Node([1.0, -1.0]), Node([-1.0, 1.0])],
            [Tri3_t()],
            Connectivity([3], [1, 2, 3]),
        )

        # Connectivity
        c2n = connectivities_indices(mesh, :c2n)
        n = get_nodes(mesh, c2n[1])
        cnodes = get_nodes(mesh, c2n[1])
        ct = cells(mesh)[1]

        # Test area of tri
        @test all(
            integrate(x -> 1.0, n, cells(mesh)[1], Quadrature(Val(degree))) ≈ 2.0 for
            degree in 1:5
        )

        # Test for P1 shape functions and products of P1 shape functions
        fs = FunctionSpace(:Lagrange, 1)
        λ = shape_functions(fs, shape(ct))
        degree = Val(2)
        quad = Quadrature(degree)

        @test integrate_ref(ξ -> λ(ξ)[1], cnodes, ct, quad) ≈ 2.0 / 3.0
        @test integrate_ref(ξ -> λ(ξ)[2], cnodes, ct, quad) ≈ 2.0 / 3.0
        @test integrate_ref(ξ -> λ(ξ)[3], cnodes, ct, quad) ≈ 2.0 / 3.0

        @test integrate_ref(ξ -> λ(ξ)[1] * λ(ξ)[1], cnodes, ct, quad) ≈ 1.0 / 3.0
        @test integrate_ref(ξ -> λ(ξ)[2] * λ(ξ)[2], cnodes, ct, quad) ≈ 1.0 / 3.0
        @test integrate_ref(ξ -> λ(ξ)[3] * λ(ξ)[3], cnodes, ct, quad) ≈ 1.0 / 3.0

        @test integrate_ref(ξ -> λ(ξ)[1] * λ(ξ)[2], cnodes, ct, quad) ≈ 1.0 / 6.0
        @test integrate_ref(ξ -> λ(ξ)[1] * λ(ξ)[3], cnodes, ct, quad) ≈ 1.0 / 6.0
        @test integrate_ref(ξ -> λ(ξ)[2] * λ(ξ)[3], cnodes, ct, quad) ≈ 1.0 / 6.0

        # Test for P2 shape functions and some products of P2 shape functions
        fs = FunctionSpace(:Lagrange, 2)
        λ = shape_functions(fs, shape(ct))

        _atol = 2 * eps()
        _rtol = 4 * eps()
        for degree in 2:8
            # atol fixed to 1e-10 to get a pass for degree 4
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[1], cnodes, ct, Quadrature(degree)),
                0.0,
                atol = _atol,
            )
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[2], cnodes, ct, Quadrature(degree)),
                0.0,
                atol = _atol,
            )
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[3], cnodes, ct, Quadrature(degree)),
                0.0,
                atol = _atol,
            )
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[4], cnodes, ct, Quadrature(degree)),
                2.0 / 3.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[5], cnodes, ct, Quadrature(degree)),
                2.0 / 3.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[6], cnodes, ct, Quadrature(degree)),
                2.0 / 3.0,
                rtol = _rtol,
            )
        end

        for degree in 4:8
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[1] * λ(ξ)[1], cnodes, ct, Quadrature(degree)),
                1.0 / 15.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[2] * λ(ξ)[2], cnodes, ct, Quadrature(degree)),
                1.0 / 15.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[3] * λ(ξ)[3], cnodes, ct, Quadrature(degree)),
                1.0 / 15.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[4] * λ(ξ)[4], cnodes, ct, Quadrature(degree)),
                16.0 / 45.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[5] * λ(ξ)[5], cnodes, ct, Quadrature(degree)),
                16.0 / 45.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[6] * λ(ξ)[6], cnodes, ct, Quadrature(degree)),
                16.0 / 45.0,
                rtol = _rtol,
            )

            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[1] * λ(ξ)[4], cnodes, ct, Quadrature(degree)),
                0.0,
                atol = _atol,
            )
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[1] * λ(ξ)[5], cnodes, ct, Quadrature(degree)),
                -2.0 / 45.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_ref(ξ -> λ(ξ)[1] * λ(ξ)[6], cnodes, ct, Quadrature(degree)),
                0.0,
                atol = _atol,
            )
        end
    end

    @testset "Square - volumic" begin
        # Quad on basic mesh
        mesh = basic_mesh()
        c2n = connectivities_indices(mesh, :c2n)
        icell = 2 # Test integration on quad '2'
        n = get_nodes(mesh, c2n[icell])
        ct = cells(mesh)[icell]

        @test all(
            isapprox(integrate(x -> 2.0, n, ct, Quadrature(degree)), 2.0) for degree in 1:13
        )
        @test all(
            isapprox(integrate(x -> 2x[1] + 3x[2], n, ct, Quadrature(degree)), 4.5) for
            degree in 1:13
        )

        # An other quad
        xmin, ymin = [0.0, 0.0]
        xmax, ymax = [1.0, 1.0]
        Δx = xmax - xmin
        Δy = ymax - ymin
        mesh = one_cell_mesh(:quad; xmin, xmax, ymin, ymax)
        c2n = connectivities_indices(mesh, :c2n)
        cnodes = get_nodes(mesh, c2n[1])
        ct = cells(mesh)[1]
        fs = FunctionSpace(:Lagrange, 1)
        λ = shape_functions(fs, shape(ct))
        ∇λ = grad_shape_functions(fs, ct, cnodes)
        degree = Val(2)
        quad = Quadrature(degree)
        @test integrate_ref(ξ -> λ(ξ)[1] * λ(ξ)[1], cnodes, ct, quad) ≈ Δx * Δy / 9
        @test integrate_ref(ξ -> λ(ξ)[1] * λ(ξ)[2], cnodes, ct, quad) ≈ Δx * Δy / 18
        @test integrate_ref(ξ -> λ(ξ)[2] * λ(ξ)[2], cnodes, ct, quad) ≈ Δx * Δy / 9
        @test integrate_ref(ξ -> ∇λ(ξ)[1, :] ⋅ ∇λ(ξ)[1, :], cnodes, ct, quad) ≈
              (Δx^2 + Δy^2) / (3 * Δx * Δy)
    end

    @testset "Line (not curved) - surfacic" begin
        Δx = 666.0
        Δy = 314.0
        mesh = basic_mesh(; coef_x = Δx, coef_y = Δy)

        g(x) = 1.0
        c2n = connectivities_indices(mesh, :c2n)

        for idegree in 1:3
            degree = Val(idegree)
            quad = Quadrature(degree)

            for icell in (1, 2)
                cnodes = get_nodes(mesh, c2n[icell])
                ct = cells(mesh)[icell]
                @test integrate(g, 1, cnodes, ct, quad) ≈ Δx
                @test integrate(g, 2, cnodes, ct, quad) ≈ Δy
                @test integrate(g, 3, cnodes, ct, quad) ≈ Δx
                @test integrate(g, 4, cnodes, ct, quad) ≈ Δy
            end

            icell = 3
            cnodes = get_nodes(mesh, c2n[icell])
            ct = cells(mesh)[icell]
            @test integrate(g, 1, cnodes, ct, quad) ≈ Δx
            @test integrate(g, 2, cnodes, ct, quad) ≈ norm([Δx, Δy])
            @test integrate(g, 3, cnodes, ct, quad) ≈ Δy
        end
    end

    @testset "Line P2 - surfacic" begin
        # Build P2-Quad with edges defined by y = alpha (1-x)(1+x)
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
        alpha = 1.0
        cnodes = [
            Node([xmin, ymin]),
            Node([xmax, ymin]),
            Node([xmax, ymax]),
            Node([xmin, ymax]),
            Node([(xmax + xmin) / 2, ymin - alpha]),
            Node([xmax + alpha, (ymin + ymax) / 2]),
            Node([(xmin + xmax) / 2, ymax + alpha]),
            Node([xmin - alpha, (ymin + ymax) / 2]),
            Node([(xmin + xmax) / 2, (ymin + ymax) / 2]),
        ]
        ctype = Quad9_t()

        # Build P2 line (equivalent to one of the edge above)
        lnodes =
            [Node([0.0, 0.0]), Node([0.0, xmax - xmin]), Node([alpha, (xmax - xmin) / 2])]

        # Compute analytic arc length
        b = xmax - xmin
        a = alpha
        L = sqrt(b^2 + 16 * a^2) / 2 + b^2 / (8a) * log((4a + sqrt(b^2 + 16 * a^2)) / b)

        # Test integration accuracy
        degquad = Val(200)
        quad = Quadrature(degquad)
        @test integrate(x -> 1, lnodes, Bar3_t(), quad) ≈ L
        @test integrate(x -> 1, 1, cnodes, ctype, quad) ≈ L
        @test integrate_ref(x -> 1, lnodes, Bar3_t(), quad) ≈ L
    end

    @testset "Sphere" begin
        mesh = read_msh(string(@__DIR__, "/../../input/mesh/sphere.msh")) # Radius = 1 => area = 4\pi
        c2n = connectivities_indices(mesh, :c2n)
        S = 0.0
        for icell in 1:ncells(mesh)
            cnodes = get_nodes(mesh, c2n[icell])
            ctype = cells(mesh)[icell]

            S += integrate_ref(ξ -> 1.0, cnodes, ctype, Quadrature(1))
        end
        @test isapprox(S, 4π; atol = 1e-1)
    end

    @testset "Line_boundary" begin
        mesh = one_cell_mesh(:line)
        c2n = connectivities_indices(mesh, :c2n)
        cnodes = get_nodes(mesh, c2n[1])
        ct = cells(mesh)[1]

        g = x -> 2.5
        @test integrate(g, 1, cnodes, ct, Quadrature(1)) ≈ 2.5
        @test integrate(g, 2, cnodes, ct, Quadrature(1)) ≈ 2.5

        g = x -> x[1] # this is x -> x but since all nodes are in R^n, we need to select the component (as for triangles below)
        @test integrate(g, 1, cnodes, ct, Quadrature(2)) ≈ -1.0
        @test integrate(g, 2, cnodes, ct, Quadrature(2)) ≈ 1.0

        #--- NEW-API version
        cinfo = CellInfo(mesh, 1)
        f2n = connectivities_indices(mesh, :f2n)

        kface = 1
        _f2n = f2n[kface]
        finfo_1 =
            Bcube.FaceInfo(cinfo, cinfo, faces(mesh)[kface], get_nodes(mesh, _f2n), _f2n)
        kface = 2
        _f2n = f2n[kface]
        finfo_2 =
            Bcube.FaceInfo(cinfo, cinfo, faces(mesh)[kface], get_nodes(mesh, _f2n), _f2n)

        g = PhysicalFunction(x -> 2.5)
        @test Bcube.integrate_face_ref(g, finfo_1, Quadrature(1)) ≈ 2.5
        @test Bcube.integrate_face_ref(g, finfo_2, Quadrature(1)) ≈ 2.5

        g = PhysicalFunction(x -> x[1]) # this is x -> x but since all nodes are in R^n, we need to select the component (as for triangles below)
        @test Bcube.integrate_face_ref(g, finfo_1, Quadrature(2)) ≈ -1.0
        @test Bcube.integrate_face_ref(g, finfo_2, Quadrature(2)) ≈ 1.0
    end

    @testset "Triangle_boundary" begin
        mesh = one_cell_mesh(:tri; xmin = 0.0, ymin = 0.0)
        c2n = connectivities_indices(mesh, :c2n)
        cnodes = get_nodes(mesh, c2n[1])
        ct = cells(mesh)[1]

        # Test for constant
        g = x -> 2.5
        @test integrate(g, 1, cnodes, ct, Quadrature(1)) ≈ 2.5
        @test integrate(g, 2, cnodes, ct, Quadrature(1)) ≈ 2.5 * √(2.0)
        @test integrate(g, 3, cnodes, ct, Quadrature(1)) ≈ 2.5

        # Test for linear function
        g = x -> x[1] + x[2]
        @test integrate(g, 1, cnodes, ct, Quadrature(2)) ≈ 0.5
        @test integrate(g, 2, cnodes, ct, Quadrature(2)) ≈ √(2.0)
        @test integrate(g, 3, cnodes, ct, Quadrature(2)) ≈ 0.5

        #----- NEW-API version
        cinfo = CellInfo(mesh, 1)
        f2n = connectivities_indices(mesh, :f2n)

        kface = 1
        _f2n = f2n[kface]
        finfo_1 =
            Bcube.FaceInfo(cinfo, cinfo, faces(mesh)[kface], get_nodes(mesh, _f2n), _f2n)
        kface = 2
        _f2n = f2n[kface]
        finfo_2 =
            Bcube.FaceInfo(cinfo, cinfo, faces(mesh)[kface], get_nodes(mesh, _f2n), _f2n)
        kface = 3
        _f2n = f2n[kface]
        finfo_3 =
            Bcube.FaceInfo(cinfo, cinfo, faces(mesh)[kface], get_nodes(mesh, _f2n), _f2n)

        # Test for constant
        g = PhysicalFunction(x -> 2.5)
        @test Bcube.integrate_face_ref(g, finfo_1, Quadrature(2)) ≈ 2.5
        @test Bcube.integrate_face_ref(g, finfo_2, Quadrature(2)) ≈ 2.5 * √(2.0)
        @test Bcube.integrate_face_ref(g, finfo_3, Quadrature(2)) ≈ 2.5

        # Test for linear function
        g = PhysicalFunction(x -> x[1] + x[2])
        @test Bcube.integrate_face_ref(g, finfo_1, Quadrature(2)) ≈ 0.5
        @test Bcube.integrate_face_ref(g, finfo_2, Quadrature(2)) ≈ √(2.0)
        @test Bcube.integrate_face_ref(g, finfo_3, Quadrature(2)) ≈ 0.5
    end

    @testset "TriangleP1_boundary_divergence_free" begin
        # Mesh with only one triangle of degree 1 : [1.0, 0.5], [3.5, 1.0], [2.0, 2.0]
        mesh = Mesh(
            [Node([1.0, 0.5]), Node([3.5, 1.0]), Node([2.0, 2.0])],
            [Tri3_t()],
            Connectivity([3], [1, 2, 3]);
            buildboundaryfaces = true,
            bc_names = Dict(1 => "boundary"),
            bc_nodes = Dict(1 => [1, 2, 3]),
        )

        # Test for divergence free vector field
        u = x -> [x[1], -x[2]]

        @test all(
            isapprox(test_divergence2d(mesh, u, degree), 0.0; atol = 1e-14) for
            degree in 2:2
        )

        #---- new api version
        # Rq : performing the integration on the face of the cell sums the contribution of each face
        # in the cell. Hence we don't need a `test_divergence2d`; the sum is implicitely performed by
        # the `compute`
        u = PhysicalFunction(u)
        Γ = BoundaryFaceDomain(mesh, "boundary")
        dΓ = Measure(Γ, 2)
        nΓ = get_face_normals(dΓ)
        val = Bcube.compute(∫(side_n(u) ⋅ side_n(nΓ))dΓ)
        @test isapprox(val[1], 0.0, atol = 1e-15)
    end

    @testset "QuadP2_boundary_divergence_free" begin
        # TODO : use `one_cell_mesh` from `mesh_generator.jl`
        # Mesh with only one quad of degree 2
        mesh = Mesh(
            [
                Node([1.0, 1.0]),
                Node([4.0, 1.0]),
                Node([4.0, 3.0]),
                Node([1.0, 3.0]),
                Node([2.5, 0.5]),
                Node([4.5, 2.0]),
                Node([2.5, 3.5]),
                Node([0.5, 2.0]),
                Node([2.5, 2.0]),
            ],
            [Quad9_t()],
            Connectivity([9], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
            buildboundaryfaces = true,
            bc_names = Dict(1 => "boundary"),
            bc_nodes = Dict(1 => [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        )

        # Test for divergence free vector field
        u = x -> [x[1], -x[2]]

        @test all(
            isapprox(test_divergence2d(mesh, u, degree), 0.0; atol = 1e-14) for
            degree in 2:2
        )

        #---- new api version
        u = PhysicalFunction(u)
        Γ = BoundaryFaceDomain(mesh, "boundary")
        dΓ = Measure(Γ, 2)
        nΓ = get_face_normals(dΓ)
        val = Bcube.compute(∫(side_n(u) ⋅ side_n(nΓ))dΓ)
        @test isapprox(val[1], 0.0, atol = 1e-14)
    end

    @testset "QuadP2_boundary_divergence_free_sincos" begin
        # TODO : use `one_cell_mesh` from `mesh_generator.jl`
        # Mesh with only one quad of degree 2
        mesh = Mesh(
            [
                Node([0.0, 0.0]),
                Node([1.5, 0.0]),
                Node([1.5, 1.5]),
                Node([0.0, 1.5]),
                Node([0.75, -0.25]),
                Node([1.75, 0.75]),
                Node([0.75, 1.75]),
                Node([-0.25, 0.75]),
                Node([0.75, 0.75]),
            ],
            [Quad9_t()],
            Connectivity([9], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
            buildboundaryfaces = true,
            bc_names = Dict(1 => "boundary"),
            bc_nodes = Dict(1 => [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        )

        # Test for divergence free vector field
        u =
            x -> [
                -2.0 * sin(π * x[1]) * sin(π * x[1]) * sin(π * x[2]) * cos(π * x[2]),
                2.0 * sin(π * x[2]) * sin(π * x[2]) * sin(π * x[1]) * cos(π * x[1]),
            ]

        @test all(
            isapprox(test_divergence2d(mesh, u, degree), 0.0; atol = 1e-14) for
            degree in 2:2
        )

        #---- new api version
        u = PhysicalFunction(u)
        Γ = BoundaryFaceDomain(mesh, "boundary")
        dΓ = Measure(Γ, 2)
        nΓ = get_face_normals(dΓ)
        val = Bcube.compute(∫(side_n(u) ⋅ side_n(nΓ))dΓ)
        @test isapprox(val[1], 0.0, atol = 1e-15)
    end

    @testset "PhysicalFunction" begin
        mesh = translate(one_cell_mesh(:line), [1.0])
        dΩ = Measure(CellDomain(mesh), 2)
        g = PhysicalFunction(x -> 2 * x)
        b = Bcube.compute(∫(g)dΩ)
        @test b[1] == 4.0
    end

    @testset "Lagrange cube" begin
        mesh = scale(translate(one_cell_mesh(:cube), [1.0, 1.0, 2.0]), 2.0)
        @test integrate(x -> 1, mesh.nodes, cells(mesh)[1], Quadrature(1)) == 64

        mesh = one_cell_mesh(:cube)
        icell = 1
        ct = cells(mesh)[icell]
        q = Quadrature(1)
        f(x) = 1
        ξηζ = SA[0.0, 0.0, 0.0]
        I3 = SA[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

        for _ in 1:10
            s = 10 * rand()
            m = translate(mesh, rand(3))
            m = scale(m, s)

            nodes = get_nodes(m)

            @test isapprox_arrays(mapping_jacobian(nodes, ct, ξηζ), s .* I3; rtol = 1e-14)
            @test isapprox(integrate(f, nodes, ct, q), 8 * s^3)
        end
    end

    @testset "Lagrange prism" begin
        # One mesh cell
        mesh = one_cell_mesh(
            :penta;
            xmin = 1.0,
            xmax = 2.0,
            ymin = 1.0,
            ymax = 2.0,
            zmin = 1.0,
            zmax = 2.5,
        )
        c2n = connectivities_indices(mesh, :c2n)
        icell = 1
        ctype = cells(mesh)[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        @test integrate(x -> 1, cnodes, ctype, Quadrature(1)) == 0.75
        dΩ = Measure(CellDomain(mesh), 2)
        g = PhysicalFunction(x -> 1)
        b = Bcube.compute(∫(g)dΩ)
        @test b[1] ≈ 0.75

        # Whole cylinder : build a cylinder of radius 1 and length 1, and compute its volume
        gen_cylinder_mesh("mesh.msh", 1.0, 10)
        mesh = read_msh("mesh.msh")
        rm("mesh.msh")

        dΩ = Measure(CellDomain(mesh), 2)
        g = PhysicalFunction(x -> 1)
        b = Bcube.compute(∫(g)dΩ)
        @test sum(b) ≈ 3.1365484905459
    end
end
