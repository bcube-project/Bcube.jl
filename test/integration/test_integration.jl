@testset "Integration" begin
    @testset "Line - volumic" begin
        # Mesh with only one line of degree 1 : [0, 1]
        mesh = line_mesh(2; xmin = 0.0, xmax = 1.0)

        # Test length of line
        @test all(
            integrate_on_ref_element(x -> 1.0, CellInfo(mesh, 1), Quadrature(Val(degree))) ≈
            1.0 for degree in 1:5
        )
    end

    @testset "Triangle - volumic" begin
        # Mesh with only one triangle of degree 1 : [-1, -1], [1, -1], [-1, 1]
        mesh = Mesh(
            [Node([-1.0, -1.0]), Node([1.0, -1.0]), Node([-1.0, 1.0])],
            [Tri3_t()],
            Connectivity([3], [1, 2, 3]),
        )
        icell = 1
        ctype = cells(mesh)[icell]
        cInfo = CellInfo(mesh, icell)

        # Test area of tri
        @test all(
            integrate_on_ref_element(x -> 1.0, cInfo, Quadrature(Val(degree))) ≈ 2.0 for
            degree in 1:5
        )

        # Test for P1 shape functions and products of P1 shape functions
        fs = FunctionSpace(:Lagrange, 1)
        λ = shape_functions(fs, shape(ctype))
        degree = Val(2)
        quad = Quadrature(degree)

        @test integrate_on_ref_element(ReferenceFunction(ξ -> λ(ξ)[1]), cInfo, quad) ≈
              2.0 / 3.0
        @test integrate_on_ref_element(ReferenceFunction(ξ -> λ(ξ)[2]), cInfo, quad) ≈
              2.0 / 3.0
        @test integrate_on_ref_element(ReferenceFunction(ξ -> λ(ξ)[3]), cInfo, quad) ≈
              2.0 / 3.0

        @test integrate_on_ref_element(
            ReferenceFunction(ξ -> λ(ξ)[1] * λ(ξ)[1]),
            cInfo,
            quad,
        ) ≈ 1.0 / 3.0
        @test integrate_on_ref_element(
            ReferenceFunction(ξ -> λ(ξ)[2] * λ(ξ)[2]),
            cInfo,
            quad,
        ) ≈ 1.0 / 3.0
        @test integrate_on_ref_element(
            ReferenceFunction(ξ -> λ(ξ)[3] * λ(ξ)[3]),
            cInfo,
            quad,
        ) ≈ 1.0 / 3.0

        @test integrate_on_ref_element(
            ReferenceFunction(ξ -> λ(ξ)[1] * λ(ξ)[2]),
            cInfo,
            quad,
        ) ≈ 1.0 / 6.0
        @test integrate_on_ref_element(
            ReferenceFunction(ξ -> λ(ξ)[1] * λ(ξ)[3]),
            cInfo,
            quad,
        ) ≈ 1.0 / 6.0
        @test integrate_on_ref_element(
            ReferenceFunction(ξ -> λ(ξ)[2] * λ(ξ)[3]),
            cInfo,
            quad,
        ) ≈ 1.0 / 6.0

        # Test for P2 shape functions and some products of P2 shape functions
        fs = FunctionSpace(:Lagrange, 2)
        λ = shape_functions(fs, shape(ctype))

        _atol = 2 * eps()
        _rtol = 4 * eps()
        for degree in 2:8
            # atol fixed to 1e-10 to get a pass for degree 4
            quad = Quadrature(degree)
            @test isapprox(
                integrate_on_ref_element(ReferenceFunction(ξ -> λ(ξ)[1]), cInfo, quad),
                0.0,
                atol = _atol,
            )
            @test isapprox(
                integrate_on_ref_element(ReferenceFunction(ξ -> λ(ξ)[2]), cInfo, quad),
                0.0,
                atol = _atol,
            )
            @test isapprox(
                integrate_on_ref_element(ReferenceFunction(ξ -> λ(ξ)[3]), cInfo, quad),
                0.0,
                atol = _atol,
            )
            @test isapprox(
                integrate_on_ref_element(ReferenceFunction(ξ -> λ(ξ)[4]), cInfo, quad),
                2.0 / 3.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_on_ref_element(ReferenceFunction(ξ -> λ(ξ)[5]), cInfo, quad),
                2.0 / 3.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_on_ref_element(ReferenceFunction(ξ -> λ(ξ)[6]), cInfo, quad),
                2.0 / 3.0,
                rtol = _rtol,
            )
        end

        for degree in 4:8
            quad = Quadrature(degree)
            @test isapprox(
                integrate_on_ref_element(
                    ReferenceFunction(ξ -> λ(ξ)[1] * λ(ξ)[1]),
                    cInfo,
                    quad,
                ),
                1.0 / 15.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_on_ref_element(
                    ReferenceFunction(ξ -> λ(ξ)[2] * λ(ξ)[2]),
                    cInfo,
                    quad,
                ),
                1.0 / 15.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_on_ref_element(
                    ReferenceFunction(ξ -> λ(ξ)[3] * λ(ξ)[3]),
                    cInfo,
                    quad,
                ),
                1.0 / 15.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_on_ref_element(
                    ReferenceFunction(ξ -> λ(ξ)[4] * λ(ξ)[4]),
                    cInfo,
                    quad,
                ),
                16.0 / 45.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_on_ref_element(
                    ReferenceFunction(ξ -> λ(ξ)[5] * λ(ξ)[5]),
                    cInfo,
                    quad,
                ),
                16.0 / 45.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_on_ref_element(
                    ReferenceFunction(ξ -> λ(ξ)[6] * λ(ξ)[6]),
                    cInfo,
                    quad,
                ),
                16.0 / 45.0,
                rtol = _rtol,
            )

            @test isapprox(
                integrate_on_ref_element(
                    ReferenceFunction(ξ -> λ(ξ)[1] * λ(ξ)[4]),
                    cInfo,
                    quad,
                ),
                0.0,
                atol = _atol,
            )
            @test isapprox(
                integrate_on_ref_element(
                    ReferenceFunction(ξ -> λ(ξ)[1] * λ(ξ)[5]),
                    cInfo,
                    quad,
                ),
                -2.0 / 45.0,
                rtol = _rtol,
            )
            @test isapprox(
                integrate_on_ref_element(
                    ReferenceFunction(ξ -> λ(ξ)[1] * λ(ξ)[6]),
                    cInfo,
                    quad,
                ),
                0.0,
                atol = _atol,
            )
        end
    end

    @testset "Square - volumic" begin
        # Quad on basic mesh
        mesh = basic_mesh()
        cInfo = CellInfo(mesh, 2) # Test integration on quad '2'

        @test all(
            isapprox(integrate_on_ref_element(x -> 2.0, cInfo, Quadrature(degree)), 2.0) for
            degree in 1:13
        )
        @test all(
            isapprox(
                integrate_on_ref_element(
                    PhysicalFunction(x -> 2x[1] + 3x[2]),
                    cInfo,
                    Quadrature(degree),
                ),
                4.5,
            ) for degree in 1:13
        )

        # An other quad
        xmin, ymin = [0.0, 0.0]
        xmax, ymax = [1.0, 1.0]
        Δx = xmax - xmin
        Δy = ymax - ymin
        mesh = one_cell_mesh(:quad; xmin, xmax, ymin, ymax)
        cInfo = CellInfo(mesh, 1)
        ctype = celltype(cInfo)
        cnodes = nodes(cInfo)
        fs = FunctionSpace(:Lagrange, 1)
        λ = shape_functions(fs, shape(ctype))
        ∇λ = ξ -> ∂λξ_∂x(fs, Val(1), ctype, cnodes, ξ)
        degree = Val(2)
        quad = Quadrature(degree)
        @test integrate_on_ref_element(
            ReferenceFunction(ξ -> λ(ξ)[1] * λ(ξ)[1]),
            cInfo,
            quad,
        ) ≈ Δx * Δy / 9
        @test integrate_on_ref_element(
            ReferenceFunction(ξ -> λ(ξ)[1] * λ(ξ)[2]),
            cInfo,
            quad,
        ) ≈ Δx * Δy / 18
        @test integrate_on_ref_element(
            ReferenceFunction(ξ -> λ(ξ)[2] * λ(ξ)[2]),
            cInfo,
            quad,
        ) ≈ Δx * Δy / 9
        @test integrate_on_ref_element(
            ReferenceFunction(ξ -> ∇λ(ξ)[1, :] ⋅ ∇λ(ξ)[1, :]),
            cInfo,
            quad,
        ) ≈ (Δx^2 + Δy^2) / (3 * Δx * Δy)
    end

    @testset "Line (not curved) - surfacic" begin
        Δx = 666.0
        Δy = 314.0
        mesh = basic_mesh(; coef_x = Δx, coef_y = Δy)
        c2f = connectivities_indices(mesh, :c2f)
        f2n = connectivities_indices(mesh, :f2n)

        g(x) = 1.0

        for idegree in 1:3
            degree = Val(idegree)
            quad = Quadrature(degree)

            for icell in (1, 2)
                finfo_1, finfo_2, finfo_3, finfo_4 =
                    map(Base.Fix1(FaceInfo, mesh), c2f[icell])

                @test integrate_on_ref_element(g, finfo_1, quad) ≈ Δx
                @test integrate_on_ref_element(g, finfo_2, quad) ≈ Δy
                @test integrate_on_ref_element(g, finfo_3, quad) ≈ Δx
                @test integrate_on_ref_element(g, finfo_4, quad) ≈ Δy
            end

            icell = 3
            finfo_1, finfo_2, finfo_3 = map(Base.Fix1(FaceInfo, mesh), c2f[icell])
            @test integrate_on_ref_element(g, finfo_1, quad) ≈ Δx
            @test integrate_on_ref_element(g, finfo_2, quad) ≈ norm([Δx, Δy])
            @test integrate_on_ref_element(g, finfo_3, quad) ≈ Δy
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
        quad_mesh = Mesh(cnodes, [ctype], Connectivity([9], collect(1:9)))

        # Build P2 line (equivalent to one of the edge above)
        lnodes =
            [Node([0.0, 0.0]), Node([0.0, xmax - xmin]), Node([alpha, (xmax - xmin) / 2])]
        line_mesh = Mesh(lnodes, [Bcube.Bar3_t()], Connectivity([3], collect(1:3)))

        # Compute analytic arc length
        b = xmax - xmin
        a = alpha
        L = sqrt(b^2 + 16 * a^2) / 2 + b^2 / (8a) * log((4a + sqrt(b^2 + 16 * a^2)) / b)

        # Test integration accuracy
        degquad = Val(200)
        quad = Quadrature(degquad)
        @test integrate_on_ref_element(x -> 1, CellInfo(line_mesh, 1), quad) ≈ L
        @test integrate_on_ref_element(x -> 1, FaceInfo(quad_mesh, 1), quad) ≈ L
    end

    @testset "Sphere" begin
        mesh = read_mesh(
            joinpath(@__DIR__, "..", "assets", "sphere-mesh-r1.msh22");
            warn = false,
        )
        c2n = connectivities_indices(mesh, :c2n)
        S = sum(1:ncells(mesh)) do icell
            cInfo = CellInfo(mesh, icell)
            integrate_on_ref_element(ξ -> 1.0, cInfo, Quadrature(1))
        end
        @test isapprox(S, 4π; atol = 1e-1)
    end

    @testset "Line_boundary" begin
        mesh = one_cell_mesh(:line)

        finfo_1 = FaceInfo(mesh, 1)
        finfo_2 = FaceInfo(mesh, 2)

        g = PhysicalFunction(x -> 2.5)
        @test integrate_on_ref_element(g, finfo_1, Quadrature(1)) ≈ 2.5
        @test integrate_on_ref_element(g, finfo_2, Quadrature(1)) ≈ 2.5

        g = PhysicalFunction(x -> x[1]) # this is x -> x but since all nodes are in R^n, we need to select the component (as for triangles below)
        @test integrate_on_ref_element(g, finfo_1, Quadrature(2)) ≈ -1.0
        @test integrate_on_ref_element(g, finfo_2, Quadrature(2)) ≈ 1.0
    end

    @testset "Triangle_boundary" begin
        mesh = one_cell_mesh(:tri; xmin = 0.0, ymin = 0.0)

        finfo_1 = FaceInfo(mesh, 1)
        finfo_2 = FaceInfo(mesh, 2)
        finfo_3 = FaceInfo(mesh, 3)

        # Test for constant
        g = PhysicalFunction(x -> 2.5)
        @test integrate_on_ref_element(g, finfo_1, Quadrature(2)) ≈ 2.5
        @test integrate_on_ref_element(g, finfo_2, Quadrature(2)) ≈ 2.5 * √(2.0)
        @test integrate_on_ref_element(g, finfo_3, Quadrature(2)) ≈ 2.5

        # Test for linear function
        g = PhysicalFunction(x -> x[1] + x[2])
        @test integrate_on_ref_element(g, finfo_1, Quadrature(2)) ≈ 0.5
        @test integrate_on_ref_element(g, finfo_2, Quadrature(2)) ≈ √(2.0)
        @test integrate_on_ref_element(g, finfo_3, Quadrature(2)) ≈ 0.5
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

        u = PhysicalFunction(u)
        Γ = BoundaryFaceDomain(mesh, "boundary")
        dΓ = Measure(Γ, 2)
        nΓ = get_face_normals(dΓ)
        val = sum(compute(∫(side_n(u) ⋅ side_n(nΓ))dΓ))
        @test isapprox(val, 0.0, atol = 1e-15)
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

        u = PhysicalFunction(u)
        Γ = BoundaryFaceDomain(mesh, "boundary")
        dΓ = Measure(Γ, 2)
        nΓ = get_face_normals(dΓ)
        val = sum(compute(∫(side_n(u) ⋅ side_n(nΓ))dΓ))
        @test isapprox(val, 0.0, atol = 1e-14)
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

        u = PhysicalFunction(u)
        Γ = BoundaryFaceDomain(mesh, "boundary")
        dΓ = Measure(Γ, 2)
        nΓ = get_face_normals(dΓ)
        val = sum(compute(∫(side_n(u) ⋅ side_n(nΓ))dΓ))
        @test isapprox(val, 0.0, atol = 1e-15)
    end

    @testset "PhysicalFunction" begin
        mesh = translate(one_cell_mesh(:line), [1.0])
        dΩ = Measure(CellDomain(mesh), 2)
        g = PhysicalFunction(x -> 2 * x[1])
        b = compute(∫(g)dΩ)
        @test b[1] == 4.0
    end

    @testset "Lagrange cube" begin
        mesh = scale(translate(one_cell_mesh(:cube), [1.0, 1.0, 2.0]), 2.0)
        cInfo = CellInfo(mesh, 1)
        @test integrate_on_ref_element(x -> 1, cInfo, Quadrature(1)) == 64

        mesh = one_cell_mesh(:cube)
        q = Quadrature(1)
        f(x) = 1
        ξηζ = SA[0.0, 0.0, 0.0]
        I3 = SA[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

        for _ in 1:10
            s = 10 * rand()
            m = translate(mesh, rand(3))
            m = scale(m, s)

            cInfo = CellInfo(m, 1)
            ctype = celltype(cInfo)
            cnodes = nodes(cInfo)

            @test isapprox_arrays(
                Bcube.mapping_jacobian(ctype, cnodes, ξηζ),
                s .* I3;
                rtol = 1e-14,
            )
            @test isapprox(integrate_on_ref_element(f, cInfo, q), 8 * s^3)
        end
    end

    @testset "Lagrange tetra" begin
        # One mesh cell
        mesh = one_cell_mesh(
            :tetra;
            xmin = 1.0,
            xmax = 3.0,
            ymin = 1.0,
            ymax = 4.0,
            zmin = 1.0,
            zmax = 5.0,
        )
        c2n = connectivities_indices(mesh, :c2n)
        cInfo = CellInfo(mesh, 1)
        @test integrate_on_ref_element(x -> 1, cInfo, Quadrature(1)) ≈ 4
        dΩ = Measure(CellDomain(mesh), 2)
        g = PhysicalFunction(x -> 1)
        b = compute(∫(g)dΩ)
        @test b[1] ≈ 4
        gref = ReferenceFunction(x -> 1)
        b = compute(∫(gref)dΩ)
        @test b[1] ≈ 4

        lx = 2.0
        ly = 3.0
        lz = 4.0
        mesh = one_cell_mesh(
            :tetra;
            xmin = 0.0,
            xmax = lx,
            ymin = 0.0,
            ymax = ly,
            zmin = 0.0,
            zmax = lz,
        )
        e14 = SA[0, 0, lz]
        e13 = SA[0, ly, 0]
        e12 = SA[lx, 0, 0]
        e24 = SA[-lx, 0, lz]
        e23 = SA[-lx, ly, 0]
        s(a, b) = 0.5 * norm(cross(a, b))
        surface_ref = (s(-e12, e23), s(-e12, e24), s(e24, e23), s(e13, e14))
        for i in 1:4
            dΓ = Measure(BoundaryFaceDomain(mesh, "F$i"), 1)
            b = compute(∫(side⁻(g))dΓ)
            I, = findnz(b)
            @test b[I[1]] ≈ surface_ref[i]
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
        cInfo = CellInfo(mesh, 1)
        @test integrate_on_ref_element(x -> 1, cInfo, Quadrature(1)) == 0.75
        dΩ = Measure(CellDomain(mesh), 2)
        g = PhysicalFunction(x -> 1)
        b = compute(∫(g)dΩ)
        @test b[1] ≈ 0.75

        # Whole cylinder : read the mesh of a cylinder of radius 1 and length 1, and compute its volume
        mesh = read_mesh(
            joinpath(@__DIR__, "..", "assets", "cylinder-mesh-Lz1-nz10.msh22");
            warn = false,
        )

        dΩ = Measure(CellDomain(mesh), 2)
        g = PhysicalFunction(x -> 1)
        b = compute(∫(g)dΩ)
        @test sum(b) ≈ 3.1365484905459
    end

    @testset "Lagrange pyramid" begin
        # One mesh cell. Volume is base_area * height / 3
        mesh = one_cell_mesh(
            :pyra;
            xmin = 1.0,
            xmax = 2.0,
            ymin = 1.0,
            ymax = 2.0,
            zmin = 1.0,
            zmax = 2.5,
        )
        c2n = connectivities_indices(mesh, :c2n)
        cInfo = CellInfo(mesh, 1)
        for d in 1:5
            @test integrate_on_ref_element(x -> 1, cInfo, Quadrature(d)) ≈ 0.5
        end
        dΩ = Measure(CellDomain(mesh), 2)
        g = PhysicalFunction(x -> 1)
        b = compute(∫(g)dΩ)
        @test b[1] ≈ 0.5
    end
end
