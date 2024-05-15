@testset "ref2loc" begin

    # TODO : add more test for gradients, waiting for Ghislain's commit

    @testset "Line" begin
        # Line
        xmin = -rand()
        xmax = rand()
        mesh = one_cell_mesh(:line; xmin, xmax)
        xtest = range(xmin, xmax; length = 5)
        c2n = connectivities_indices(mesh, :c2n)
        cnodes = get_nodes(mesh, c2n[1])
        ctype = cells(mesh)[1]
        Finv = Bcube.mapping_inv(ctype, cnodes)

        xc = center(ctype, cnodes)
        @test isapprox_arrays(xc, [(xmin + xmax) / 2])

        #- Taylor degree 0
        fs = FunctionSpace(:Taylor, 0)
        dhl = DofHandler(mesh, fs, 1, false)
        λ = x -> shape_functions(fs, shape(ctype), Finv(x))
        r = rand()
        q = [r]
        u = interpolate(λ, q[get_dof(dhl, 1)])
        @test isapprox_arrays(λ(rand(1)), [1.0])
        @test all(isapprox(u([x]), r; rtol = eps()) for x in xtest)
        #-- new api
        U = TrialFESpace(fs, mesh, :discontinuous)
        u = FEFunction(U, q)
        cInfo = CellInfo(mesh, 1)
        _u = Bcube.materialize(u, cInfo)
        @test all(
            isapprox(_u(CellPoint(x, cInfo, Bcube.PhysicalDomain())), r; atol = eps()) for
            x in xtest
        )

        #- Taylor degree 1
        fs = FunctionSpace(:Taylor, 1)
        dhl = DofHandler(mesh, fs, 1, false)
        λ = x -> shape_functions(fs, shape(ctype), Finv(x))
        coef = rand()
        q = [coef * (xmin + xmax) / 2, (xmax - xmin) * coef] # f(x) = coef*x
        u = interpolate(λ, q[get_dof(dhl, 1)])
        @test isapprox(λ([xmin]), [1.0, -0.5])
        @test isapprox_arrays(λ(xc), [1.0, 0.0])
        @test isapprox_arrays(λ([xmax]), [1.0, 0.5])
        @test all(isapprox(u([x]), coef * x) for x in xtest)
        #-- new api
        U = TrialFESpace(fs, mesh, :discontinuous)
        u = FEFunction(U, q)
        cInfo = CellInfo(mesh, 1)
        _u = Bcube.materialize(u, cInfo)
        @test all(
            isapprox(
                _u(CellPoint(x, cInfo, Bcube.PhysicalDomain())),
                coef * x;
                atol = eps(),
            ) for x in xtest
        )

        # Normals
        # Bar2_t in 1D
        xmin = -2.0
        xmax = 3.0
        cnodes = [Node([xmin]), Node([xmax])]
        ctype = Bar2_t()
        @test isapprox_arrays(normal(ctype, cnodes, 1, rand(1)), [-1.0])
        @test isapprox_arrays(normal(ctype, cnodes, 2, rand(1)), [1.0])

        # Bar2_t in 2D
        xmin = [-2.0, -1.0]
        xmax = [3.0, 4.0]
        cnodes = [Node(xmin), Node(xmax)]
        ctype = Bar2_t()
        @test normal(ctype, cnodes, 1, rand(1)) ≈ √(2) / 2 .* [-1, -1]
        @test normal(ctype, cnodes, 2, rand(1)) ≈ √(2) / 2 .* [1, 1]

        # Bar2_t in 3D
        xmin = [-2.0, -1.0, 5.0]
        xmax = [3.0, 4.0, 0.0]
        cnodes = [Node(xmin), Node(xmax)]
        ctype = Bar2_t()
        @test normal(ctype, cnodes, 1, rand(1)) ≈ √(3) / 3 * [-1, -1, 1]
        @test normal(ctype, cnodes, 2, rand(1)) ≈ √(3) / 3 * [1, 1, -1]

        # Bar3_t in 3D (checked graphically, see notebook)
        cnodes = [Node([0.0, 0.0, 1.0]), Node([1.0, 0.0, 3.0]), Node([0.5, 0.5, 2.0])]
        ctype = Bar3_t()
        @test normal(ctype, cnodes, 1, rand(1)) ≈ 1.0 / 3.0 .* [-1, -2, -2]
        @test normal(ctype, cnodes, 2, rand(1)) ≈ 1.0 / 3.0 .* [1, -2, 2]
    end

    @testset "Triangle" begin
        # Normals
        # Triangle3_t in 2D
        cnodes = [Node([0.0, 0.0]), Node([1.0, 1.0]), Node([0.0, 2.0])]
        ctype = Tri3_t()
        @test normal(ctype, cnodes, 1, myrand(1, -1.0, 1.0)) ≈ √(2) / 2 .* [1, -1]
        @test normal(ctype, cnodes, 2, myrand(1, -1.0, 1.0)) ≈ √(2) / 2 .* [1, 1]
        @test normal(ctype, cnodes, 3, myrand(1, -1.0, 1.0)) ≈ [-1.0, 0.0]

        # Triangle3_t in 3D (checked graphically, see notebook)
        cnodes = [Node([-0.5, -1.0, -0.3]), Node([0.5, 0.0, 0.0]), Node([-0.5, 1.0, 0.5])]
        ctype = Tri3_t()
        @test normal(ctype, cnodes, 1, myrand(1, -1.0, 1.0)) ≈
              [0.7162290778315695, -0.6203055406219843, -0.3197451240319506]
        @test normal(ctype, cnodes, 2, myrand(1, -1.0, 1.0)) ≈
              [0.7396002616336388, 0.647150228929434, 0.1849000654084097]
        @test normal(ctype, cnodes, 3, myrand(1, -1.0, 1.0)) ≈
              [-0.9957173250742359, -0.034335080174973664, 0.08583770043743415]
    end

    @testset "Quad" begin
        # Quad4
        xmin, ymin = -rand(2)
        xmax, ymax = 3.0 .+ rand(2)
        Δx = xmax - xmin
        Δy = ymax - ymin
        mesh = one_cell_mesh(:quad; xmin, xmax, ymin, ymax)
        xtest = [
            [x, y] for (x, y) in Base.Iterators.product(
                range(xmin, xmax; length = 3),
                range(ymin, ymax; length = 3),
            )
        ]
        c2n = connectivities_indices(mesh, :c2n)
        cnodes = get_nodes(mesh, c2n[1])
        ctype = cells(mesh)[1]
        Finv = Bcube.mapping_inv(ctype, cnodes)

        xc = center(ctype, cnodes)
        @test all(
            map(
                (x, y) -> isapprox(x, y; rtol = eps()),
                xc,
                [(xmin + xmax) / 2, (ymin + ymax) / 2],
            ),
        )

        #- Taylor degree 0
        fs = FunctionSpace(:Taylor, 0)
        dhl = DofHandler(mesh, fs, 1, false)
        λ = shape_functions(fs, shape(ctype))
        r = rand()
        q = [r]
        u = interpolate(λ, q[get_dof(dhl, 1)])
        @test isapprox_arrays(λ(rand(2)), [1.0])
        @test all(isapprox(u(x), r; rtol = eps()) for x in xtest)
        #-- new api
        U = TrialFESpace(fs, mesh, :discontinuous)
        u = FEFunction(U, q)
        cInfo = CellInfo(mesh, 1)
        _u = Bcube.materialize(u, cInfo)
        @test all(
            isapprox(_u(CellPoint(x, cInfo, Bcube.PhysicalDomain())), r; atol = eps()) for
            x in xtest
        )

        #- Taylor degree 1
        fs = FunctionSpace(:Taylor, 1)
        dhl = DofHandler(mesh, fs, 1, false)
        λ = x -> shape_functions(fs, shape(ctype), Finv(x))
        coefx, coefy = rand(2)
        q = [[coefx, coefy] ⋅ xc, Δx * coefx, Δy * coefy] # f(x,y) = coefx*x + coefy*y
        u = interpolate(λ, q[get_dof(dhl, 1)])
        xr = rand(2)
        @test isapprox(λ(xr), [1.0, (xr[1] - xc[1]) / Δx, (xr[2] - xc[2]) / Δy])
        @test all(isapprox(u(x), [coefx, coefy] ⋅ x) for x in xtest)
        #-- new api
        U = TrialFESpace(fs, mesh, :discontinuous)
        u = FEFunction(U, q)
        cInfo = CellInfo(mesh, 1)
        _u = Bcube.materialize(u, cInfo)
        @test all(
            isapprox(
                _u(CellPoint(x, cInfo, Bcube.PhysicalDomain())),
                [coefx, coefy] ⋅ x;
                atol = 10 * eps(),
            ) for x in xtest
        )

        # Normals
        A = [0.0, 0.0]
        B = [1.0, 1.0]
        C = [0.0, 2.0]
        D = [-1.0, 1.0]
        mesh = Mesh(
            [Node(A), Node(B), Node(C), Node(D)],
            [Quad4_t()],
            Connectivity([4], [1, 2, 3, 4]),
        )
        c2n = connectivities_indices(mesh, :c2n)
        cnodes = get_nodes(mesh, c2n[1])
        ctype = cells(mesh)[1]

        AB = B - A
        BC = C - B
        CD = D - C
        DA = A - D

        nAB = normalize(-[-AB[2], AB[1]])
        nBC = normalize(-[-BC[2], BC[1]])
        nCD = normalize(-[-CD[2], CD[1]])
        nDA = normalize(-[-DA[2], DA[1]])

        @test isapprox_arrays(normal(ctype, cnodes, 1, rand(2)), nAB)
        @test isapprox_arrays(normal(ctype, cnodes, 2, rand(2)), nBC)
        @test isapprox_arrays(normal(ctype, cnodes, 3, rand(2)), nCD)
        @test isapprox_arrays(normal(ctype, cnodes, 4, rand(2)), nDA)

        # Quad9
        # Normals (checked graphically, see notebook)
        cnodes = [
            Node([1.0, 1.0]),
            Node([4.0, 1.0]),
            Node([4.0, 3.0]),
            Node([1.0, 3.0]),
            Node([2.5, 0.5]),
            Node([4.5, 2.0]),
            Node([2.5, 3.5]),
            Node([0.5, 2.0]),
            Node([2.5, 2.0]),
        ]
        ctype = Quad9_t()

        xq = [0.5773502691896258]
        @test normal(ctype, cnodes, 1, -xq) ≈ [-0.35921060405354993, -0.9332565252573828]
        @test normal(ctype, cnodes, 1, xq) ≈ [0.3592106040535497, -0.9332565252573829]
        @test normal(ctype, cnodes, 2, -xq) ≈ [0.8660254037844385, -0.5000000000000004]
        @test normal(ctype, cnodes, 2, xq) ≈ [0.8660254037844386, 0.5000000000000003]
        @test normal(ctype, cnodes, 3, -xq) ≈ [0.3592106040535492, 0.9332565252573829]
        @test normal(ctype, cnodes, 3, xq) ≈ [-0.3592106040535494, 0.9332565252573829]
        @test normal(ctype, cnodes, 4, -xq) ≈ [-0.8660254037844388, 0.5]
        @test normal(ctype, cnodes, 4, xq) ≈ [-0.8660254037844385, -0.5000000000000001]
    end
end
