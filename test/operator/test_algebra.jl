import Bcube:
    Connectivity, connectivities_indices, cells, CellInfo, get_nodes, spacedim, nodes, Mesh

@testset "Algebra" begin
    @testset "Gradient" begin
        # We test the mapping of a gradient. The idea is to compute the integral of a function `f` whose
        # gradient is constant. Then the result must be this constant multiplied by the cell area.
        # But since we need to input a `ReferenceFunction`, we need to build `f ∘ F` where `F` is the mapping.
        # We also need a geometric function to compute the area of a convex quad.
        function convex_quad_area(cnodes)
            n1, n2, n3, n4 = cnodes
            return (
                abs((n1.x[1] - n3.x[1]) * (n2.x[2] - n4.x[2])) +
                abs((n2.x[1] - n4.x[1]) * (n1.x[2] - n3.x[2]))
            ) / 2
        end

        cnodes = [Node([0.0, 0.0]), Node([1.0, 0.0]), Node([2.0, 1.5]), Node([1.0, 1.5])]
        celltypes = [Quad4_t()]
        cell2node = Connectivity([4], [1, 2, 3, 4])
        mesh = Mesh(cnodes, celltypes, cell2node)
        # mesh = one_cell_mesh(:quad)

        c2n = connectivities_indices(mesh, :c2n)
        icell = 1
        cnodes = get_nodes(mesh, c2n[icell])
        ctype = cells(mesh)[icell]
        cInfo = CellInfo(mesh, icell)

        qDegree = Val(2)

        # Scalar test : gradient of scalar `f` in physical coordinates is [1, 2]
        function f1(ξ)
            x, y = Bcube.mapping(ctype, cnodes, ξ)
            return x + 2y
        end
        g = ReferenceFunction(f1)

        _g = Bcube.materialize(∇(g), cInfo)
        res = Bcube.integrate_on_ref(_g, cInfo, Quadrature(qDegree))
        @test all(isapprox.(res ./ convex_quad_area(cnodes), [1.0, 2.0]))

        # Vector test : gradient of vector `f` in physical coordinates is [[1,2],[3,4]]
        function f2(ξ)
            x, y = Bcube.mapping(ctype, cnodes, ξ)
            return [x + 2y, 3x + 4y]
        end
        g = ReferenceFunction(f2)

        _g = Bcube.materialize(∇(g), cInfo)
        res = Bcube.integrate_on_ref(_g, cInfo, Quadrature(qDegree))
        @test all(isapprox.(res ./ convex_quad_area(cnodes), [1.0 2.0; 3.0 4.0]))

        # Gradient of scalar PhysicalFunction
        # Physical function is [x,y] -> x so its gradient is [x,y] -> [1, 0]
        # so the integral is simply the volume of Ω
        mesh = one_cell_mesh(:quad)
        translate!(mesh, SA[-0.5, 1.0]) # the translation vector can be anything
        scale!(mesh, 2.0)
        dΩ = Measure(CellDomain(mesh), 1)
        @test Bcube.compute(∫(∇(PhysicalFunction(x -> x[1])) ⋅ [1, 1])dΩ)[1] == 16.0

        # Gradient of a vector PhysicalFunction
        mesh = one_cell_mesh(:quad)
        translate!(mesh, SA[π, -3.14]) # the translation vector can be anything
        scale!(mesh, 2.0)
        sizeU = spacedim(mesh)
        U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh; sizeU)
        V = TestFESpace(U)
        _f = x -> SA[2 * x[1]^2, x[1] * x[2]]
        f = PhysicalFunction(_f, sizeU)
        ∇f = PhysicalFunction(x -> ForwardDiff.jacobian(_f, x), (sizeU, spacedim(mesh)))
        dΩ = Measure(CellDomain(mesh), 3)
        l(v) = ∫(tr(∇(f) - ∇f) ⋅ v)dΩ
        _a = assemble_linear(l, V)
        @test all(isapprox.(_a, [0.0, 0.0, 0.0, 0.0]; atol = 100 * eps()))

        @testset "TangentialGradient" begin
            ctype = Bar2_t()

            nodes = [Node([-1.0]), Node([1.0])]
            celltypes = [ctype]
            cell2node = Bcube.Connectivity([2], [1, 2])
            mesh = Bcube.Mesh(nodes, celltypes, cell2node)

            nodes_hypersurface = [Node([0.0, -1.0]), Node([0.0, 1.0])]
            celltypes = [ctype]
            cell2node = Bcube.Connectivity([2], [1, 2])
            mesh_hypersurface = Bcube.Mesh(nodes_hypersurface, celltypes, cell2node)

            fs = FunctionSpace(:Lagrange, 1)
            n = Val(1)

            ∇_volumic = Bcube.∂λξ_∂x(fs, n, ctype, nodes, [0.0])
            ∇_hyper = Bcube.∂λξ_∂x_hypersurface(fs, n, ctype, nodes_hypersurface, [0.0])
            @test all(isapprox.(∇_volumic, ∇_hyper[:, 2]))

            h(x) = x[1]
            ∇_volumic = Bcube.∂fξ_∂x(h, n, ctype, nodes, [0.0])
            ∇_hyper = Bcube.∂fξ_∂x_hypersurface(h, n, ctype, nodes_hypersurface, [0.0])
            @test ∇_volumic[1] == ∇_hyper[2]
        end

        @testset "AbstractLazy" begin
            mesh = one_cell_mesh(:quad)
            scale!(mesh, 3.0)
            translate!(mesh, [4.0, 0.0])
            U_sca = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
            U_vec = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh; size = 2)
            V_sca = TestFESpace(U_sca)
            V_vec = TestFESpace(U_vec)
            u_sca = FEFunction(U_sca)
            u_vec = FEFunction(U_vec)
            dΩ = Measure(CellDomain(mesh), 2)
            projection_l2!(u_sca, PhysicalFunction(x -> 3 * x[1] - 4x[2]), dΩ)
            projection_l2!(
                u_vec,
                PhysicalFunction(x -> SA[2x[1] + 5x[2], 4x[1] - 3x[2]]),
                dΩ,
            )

            l1(v) = ∫((∇(π * u_sca) ⋅ ∇(2 * u_sca)) ⋅ v)dΩ
            l2(v) = ∫((∇(u_sca) ⋅ ∇(u_sca)) ⋅ v)dΩ

            a1_sca = assemble_linear(l1, V_sca)
            a2_sca = assemble_linear(l2, V_sca)
            @test all(a1_sca .≈ (2π .* a2_sca))

            V_vec = TestFESpace(U_vec)
            l1_vec(v) = ∫((∇(π * u_vec) * u_vec) ⋅ v)dΩ
            l2_vec(v) = ∫((∇(u_vec) * u_vec) ⋅ v)dΩ
            a1_vec = assemble_linear(l1_vec, V_vec)
            a2_vec = assemble_linear(l2_vec, V_vec)
            @test all(a1_vec .≈ (π .* a2_vec))

            # Testing without assemble_*linear
            θ = π / 5
            s = 3
            t = SA[-1, 2]
            R = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]
            mesh = one_cell_mesh(:quad)

            transform!(mesh, x -> R * (s .* x .+ t)) # scale, translate and rotate

            # Select a cell and get its info
            c = CellInfo(mesh, 1)
            cnodes = nodes(c)
            ctype = Bcube.celltype(c)
            F = Bcube.mapping(ctype, cnodes)
            # tJinv = transpose(R ./ s) # if we want the analytic one...
            tJinv(ξ) = transpose(Bcube.mapping_jacobian_inv(ctype, cnodes, ξ))

            # Test 1
            u1 = PhysicalFunction(x -> x[1])
            u2 = PhysicalFunction(x -> x[2])
            u_a = u1 * u1 + 2 * u1 * u2 + u2 * u2 * u2
            u_b = PhysicalFunction(x -> x[1]^2 + 2 * x[1] * x[2] + x[2]^3)
            ∇u_ana = x -> SA[2 * (x[1] + x[2]); 2 * x[1] + 3 * x[2]^2]

            ξ = Bcube.CellPoint(SA[0.5, -0.1], c, Bcube.ReferenceDomain())
            x = Bcube.change_domain(ξ, Bcube.PhysicalDomain())
            ∇u = ∇u_ana(Bcube.get_coord(x))
            ∇u_a_ref = Bcube.materialize(∇(u_a), ξ)
            ∇u_b_ref = Bcube.materialize(∇(u_b), ξ)
            ∇u_a_phy = Bcube.materialize(∇(u_a), x)
            ∇u_b_phy = Bcube.materialize(∇(u_b), x)
            @test all(∇u_a_ref .≈ ∇u)
            @test all(∇u_b_ref .≈ ∇u)
            @test all(∇u_a_phy .≈ ∇u)
            @test all(∇u_b_phy .≈ ∇u)

            # Test 2
            u1 = ReferenceFunction(ξ -> ξ[1])
            u2 = ReferenceFunction(ξ -> ξ[2])
            u_a = u1 * u1 + 2 * u1 * u2 + u2 * u2 * u2
            u_b = ReferenceFunction(ξ -> ξ[1]^2 + 2 * ξ[1] * ξ[2] + ξ[2]^3)
            ∇u_ana = ξ -> SA[2 * (ξ[1] + ξ[2]); 2 * ξ[1] + 3 * ξ[2]^2]

            x = Bcube.CellPoint(SA[0.5, -0.1], c, Bcube.PhysicalDomain())
            ξ = Bcube.change_domain(x, Bcube.ReferenceDomain()) # not always possible, but ok of for quad
            ∇u = ∇u_ana(Bcube.get_coord(ξ))
            _tJinv = tJinv(Bcube.get_coord(ξ))
            ∇u_a_ref = Bcube.materialize(∇(u_a), ξ)
            ∇u_b_ref = Bcube.materialize(∇(u_b), ξ)
            ∇u_a_phy = Bcube.materialize(∇(u_a), x)
            ∇u_b_phy = Bcube.materialize(∇(u_b), x)
            @test all(∇u_a_ref .≈ _tJinv * ∇u)
            @test all(∇u_b_ref .≈ _tJinv * ∇u)
            @test all(∇u_a_phy .≈ _tJinv * ∇u)
            @test all(∇u_b_phy .≈ _tJinv * ∇u)

            # Test 3
            u_phy = PhysicalFunction(x -> x[1]^2 + 2 * x[1] * x[2] + x[2]^3)
            u_ref = ReferenceFunction(ξ -> ξ[1]^2 + 2 * ξ[1] * ξ[2] + ξ[2]^3)
            ∇u_ana = t -> SA[2 * (t[1] + t[2]); 2 * t[1] + 3 * t[2]^2]

            ξ = Bcube.CellPoint(SA[0.5, -0.1], c, Bcube.ReferenceDomain())
            x = Bcube.change_domain(ξ, Bcube.PhysicalDomain())
            ∇u_ref = ∇u_ana(Bcube.get_coord(ξ))
            ∇u_phy = ∇u_ana(Bcube.get_coord(x))
            _tJinv = tJinv(Bcube.get_coord(ξ))
            @test all(Bcube.materialize(∇(u_phy + u_ref), ξ) .≈ ∇u_phy + _tJinv * ∇u_ref)
            @test all(Bcube.materialize(∇(u_phy + u_ref), x) .≈ ∇u_phy + _tJinv * ∇u_ref)
        end
    end

    @testset "algebra" begin
        f = PhysicalFunction(x -> 0)
        a = Bcube.NullOperator()
        @test dcontract(a, a) == a
        @test dcontract(rand(), a) == a
        @test dcontract(a, rand()) == a
        @test dcontract(f, a) == a
        @test dcontract(a, f) == a
    end

    @testset "UniformScaling" begin
        mesh = one_cell_mesh(:quad)
        U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh; size = 2)
        V = TestFESpace(U)
        p = PhysicalFunction(x -> 3)
        dΩ = Measure(CellDomain(mesh), 1)
        l(v) = ∫((p * I) ⊡ ∇(v))dΩ
        a = assemble_linear(l, V)
        @test all(a .≈ [-3.0, 3.0, -3.0, 3.0, -3.0, -3.0, 3.0, 3.0])
    end

    @testset "Pow" begin
        mesh = one_cell_mesh(:quad)
        u = PhysicalFunction(x -> 3.0)
        v = PhysicalFunction(x -> 2.0)
        cinfo = CellInfo(mesh, 1)
        cpoint = Bcube.CellPoint(SA[0.1, 0.3], cinfo, Bcube.ReferenceDomain())

        @test Bcube.materialize(u * u, cinfo)(cpoint) ≈ 9
        @test Bcube.materialize(u^2, cinfo)(cpoint) ≈ 9
        @test Bcube.materialize(u^v, cinfo)(cpoint) ≈ 9

        a = Bcube.NullOperator()
        @test a^u == a
    end
end
