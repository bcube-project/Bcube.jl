@testset "Assembler" begin
    @testset "linear-single-scalar-fe" begin
        #---- Mesh 1
        nx = 5
        mesh = line_mesh(nx)
        Δx = 1.0 / (nx - 1)

        # Function space
        fs = FunctionSpace(:Lagrange, 1)

        # Finite element space
        V = TestFESpace(fs, mesh)

        # Measure
        dΩ = Measure(CellDomain(mesh), 3)

        # Linear forms
        f = PhysicalFunction(x -> x[1])
        l1(v) = ∫(v)dΩ
        l2(v) = ∫(f * v)dΩ
        l3(v) = ∫((c * u) ⋅ ∇(v))dΩ

        # Compute integrals
        b1 = assemble_linear(l1, V)
        b2 = assemble_linear(l2, V)

        # Check results
        @test b1 == SA[Δx / 2, Δx, Δx, Δx, Δx / 2]

        res2 = [Δx^2 * (i - 1) for i in 1:nx]
        res2[1] = Δx^2 / 6.0
        res2[nx] = Δx^2 / 2.0 * (nx - 4.0 / 3.0)
        @test all(isapprox.(b2, res2))

        #---- Mesh 2
        degree = 1
        α = 10.0
        c = SA[0.5, 1.0] # [cx, cy]
        sx = 2
        sy = 3
        mesh = one_cell_mesh(:quad)
        transform!(mesh, x -> x .* SA[sx, sy])
        Δx = 2 * sx
        Δy = 2 * sy
        fs = FunctionSpace(:Lagrange, degree)
        U = TrialFESpace(fs, mesh; isContinuous = false)
        V = TestFESpace(U)
        dΩ = Measure(CellDomain(mesh), 2 * degree + 1)
        u = FEFunction(U, α .* ones(get_ndofs(V)))

        # Rq:
        # Mesh is defined by (Δx, Δy)
        # λ[1] <=> λ associated to node [-1, -1] (i.e node 1 of reference square)
        # λ[2] <=> λ associated to node [ 1, -1] (i.e node 2 of reference square)
        # λ[3] <=> λ associated to node [-1,  1] (i.e node 4 of reference square)
        # λ[4] <=> λ associated to node [ 1,  1] (i.e node 3 of reference square)

        l(v) = ∫((c * u) ⋅ ∇(v))dΩ
        b = assemble_linear(l, V)

        b_ref = [
            -α / 2 * (c[1] * Δy + c[2] * Δx),
            α / 2 * (c[1] * Δy - c[2] * Δx),
            α / 2 * (-c[1] * Δy + c[2] * Δx),
            α / 2 * (c[1] * Δy + c[2] * Δx),
        ]

        @test all(b .≈ b_ref)

        # test with operator composition
        _f(u, ∇v) = (c * u) ⋅ ∇v
        l_compo(v) = ∫(_f ∘ (u, ∇(v)))dΩ
        b_compo = assemble_linear(l_compo, V)

        @test all(b_compo .≈ b_ref)
    end

    @testset "linear-single-vector-fe" begin
        # Mesh
        nx = 5
        mesh = line_mesh(nx)
        Δx = 1.0 / (nx - 1)

        # Function space
        fs = FunctionSpace(:Lagrange, 1)

        # Finite element space
        V = TestFESpace(fs, mesh; size = 2)

        # Measure
        dΩ = Measure(CellDomain(mesh), 3)

        # Linear forms
        f = PhysicalFunction(x -> SA[x[1], x[1] / 2.0])
        l1(v) = ∫(f ⋅ v)dΩ

        # Compute integrals
        b1 = assemble_linear(l1, V)

        # Check results
        # (obtained from single-scalar test, and knowing the dof numbering)
        res1 = [Δx^2 * (i - 1) for i in 1:nx]
        res1[1] = Δx^2 / 6.0
        res1[nx] = Δx^2 / 2.0 * (nx - 4.0 / 3.0)

        @test b1[1] ≈ res1[1]
        @test b1[2] ≈ res1[2]
        @test b1[3] ≈ res1[1] / 2.0
        @test b1[4] ≈ res1[2] / 2.0
        @test all(isapprox.(b1[5:2:end], res1[3:end]))
        @test all(isapprox.(b1[6:2:end], res1[3:end] ./ 2.0))
    end

    @testset "linear-multi-scalar-fe" begin
        # Mesh
        nx = 5
        mesh = line_mesh(nx)
        Δx = 1.0 / (nx - 1)

        # Function space
        fs = FunctionSpace(:Lagrange, 1)

        # Finite element space
        V1 = TestFESpace(fs, mesh)
        V2 = TestFESpace(fs, mesh)
        V = MultiFESpace(V1, V2; arrayOfStruct = false)

        # Measure
        dΩ = Measure(CellDomain(mesh), 3)

        # Linear forms
        f1 = PhysicalFunction(x -> x[1])
        f2 = PhysicalFunction(x -> x[1] / 2.0)
        l1((v1, v2)) = ∫(f1 * v1 + f2 * v2)dΩ

        # Compute integrals
        b1 = assemble_linear(l1, V)

        # Check results
        # (obtained from single-scalar test, and knowing the dof numbering)
        res1 = [Δx^2 * (i - 1) for i in 1:nx]
        res1[1] = Δx^2 / 6.0
        res1[nx] = Δx^2 / 2.0 * (nx - 4.0 / 3.0)

        @test all(isapprox.(b1[1:nx], res1))
        @test all(isapprox.(b1[(nx + 1):end], res1 ./ 2.0))

        # test with operator composition
        f_l1_compo(v1, v2, f1, f2) = f1 * v1 + f2 * v2
        l1_compo((v1, v2)) = ∫(f_l1_compo ∘ (v1, v2, f1, f2))dΩ
        b1_compo = assemble_linear(l1_compo, V)

        @test all(b1_compo .≈ b1)

        # test `MultiIntegration`
        l_multi(v) = 3 * l1(v) - 4 * l1(v) + 7 * l1(v)
        b_multi = assemble_linear(l_multi, V)

        @test all(b_multi .≈ 6 * b1)
    end

    @testset "linear-multi-vector-fe" begin
        # Mesh
        mesh = one_cell_mesh(:tri)

        # Function space
        fs_u = FunctionSpace(:Lagrange, 1)
        fs_p = FunctionSpace(:Lagrange, 1)

        # Finite element spaces
        U_xy = TrialFESpace(fs_u, mesh; size = 2)
        U_z = TrialFESpace(fs_u, mesh)
        U_p = TrialFESpace(fs_p, mesh)
        V_xy = TestFESpace(U_xy)
        V_z = TestFESpace(U_z)
        V_p = TestFESpace(U_p)
        U = MultiFESpace(U_p, U_xy, U_z)
        V = MultiFESpace(V_p, V_xy, V_z)

        # Measure
        dΩ = Measure(CellDomain(mesh), 3)

        # bi-linear form
        a((u_p, u_xy, u_z), (v_p, v_xy, v_z)) = ∫(u_xy ⋅ v_xy + u_p ⋅ v_z + ∇(u_p) ⋅ v_xy)dΩ

        # Compute integrals
        A = assemble_bilinear(a, U, V)

        # Test a few values...
        @test A[12, 1] ≈ 1.0 / 6.0
        @test A[12, 2] ≈ 1.0 / 6.0
        @test A[12, 3] ≈ 1.0 / 3.0
    end

    @testset "Bilinear rectangular system" begin
        mesh = one_cell_mesh(:line)

        U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
        V = TestFESpace(FunctionSpace(:Lagrange, 2), mesh)

        dΩ = Measure(CellDomain(mesh), 2)
        a(u, v) = ∫(∇(u) ⋅ v)dΩ
        l(v) = ∫(v)dΩ

        A = assemble_bilinear(a, U, V)
        b = assemble_linear(l, V)

        @test all((A[1, 1], A[1, 2]) .≈ (-1.0 / 6.0, 1.0 / 6.0))
        @test all((A[2, 1], A[2, 2]) .≈ (-2.0 / 3.0, 2.0 / 3.0))
        @test all((A[3, 1], A[3, 2]) .≈ (-1.0 / 6.0, 1.0 / 6.0))
    end

    # @testset "Symbolic (to be completed)" begin
    #     using MultivariatePolynomials
    #     using TypedPolynomials

    #     @polyvar x y
    #     λ1 = (x - 1) * (y - 1) / 4
    #     λ2 = -(x + 1) * (y - 1) / 4
    #     λ4 = (x + 1) * (y + 1) / 4 # to match Bcube ordering
    #     λ3 = -(x - 1) * (y + 1) / 4
    #     λall = (λ1, λ2, λ3, λ4)

    #     ∇λ1 = differentiate(λ1, (x, y))
    #     ∇λ2 = differentiate(λ2, (x, y))
    #     ∇λ3 = differentiate(λ3, (x, y))
    #     ∇λ4 = differentiate(λ4, (x, y))
    #     ∇λall = (∇λ1, ∇λ2, ∇λ3, ∇λ4)

    #     function integrate_poly_on_quad(p)
    #         a = antidifferentiate(p, x)
    #         b = subs(a, x => 1) - subs(a, x => -1)
    #         c = antidifferentiate(b, y)
    #         d = subs(c, y => 1) - subs(c, y => -1)
    #         return d
    #     end

    #     interp_sca(base, values) = [base...] ⋅ values
    #     function interp_vec(base, values)
    #         n = floor(Int, length(values) / 2)
    #         return [interp_sca(base, values[1:n]), interp_sca(base, values[n+1:end])]
    #     end

    #     # One cell mesh (for volumic integration)
    #     mesh = one_cell_mesh(:quad)
    #     factor = 1
    #     scale!(mesh, factor)
    #     degree = 1
    #     dΩ = Measure(CellDomain(mesh), 2 * degree + 1)
    #     fs = FunctionSpace(:Lagrange, degree)
    #     U_sca = TrialFESpace(fs, mesh, :discontinuous)
    #     U_vec = TrialFESpace(fs, mesh, :discontinuous; size=2)
    #     V_sca = TestFESpace(U_sca)
    #     V_vec = TestFESpace(U_vec)
    #     ρ = FEFunction(U_sca)
    #     ρu = FEFunction(U_vec)
    #     ρE = FEFunction(U_sca)
    #     U = MultiFESpace(U_sca, U_vec, U_sca)
    #     V = MultiFESpace(V_sca, V_vec, V_sca)

    #     # bilinear tests
    #     a(u, v) = ∫(u ⋅ v)dΩ
    #     M1 = assemble_bilinear(a, U_sca, V_sca)
    #     M2 = factor^2 * [integrate_poly_on_quad(λj * λi) for λj in λall, λi in λall]
    #     @show all(isapprox.(M1, M2))

    #     a(u, v) = ∫(∇(u) ⋅ ∇(v))dΩ
    #     M1 = assemble_bilinear(a, U_sca, V_sca)
    #     M2 = [integrate_poly_on_quad(∇λj ⋅ ∇λi) for ∇λj in ∇λall, ∇λi in ∇λall]
    #     @show all(isapprox.(M1, M2))

    #     # linear tests
    #     values_ρ = [1.0, 2.0, 3.0, 4.0]
    #     values_ρu = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    #     ρ.dofValues .= values_ρ
    #     ρu.dofValues .= values_ρu
    #     ρu_sym = interp_vec(λall, values_ρu)

    #     l(v) = ∫(ρu ⋅ ∇(v))dΩ
    #     b1 = assemble_linear(l, V_sca)
    #     b2 = [integrate_poly_on_quad(ρu_sym ⋅ ∇λi) for ∇λi in ∇λall]
    #     @show all(isapprox.(b1, b2))
end
