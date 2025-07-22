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

    @testset "Poisson DG" begin
        degree = 3
        degree_quad = 2 * degree + 1
        γ = degree * (degree + 1)
        n = 4
        Lx = 1.0
        h = Lx / n

        uref = PhysicalFunction(x -> 3 * x[1] + x[2]^2 + 2 * x[1]^3)
        f = PhysicalFunction(x -> -2 - 12 * x[1])
        g = uref # boundary condition

        avg(u) = 0.5 * (side⁺(u) + side⁻(u))

        # Build mesh
        mesh = rectangle_mesh(
            n + 1,
            n + 1;
            xmin = -Lx / 2,
            xmax = Lx / 2,
            ymin = -Lx / 2,
            ymax = Lx / 2,
        )

        # Choose degree and define function space, trial space and test space
        fs = FunctionSpace(:Lagrange, degree)
        U = TrialFESpace(fs, mesh, :discontinuous)
        V = TestFESpace(U)

        # Define volume and boundary measures
        dΩ = Measure(CellDomain(mesh), degree_quad)
        dΓ = Measure(InteriorFaceDomain(mesh), degree_quad)
        dΓb = Measure(BoundaryFaceDomain(mesh), degree_quad)
        nΓ = get_face_normals(dΓ)
        nΓb = get_face_normals(dΓb)

        a_Ω(u, v) = ∫(∇(v) ⋅ ∇(u))dΩ
        l_Ω(v) = ∫(v * f)dΩ

        function a_Γ(u, v)
            ∫(
                -jump(v, nΓ) ⋅ avg(∇(u)) - avg(∇(v)) ⋅ jump(u, nΓ) +
                γ / h * jump(v, nΓ) ⋅ jump(u, nΓ),
            )dΓ
        end

        fa_Γb(u, ∇u, v, ∇v, n) = -v * (∇u ⋅ n) - (∇v ⋅ n) * u + (γ / h) * v * u
        a_Γb(u, v) = ∫(fa_Γb ∘ map(side⁻, (u, ∇(u), v, ∇(v), nΓb)))dΓb

        fl_Γb(v, ∇v, n, g) = -(∇v ⋅ n) * g + (γ / h) * v * g
        l_Γb(v) = ∫(fl_Γb ∘ map(side⁻, (v, ∇(v), nΓb, g)))dΓb

        a(u, v) = a_Ω(u, v) + a_Γ(u, v) + a_Γb(u, v)
        l(v) = l_Ω(v) + l_Γb(v)

        sys = Bcube.AffineFESystem(a, l, U, V)
        uh = Bcube.solve(sys)

        l2(u) = sqrt(sum(compute(∫(u ⋅ u)dΩ)))
        h1(u) = sqrt(sum(compute(∫(u ⋅ u + ∇(u) ⋅ ∇(u))dΩ)))
        e = uref - uh

        el2 = l2(e)
        eh1 = h1(e)
        tol = 1.e-12
        @test el2 < tol
        @test eh1 < tol
    end

    @testset "Constrained Poisson" begin
        n = 2
        Lx = 2.0

        # Build mesh
        mesh = rectangle_mesh(
            n + 1,
            n + 1;
            xmin = -Lx / 2,
            xmax = Lx / 2,
            ymin = -Lx / 2,
            ymax = Lx / 2,
            bnd_names = ("West", "East", "South", "North"),
        )

        # Choose degree and define function space, trial space and test space
        degree = 2
        fs = FunctionSpace(:Lagrange, degree)
        U = TrialFESpace(fs, mesh)
        V = TestFESpace(U)

        Λᵤ = MultiplierFESpace(mesh, 1)
        Λᵥ = TestFESpace(Λᵤ)

        # The usual trial FE space and multiplier space are combined into a MultiFESpace
        P = MultiFESpace(U, Λᵤ)
        Q = MultiFESpace(V, Λᵥ)

        # Define volume and boundary measures
        dΩ = Measure(CellDomain(mesh), 2 * degree + 1)
        Γ₁ = BoundaryFaceDomain(mesh, ("West",))
        dΓ₁ = Measure(Γ₁, 2 * degree + 1)
        Γ = BoundaryFaceDomain(mesh, ("East", "South", "West", "North"))
        dΓ = Measure(Γ, 2 * degree + 1)

        # Define solution FE Function
        ϕ = FEFunction(U)

        f = PhysicalFunction(x -> 1.0)

        int_val = 4.0 / 3.0

        volume = sum(compute(∫(PhysicalFunction(x -> 1.0))dΩ))

        # Define bilinear and linear forms
        function a((u, λᵤ), (v, λᵥ))
            ∫(∇(u) ⋅ ∇(v))dΩ + ∫(side⁻(λᵤ) * side⁻(v))dΓ + ∫(side⁻(λᵥ) * side⁻(u))dΓ
        end

        function l((v, λᵥ))
            ∫(f * v + int_val * λᵥ / volume)dΩ + ∫(-2.0 * side⁻(v) + 0.0 * side⁻(λᵥ))dΓ₁
        end

        # Assemble to get matrices and vectors
        A = assemble_bilinear(a, P, Q)
        L = assemble_linear(l, Q)
        # Solve problem
        sol = A \ L

        ϕ = FEFunction(Q)

        # Compare to analytical solution
        set_dof_values!(ϕ, sol)
        u, λ = ϕ

        u_ref = PhysicalFunction(x -> -0.5 * (x[1] - 1.0)^2 + 1.0)
        error = u_ref - u

        l2(u) = sqrt(sum(compute(∫(u ⋅ u)dΩ)))
        el2 = l2(error)
        tol = 1.e-15
        @test el2 < tol
    end

    @testset "Heat solver 3d" begin
        function heat_solver(mesh, degree, dirichlet_dict, q, η, T_analytical)
            fs = FunctionSpace(:Lagrange, degree)
            U = TrialFESpace(fs, mesh, dirichlet_dict)
            V = TestFESpace(U)
            dΩ = Measure(CellDomain(mesh), 2 * degree + 1)
            a(u, v) = ∫(η * ∇(u) ⋅ ∇(v))dΩ
            l(v) = ∫(q * v)dΩ
            sys = AffineFESystem(a, l, U, V)
            ϕ = Bcube.solve(sys)
            Tcn = var_on_centers(ϕ, mesh)
            Tca = map(T_analytical, get_cell_centers(mesh))
            return norm(Tcn .- Tca, Inf) / norm(Tca, Inf)
        end

        function driver_heat_solver()
            mesh = read_mesh(
                joinpath(@__DIR__, "..", "assets", "hexa-mesh-tetra-xc0-yc0-zc0.msh22");
                warn = false,
            )

            λ = 100.0
            η = λ

            degree = 1
            dirichlet_dict = Dict("xmin" => 260.0, "xmax" => 285.0)
            q = 0.0
            T_analytical(x) = 260.0 * (0.5 - x[1]) + 285 * (x[1] + 0.5)
            err = heat_solver(mesh, degree, dirichlet_dict, q, η, T_analytical)
            @test err < 1.0e-14

            mesh = hexa_mesh(
                5,
                5,
                5;
                xmin = -0.5,
                xmax = 0.5,
                ymin = -0.5,
                ymax = 0.5,
                zmin = -0.5,
                zmax = 0.5,
            )
            degree = 2
            dirichlet_dict = Dict("xmin" => 260.0)
            q = 1500.0
            T2_analytical(x) = 260.0 + (q / λ) * x[1] * (1.0 - 0.5 * x[1])
            scale(x) = (x[1] + 0.5)
            err = heat_solver(mesh, degree, dirichlet_dict, q, η, T2_analytical ∘ scale)
            @test err < 1.0e-14
            #
        end
        driver_heat_solver()
    end

    @testset "Subdomain" begin
        n = 5
        mesh = line_mesh(2 * n + 1; xmin = 0, xmax = 1)
        sub1 = 1:2:(2 * n)
        sub2 = 2:2:(2 * n)
        dΩ = Measure(CellDomain(mesh), 1)
        dΩ1 = Measure(CellDomain(mesh, sub1), 1)
        dΩ2 = Measure(CellDomain(mesh, sub2), 1)
        f = PhysicalFunction(x -> 1.0)
        @test sum(Bcube.compute(∫(f)dΩ1)) == 0.5
        @test sum(Bcube.compute(∫(f)dΩ2)) == 0.5
    end

    @testset "Arbitrary precision" begin
        # to be moved elsewhere? This is a bit "transverse". We could split it in
        # multiple places but we would duplicate a lot of code
        T_int = Int32
        T_float = Float32
        mesh = one_cell_mesh(:line; xmin = T_float(0.0), xmax = T_float(1.0), T_int)

        # Check mesh
        @test eltype(get_coords(first(get_nodes(mesh)))) == T_float
        @test all(Bcube.indices.(values(Bcube.connectivities(mesh)))) do conn
            return eltype(conn) == eltype(conn.offsets) == eltype(conn.indices) == T_int
        end

        # Check domain and measure
        Ω = CellDomain(mesh)
        @test eltype(Ω.indices) == T_int
        dΩ = Measure(Ω, T_int(1))

        # Check FESpace
        fs = FunctionSpace(:Lagrange, Int32(1))
        U = TrialFESpace(fs, mesh, Dict("xmin" => T_float(0.0)))
        V = TestFESpace(U)
        dhl = Bcube._get_dhl(U)
        @test eltype(dhl.iglob) == eltype(dhl.offset) == typeof(dhl.ndofs_tot) == T_int

        # Check FEFunction
        u = FEFunction(U, T_float)
        @test eltype(get_dof_values(u)) == T_float

        # Check bilinear assembly
        a(u, v) = ∫(u * v)dΩ
        M = assemble_bilinear(a, U, V; T = T_float)
        @test eltype(M) == T_float

        # Check linear assembly (without FEFunction)
        b(v) = ∫(v)dΩ
        B = assemble_linear(b, V; T = T_float)
        @test eltype(B)

        # Check linear assembly (with FEFunction)
        c(v) = ∫(u * v)dΩ
        C = assemble_linear(c, V; T = T_float)
        @test eltype(C) == T_float # FAILING FOR NOW
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
