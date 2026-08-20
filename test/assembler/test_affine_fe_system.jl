@testset "AffineFESystem" begin
    function test_affine_fe_scalar_ode(degree, rtol)
        # We solve scalar the ODE u'(x) = x with u(x=0) = 1,
        # whose solution is u(x) = x^2/2 + 1
        mesh = line_mesh(11; xmin = 0.0)
        dΩ = Measure(CellDomain(mesh), 2)
        U = TrialFESpace(FunctionSpace(:Lagrange, degree), mesh, Dict("xmin" => 1.0))
        V = TestFESpace(U)
        a(u, v) = ∫(∇(u) ⋅ v)dΩ
        l(v) = ∫(PhysicalFunction(x -> x[1]) * v)dΩ

        # Using the AffineFESystem
        sys = Bcube.AffineFESystem(a, l, U, V)
        A = assemble_bilinear(a, U, V)
        b = assemble_linear(l, V)
        u_sys = Bcube.solve(sys)
        y_sys_dofs = get_dof_values(u_sys)
        y_sys_vertices = var_on_vertices(u_sys, mesh)

        # "Manual" solve
        A[1, :] .= 0.0
        A[1, 1] = 1.0
        b[1] = 1.0
        y_man_dofs = A \ b

        # Exact sol
        f = PhysicalFunction(x -> x[1]^2 / 2 + 1)
        y_exact_vertices = var_on_vertices(f, mesh)

        @test isapprox_arrays(y_sys_dofs, y_man_dofs; rtol = 1e-15)
        @test isapprox_arrays(y_exact_vertices, y_sys_vertices; rtol = rtol) # FE discretization errors comes into play here
    end

    test_affine_fe_scalar_ode(1, 2e-3)
    test_affine_fe_scalar_ode(2, 1e-15)

    @testset "MultiFESpace" begin
        # Solve a system of two independent ODEs using a MultiFESpace:
        #   u1'(x) = 1   with u1(x=0) = 0  =>  u1(x) = x
        #   u2'(x) = x   with u2(x=0) = 1  =>  u2(x) = x^2/2 + 1
        # Using degree-2 Lagrange FE so the exact (polynomial) solutions are
        # represented up to machine precision.
        mesh = line_mesh(11; xmin = 0.0)
        dΩ = Measure(CellDomain(mesh), 3)

        degree = 2
        U1 = TrialFESpace(FunctionSpace(:Lagrange, degree), mesh, Dict("xmin" => 0.0))
        U2 = TrialFESpace(FunctionSpace(:Lagrange, degree), mesh, Dict("xmin" => 1.0))
        V1 = TestFESpace(U1)
        V2 = TestFESpace(U2)

        U = MultiFESpace(U1, U2)
        V = MultiFESpace(V1, V2)

        a((u1, u2), (v1, v2)) = ∫(∇(u1) ⋅ v1 + ∇(u2) ⋅ v2)dΩ
        l((v1, v2)) =
            ∫(PhysicalFunction(x -> 1.0) * v1 + PhysicalFunction(x -> x[1]) * v2)dΩ

        # Build and solve — this exercises the bugfix in the constructor for MultiFESpace
        sys = Bcube.AffineFESystem(a, l, U, V)
        u_sys = Bcube.solve(sys)

        # Compare system dofs against a "manual" solve using the same assembled arrays
        A = assemble_bilinear(a, U, V)
        b = assemble_linear(l, V)
        d = Bcube.assemble_dirichlet_vector(U, V, mesh)
        b0 = b - A * d
        Bcube.apply_homogeneous_dirichlet!(A, b0, U, V, mesh)
        x_man = A \ b0
        y_man_dofs = x_man .+ d
        @test isapprox_arrays(get_dof_values(u_sys), y_man_dofs; rtol = 1e-15)

        # Extract individual FEFunctions and compare with exact solutions on vertices
        u1_sys, u2_sys = get_fe_functions(u_sys)
        exact1 = PhysicalFunction(x -> x[1])
        exact2 = PhysicalFunction(x -> x[1]^2 / 2 + 1)
        @test isapprox_arrays(
            var_on_vertices(u1_sys, mesh),
            var_on_vertices(exact1, mesh);
            rtol = 1e-15,
        )
        @test isapprox_arrays(
            var_on_vertices(u2_sys, mesh),
            var_on_vertices(exact2, mesh);
            rtol = 1e-15,
        )
    end
end
