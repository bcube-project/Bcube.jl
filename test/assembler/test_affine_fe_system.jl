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
end
