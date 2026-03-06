@testset "AffineFESystem" begin
    # We solve scalar the ODE u'(x) = x with u(x=0) = 1,
    # whose solution is u(x) = x^2/2 + 1
    mesh = line_mesh(11; xmin = 0.0)
    dΩ = Measure(CellDomain(mesh), 2)
    U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, Dict("xmin" => 1.0))
    V = TestFESpace(U)
    a(u, v) = ∫(∇(u) ⋅ v)dΩ
    l(v) = ∫(PhysicalFunction(x -> x[1]) * v)dΩ

    # Using the AffineFESystem
    sys = Bcube.AffineFESystem(a, l, U, V)
    A = assemble_bilinear(a, U, V)
    b = assemble_linear(l, V)
    y_sys = get_dof_values(Bcube.solve(sys))

    # "Manual" solve
    A[1, :] .= 0.0
    A[1, 1] = 1.0
    b[1] = 1.0
    y_man = A \ b

    # Exact sol
    x = first.(get_coords.(get_nodes(mesh)))
    y_exact = x .^ 2 ./ 2 .+ 1

    @test isapprox_arrays(y_sys, y_man) # should be identical
    @test isapprox_arrays(y_exact, y_man; rtol = 2e-3) # FE discretization errors comes into play here
end