@testset "Dirichlet" begin
    # Test `apply_dirichlet_to_matrix!`
    mesh = rectangle_mesh(3, 2)
    dΩ = Measure(CellDomain(mesh), 1)
    U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, Dict("xmin" => 0.0))
    V = TestFESpace(U)
    A = assemble_bilinear((u, v) -> ∫(u ⋅ v)dΩ, U, V)
    B = copy(A)
    Bcube.apply_dirichlet_to_matrix!((A, B), U, V, mesh; diag_values = (2.0, 0.0))
    IA, JA, VA = findnz(A)
    @test IA == [
        1,
        2,
        3,
        4,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        1,
        2,
        3,
        4,
        5,
        6,
        2,
        4,
        5,
        6,
        2,
        4,
        5,
        6,
    ]
    @test JA == [
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
    ]
    @test VA == [
        2.0,
        0.03125,
        0.0,
        0.03125,
        0.0,
        0.0625,
        0.0,
        0.0625,
        0.03125,
        0.03125,
        0.0,
        0.03125,
        2.0,
        0.03125,
        0.0,
        0.0625,
        0.0,
        0.0625,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
    ]
    IB, JB, VB = findnz(B)
    @test IB == IA
    @test JB == JA
    @test VB == [
        0.0,
        0.03125,
        0.0,
        0.03125,
        0.0,
        0.0625,
        0.0,
        0.0625,
        0.03125,
        0.03125,
        0.0,
        0.03125,
        0.0,
        0.03125,
        0.0,
        0.0625,
        0.0,
        0.0625,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
    ]

    # Test `assemble_dirichlet_vector`
    mesh = rectangle_mesh(4, 3)
    f(x, t) = x[2]
    U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, Dict("xmin" => f))
    V = TestFESpace(U)
    x = Bcube.assemble_dirichlet_vector(U, V, mesh)
    @test x == sparsevec([3, 9], [0.5, 1.0], 12)
end
