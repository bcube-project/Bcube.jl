@testset "FESpace" begin
    mesh = one_cell_mesh(:line)
    U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, Dict("xmin" => 3.0); size = 2)
    @test Bcube.is_continuous(U)
    @test !Bcube.is_discontinuous(U)
    @test Bcube.get_ncomponents(U) == 2
    bctag = first(Bcube.get_dirichlet_boundary_tags(U))
    @test Bcube.get_dirichlet_values(U, bctag)(0.0, 0.0) == 3.0
end
