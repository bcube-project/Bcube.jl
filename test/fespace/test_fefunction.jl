@testset "FEFunction" begin
    @testset "SingleFEFunction" begin
        mesh = line_mesh(3)
        U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
        u = FEFunction(U, 3.0)
        @test all(get_dof_values(u, 2) .== [3.0, 3.0])
    end

    @testset "MultiFEFunction" begin
        mesh = one_cell_mesh(:line)
        U1 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
        U2 = TrialFESpace(FunctionSpace(:Lagrange, 0), mesh)
        U = MultiFESpace(U1, U2)
        vals = [1.0, 2.0, 3.0 * im]
        u = FEFunction(U, vals)

        @test all(Bcube.get_dof_type(u) .== (ComplexF64, ComplexF64))
        @test all(get_dof_values(u) .== vals)
    end
end