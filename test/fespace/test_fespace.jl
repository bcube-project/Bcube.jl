@testset "FESpace" begin
    @testset "Misc." begin
        mesh = one_cell_mesh(:line)
        U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, Dict("xmin" => 3.0); size = 2)
        @test Bcube.is_continuous(U)
        @test !Bcube.is_discontinuous(U)
        @test Bcube.get_ncomponents(U) == 2
        bctag = first(Bcube.get_dirichlet_boundary_tags(U))
        @test Bcube.get_dirichlet_values(U, bctag)(0.0, 0.0) == 3.0
    end

    @testset "Sparsity pattern" begin
        # The test consists in assembling the ~most complex bilinear form possible
        # and to check that the non-zeros elements are all in the "built" sparsity
        # pattern.
        mesh = rectangle_mesh(9, 4)
        dΩ = Measure(CellDomain(mesh), 3)
        U1 = TrialFESpace(FunctionSpace(:Lagrange, 2), mesh)
        V1 = TestFESpace(U1)
        U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
        V2 = TestFESpace(U2)
        U = MultiFESpace(U1, U2)
        V = MultiFESpace(V1, V2)
        a1((u1, u2), (v1, v2)) = ∫(u1 * v1 + u1 * v2 + u2 * v1 + u2 * v2)dΩ
        A = assemble_bilinear(a1, U, V)
        A.nzval .= 1.0
        J = Bcube.build_jacobian_sparsity_pattern(U, mesh)
        D = J - A
        @test all(D.nzval .> 0.0) # test that non-zeros elements of "A" are included in "J"

        function f(q, ∇q, v, ∇v)
            u1, u2 = q
            ∇u1, ∇u2 = ∇q
            v1, v2 = v
            ∇v1, ∇v2 = ∇v
            return u1 * v1 + u2 ⋅ v2 + u2 ⋅ ∇v1 + ∇u1 ⋅ v2
        end
        mesh = rectangle_mesh(3, 4)
        dΩ = Measure(CellDomain(mesh), 3)
        U1 = TrialFESpace(FunctionSpace(:Lagrange, 2), mesh)
        V1 = TestFESpace(U1)
        U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh; size = 2)
        V2 = TestFESpace(U2)
        U = MultiFESpace(U1, U2)
        V = MultiFESpace(V1, V2)
        a2(q, v) = ∫(f ∘ (q, map(∇, q), v, map(∇, v)))dΩ
        A = assemble_bilinear(a2, U, V)
        A.nzval .= 1.0
        J = Bcube.build_jacobian_sparsity_pattern(U, mesh)
        D = J - A
        @test all(D.nzval .> 0.0) # test that non-zeros elements of "A" are included in "J"
    end
end
