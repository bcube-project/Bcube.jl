@testset "FEFunction" begin
    @testset "SingleFEFunction" begin
        mesh = line_mesh(3)
        U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
        u = FEFunction(U, 3.0)
        @test all(get_dof_values(u, 2) .== [3.0, 3.0])

        # Init with LazyOp - Order 0, size 1, continuous
        mesh = line_mesh(4; xmax = 6)
        fs = FunctionSpace(:Lagrange, 0)
        U = TrialFESpace(fs, mesh)
        f = PhysicalFunction(x -> x[1])
        u = FEFunction(U, mesh, f)
        @test u.dofValues == [1.0, 3.0, 5.0]

        # Init with LazyOp -  Order 1, size 1, continuous
        mesh = line_mesh(4; xmax = 3)
        fs = FunctionSpace(:Lagrange, 1)
        U = TrialFESpace(fs, mesh)
        f = PhysicalFunction(x -> x[1])
        u = FEFunction(U, mesh, f)
        @test u.dofValues == [0.0, 1.0, 2.0, 3.0]

        # Init with LazyOp -  Order 0, size 2, continuous
        mesh = line_mesh(4; xmax = 6)
        fs = FunctionSpace(:Lagrange, 0)
        U = TrialFESpace(fs, mesh; size = 2)
        f = PhysicalFunction(x -> [x[1], x[1]^2])
        u = FEFunction(U, mesh, f)
        c = Bcube.CellInfo(mesh, 2)
        op = Bcube.materialize(u, c)
        @test Bcube.materialize(op, Bcube.CellPoint([0.0], c, Bcube.ReferenceDomain())) ==
              [3.0, 9.0]

        # Init with LazyOp -  Order 1, size 2, continuous
        mesh = line_mesh(4; xmax = 3)
        fs = FunctionSpace(:Lagrange, 1)
        U = TrialFESpace(fs, mesh; size = 2)
        f = PhysicalFunction(x -> [x[1], x[1]^2])
        u = FEFunction(U, mesh, f)
        c = Bcube.CellInfo(mesh, 1)
        op = Bcube.materialize(u, c)
        @test Bcube.materialize(op, Bcube.CellPoint([1.0], c, Bcube.ReferenceDomain())) ==
              [1.0, 1.0]
        @test Bcube.materialize(op, Bcube.CellPoint([-1.0], c, Bcube.ReferenceDomain())) ==
              [0.0, 0.0]
        c = Bcube.CellInfo(mesh, 2)
        op = Bcube.materialize(u, c)
        @test Bcube.materialize(op, Bcube.CellPoint([-1.0], c, Bcube.ReferenceDomain())) ==
              [1.0, 1.0]
        @test Bcube.materialize(op, Bcube.CellPoint([1.0], c, Bcube.ReferenceDomain())) ==
              [2.0, 4.0]

        # Order 1, size 2, discontinuous
        mesh = line_mesh(2; xmin = 1.0, xmax = 2)
        fs = FunctionSpace(:Lagrange, 1)
        U = TrialFESpace(fs, mesh, :discontinuous; size = 3)
        f = PhysicalFunction(x -> [x[1], x[1]^2, x[1]^3])
        u = FEFunction(U, mesh, f)
        @test u.dofValues == [1.0, 2.0, 1.0, 4.0, 1.0, 8.0]

        # Order 1, size 2, discontinuous
        mesh = line_mesh(4; xmax = 3)
        fs = FunctionSpace(:Lagrange, 1)
        U = TrialFESpace(fs, mesh, :discontinuous; size = 2)
        f = PhysicalFunction(x -> [x[1], x[1]^2])
        u = FEFunction(U, mesh, f)
        c = Bcube.CellInfo(mesh, 1)
        op = Bcube.materialize(u, c)
        @test Bcube.materialize(op, Bcube.CellPoint([1.0], c, Bcube.ReferenceDomain())) ==
              [1.0, 1.0]
        @test Bcube.materialize(op, Bcube.CellPoint([-1.0], c, Bcube.ReferenceDomain())) ==
              [0.0, 0.0]
        c = Bcube.CellInfo(mesh, 2)
        op = Bcube.materialize(u, c)
        @test Bcube.materialize(op, Bcube.CellPoint([-1.0], c, Bcube.ReferenceDomain())) ==
              [1.0, 1.0]
        @test Bcube.materialize(op, Bcube.CellPoint([1.0], c, Bcube.ReferenceDomain())) ==
              [2.0, 4.0]
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

        c1 = Bcube.CellInfo(mesh, 1)
        p1 = Bcube.CellPoint(SA[0.0], c1, Bcube.ReferenceDomain())
        up1_ref = (1.5, 3 * im)

        uc1 = Bcube.materialize(u, c1)
        @test uc1 isa Bcube.LazyWrap{<:NTuple{2, Bcube.AbstractCellFunction}}
        @test all(Base.materialize(uc1, p1) .≈ up1_ref)

        expr1 = 2 .* (u...,) .+ 1
        expr2 = ((u...) -> 2 .* u .+ 1) ∘ u
        for expr in (expr1, expr2)
            expr_c1 = Bcube.materialize(expr, c1)
            @test all(Base.materialize(expr_c1, p1) .≈ (2.0 .* up1_ref .+ 1))
        end

        u3 = FEFunction(U, [1.0, 2.0, 3.0])
        f3(u, ∇u) = @. 2 * u + ∇u ⋅ SA[10.0]
        expr3 = f3 ∘ (u3, ∇(u3))
        expr3_c1 = Bcube.materialize(expr3, c1)
        val3 = Base.materialize(expr3_c1, p1)
        val3_ref = (2.0 .* (1.5, 3) .+ (SA[0.5], SA[0.0]) .⋅ SA[10.0])
        @test all(val3 .≈ val3_ref)

        # Test initialization with tuple of AbstractLazy
        f1 = PhysicalFunction(x -> x[1])
        f2 = PhysicalFunction(x -> x[1]^2)
        u_lazy = FEFunction(U, mesh, (f1, f2))
        @test get_dof_values(u_lazy) ≈ [0.0, 1.0, 0.25]
    end
end
