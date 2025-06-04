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
    end
end
