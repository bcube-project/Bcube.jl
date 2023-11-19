@testset "DofHandler" begin
    @testset "2D" begin
        @testset "Discontinuous" begin

            #---- Mesh with one cell
            mesh = one_cell_mesh(:quad)

            # scalar - discontinuous
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 1), 1, false)

            @test Bcube.dof(dhl, 1) == collect(1:4)
            @test Bcube.dof(dhl, 1, 1, 3) == 3
            @test Bcube.ndofs(dhl, 1) == 4

            # two scalar variables sharing same space
            U_sca = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U = MultiFESpace(U_sca, U_sca)

            m = Bcube.get_mapping(U, 2)
            dhl = Bcube._get_dhl(Bcube.get_fespace(U)[2])
            @test m[Bcube.get_dofs(U_sca, 1)] == collect(5:8)
            @test m[Bcube.dof(dhl, 1, 1, 3)] == 7
            @test Bcube.ndofs(dhl, 1) == 4

            # Two scalar variables, different orders
            U1 = TrialFESpace(FunctionSpace(:Lagrange, 2), mesh, :discontinuous)
            U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U = MultiFESpace(U1, U2)

            m1 = Bcube.get_mapping(U, 1)
            m2 = Bcube.get_mapping(U, 2)

            dhl1 = Bcube._get_dhl(Bcube.get_fespace(U)[1])
            dhl2 = Bcube._get_dhl(Bcube.get_fespace(U)[2])

            @test m1[Bcube.get_dofs(U1, 1)] == collect(1:9)
            @test m2[Bcube.get_dofs(U2, 1)] == collect(10:13)
            @test m1[Bcube.dof(dhl1, 1, 1, 3)] == 3
            @test m2[Bcube.dof(dhl2, 1, 1, 3)] == 12
            @test Bcube.ndofs(dhl1, 1) == 9
            @test Bcube.ndofs(dhl2, 1) == 4

            # One vector variable
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 1), 2, false)

            @test ndofs(dhl, 1) == 8
            @test dof(dhl, 1, 1) == collect(1:4)
            @test dof(dhl, 1, 2) == collect(5:8)
            @test dof(dhl, 1, 1, 3) == 3
            @test dof(dhl, 1, 2, 3) == 7

            # Three variables : one scalar, one vector, one scalar
            U_ρ = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U_ρu = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous; size = 3)
            U_ρE = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U = MultiFESpace(U_ρ, U_ρu, U_ρE)

            m_ρ = Bcube.get_mapping(U, 1)
            m_ρu = Bcube.get_mapping(U, 2)
            m_ρE = Bcube.get_mapping(U, 3)

            @test m_ρ[Bcube.get_dofs(U_ρ, 1)] == collect(1:4)
            @test m_ρu[Bcube.get_dofs(U_ρu, 1)] == collect(5:16)
            @test m_ρE[Bcube.get_dofs(U_ρE, 1)] == collect(17:20)

            #---- Basic mesh
            mesh = basic_mesh()

            # One scalar FESpace
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 1), 1, false)

            @test dof(dhl, 1) == collect(1:4)
            @test dof(dhl, 2) == collect(5:8)
            @test dof(dhl, 3) == collect(9:11)
            @test max_ndofs(dhl) == 4

            # Two scalar variables sharing same FESpace
            U_sca = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U = MultiFESpace(U_sca, U_sca)

            m1 = Bcube.get_mapping(U, 1)
            m2 = Bcube.get_mapping(U, 2)

            @test m1[Bcube.get_dofs(U_sca, 1)] == collect(1:4)
            @test m2[Bcube.get_dofs(U_sca, 1)] == collect(5:8)
            @test m1[Bcube.get_dofs(U_sca, 3)] == collect(17:19)
            @test m2[Bcube.get_dofs(U_sca, 3)] == collect(20:22)

            # Two vars, different orders
            U1 = TrialFESpace(FunctionSpace(:Lagrange, 2), mesh, :discontinuous)
            U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U = MultiFESpace(U1, U2)

            m1 = Bcube.get_mapping(U, 1)
            m2 = Bcube.get_mapping(U, 2)

            dhl1 = Bcube._get_dhl(Bcube.get_fespace(U)[1])
            dhl2 = Bcube._get_dhl(Bcube.get_fespace(U)[2])

            @test m1[Bcube.get_dofs(U1, 1)] == collect(1:9)
            @test m2[Bcube.get_dofs(U2, 1)] == collect(10:13)
            @test m1[Bcube.get_dofs(U1, 3)] == collect(27:32)
            @test m2[Bcube.get_dofs(U2, 3)] == collect(33:35)
            @test max_ndofs(dhl1) == 9
        end

        @testset "Continuous" begin
            #---- Mesh with one cell
            mesh = one_cell_mesh(:quad)

            # One scalar
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 1), 1, true)

            @test Bcube.dof(dhl, 1) == collect(1:4)

            # Two scalar variables sharing the same space
            U_sca = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U = MultiFESpace(U_sca, U_sca)

            m1 = Bcube.get_mapping(U, 1)
            m2 = Bcube.get_mapping(U, 2)

            @test m1[Bcube.get_dofs(U_sca, 1)] == collect(1:4)
            @test m2[Bcube.get_dofs(U_sca, 1)] == collect(5:8)

            # Two vars, different orders
            U1 = TrialFESpace(FunctionSpace(:Lagrange, 2), mesh, :discontinuous)
            U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U = MultiFESpace(U1, U2)

            m1 = Bcube.get_mapping(U, 1)
            m2 = Bcube.get_mapping(U, 2)

            dhl1 = Bcube._get_dhl(Bcube.get_fespace(U)[1])
            dhl2 = Bcube._get_dhl(Bcube.get_fespace(U)[2])

            @test m1[Bcube.get_dofs(U1, 1)] == collect(1:9)
            @test m2[Bcube.get_dofs(U2, 1)] == collect(10:13)

            #---- Rectangle mesh
            # (from corrected bug -> checked graphically, by hand!)
            mesh = rectangle_mesh(3, 2; type = :quad)
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 3), 1, true)
            @test Bcube.dof(dhl, 1) == collect(1:16)
            @test Bcube.dof(dhl, 2) ==
                  [4, 17, 18, 19, 8, 20, 21, 22, 12, 23, 24, 25, 16, 26, 27, 28]

            #---- Basic mesh
            mesh = basic_mesh()

            # One scalar
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 1), 1, true)
            @test Bcube.dof(dhl, 1) == [1, 2, 3, 4]
            @test Bcube.dof(dhl, 2) == [2, 5, 4, 6]
            @test Bcube.dof(dhl, 3) == [5, 7, 6]

            # Two scalar variables sharing the same space
            U_sca = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous) # this one should be elsewhere
            U = MultiFESpace(U_sca, U_sca)

            m1 = Bcube.get_mapping(U, 1)
            m2 = Bcube.get_mapping(U, 2)

            @test m1[Bcube.get_dofs(U_sca, 1)] == [1, 2, 3, 4]
            @test m2[Bcube.get_dofs(U_sca, 1)] == [5, 6, 7, 8]
            @test m1[Bcube.get_dofs(U_sca, 2)] == [9, 10, 11, 12]
            @test m2[Bcube.get_dofs(U_sca, 2)] == [13, 14, 15, 16]
            @test m1[Bcube.get_dofs(U_sca, 3)] == [17, 18, 19]
            @test m2[Bcube.get_dofs(U_sca, 3)] == [20, 21, 22]

            #  One scalar variable of order 2 on second order quads
            mesh = rectangle_mesh(3, 2; order = 2)
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 2), 1, true)

            @test Bcube.dof(dhl, 1) == collect(1:9)
            @test Bcube.dof(dhl, 2) == [3, 10, 11, 6, 12, 13, 9, 14, 15]

            # A square domain composed of four (2x2) Quad9
            mesh = rectangle_mesh(3, 3; order = 2)
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 2), 1, true)

            @test Bcube.dof(dhl, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
            @test Bcube.dof(dhl, 2) == [3, 10, 11, 6, 12, 13, 9, 14, 15]
            @test Bcube.dof(dhl, 3) == [7, 8, 9, 16, 17, 18, 19, 20, 21]
            @test Bcube.dof(dhl, 4) == [9, 14, 15, 18, 22, 23, 21, 24, 25]

            # Two quads of order 1 with variable of order > 1
            mesh = rectangle_mesh(3, 2)
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 2), 1, true)

            @test Bcube.dof(dhl, 1) == collect(1:9)
            @test Bcube.dof(dhl, 2) == [3, 10, 11, 6, 12, 13, 9, 14, 15]
        end
    end

    @testset "3D" begin
        @testset "Discontinuous" begin
            #---- Mesh with one cell
            mesh = one_cell_mesh(:cube)

            # One scalar space
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 1), 1, false)

            @test Bcube.dof(dhl, 1) == collect(1:8)
            @test Bcube.dof(dhl, 1, 1, 3) == 3
            @test Bcube.ndofs(dhl, 1) == 8

            # Two scalar variables sharing same space
            U_sca = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous) # this one should be elsewhere
            U = MultiFESpace(U_sca, U_sca)

            m = Bcube.get_mapping(U, 2)
            dhl = Bcube._get_dhl(Bcube.get_fespace(U)[2])

            @test m[Bcube.get_dofs(U_sca, 1)] == collect(9:16)
            @test m[Bcube.dof(dhl, 1, 1, 3)] == 11
            @test Bcube.ndofs(dhl, 1) == 8

            # Two scalar variables, different orders
            # fes_u = FESpace(FunctionSpace(:Lagrange, 2), :discontinuous)
            # fes_v = FESpace(FunctionSpace(:Lagrange, 1), :discontinuous)
            # u = CellVariable(:u, mesh, fes_u)
            # v = CellVariable(:v, mesh, fes_v)
            # sys = System((u, v))

            # @test dof(sys, u, 1) == collect(1:9)
            # @test dof(sys, v, 1) == collect(10:13)
            # @test dof(sys, u, 1, 1, 3) == 3
            # @test dof(sys, v, 1, 1, 3) == 12
            # @test ndofs(get_dofhandler(u), 1) == 9
            # @test ndofs(get_dofhandler(v), 1) == 4

            # One vector variable
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 1), 2, false)

            @test Bcube.ndofs(dhl, 1) == 16
            @test Bcube.dof(dhl, 1, 1) == collect(1:8)
            @test Bcube.dof(dhl, 1, 2) == collect(9:16)
            @test Bcube.dof(dhl, 1, 1, 3) == 3
            @test Bcube.dof(dhl, 1, 2, 3) == 11

            # Three variables : one scalar, one vector, one scalar
            U_ρ = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U_ρu = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous; size = 3)
            U_ρE = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U = MultiFESpace(U_ρ, U_ρu, U_ρE)

            m_ρ = Bcube.get_mapping(U, 1)
            m_ρu = Bcube.get_mapping(U, 2)
            m_ρE = Bcube.get_mapping(U, 3)

            @test m_ρ[Bcube.get_dofs(U_ρ, 1)] == collect(1:8)
            @test m_ρu[Bcube.get_dofs(U_ρu, 1)] == collect(9:32)
            @test m_ρE[Bcube.get_dofs(U_ρE, 1)] == collect(33:40)

            #---- Mesh with 2 cubes side by side
            path = joinpath(tempdir, "mesh.msh")
            Bcube._gen_2cubes_mesh(path)
            mesh = read_msh(path)

            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 2), 1, false)

            @test Bcube.dof(dhl, 1) == collect(1:27)
            @test Bcube.dof(dhl, 2) == collect(28:54)

            #---- Mesh with 4 cubes (pile)
            mesh = read_msh("../input/mesh/cube_pile.msh")

            # One scalar FESpace
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 1), 1, false)

            for icell in 1:ncells(mesh)
                @test Bcube.dof(dhl, icell) == collect((1 + (icell - 1) * 8):(icell * 8))
                @test Bcube.ndofs(dhl, icell) == 8
            end
        end

        @testset "Continuous" begin
            mesh = read_msh("../input/mesh/cube_pile.msh")

            # One scalar FESpace
            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 1), 1, true)
            c2n = connectivities_indices(mesh, :c2n)

            # dof index            01  02  03  04  05  06  07  08  09  10  11  12  13  14  15  16  17  18  19  20
            # corresponding node   01  02  05  04  09  10  11  12  03  06  13  14  08  07  15  16  17  18  19  20
            @test Bcube.dof(dhl, 1) == [1, 2, 3, 4, 5, 6, 7, 8]
            @test Bcube.dof(dhl, 2) == [2, 9, 4, 10, 6, 11, 8, 12]
            @test Bcube.dof(dhl, 3) == [4, 10, 13, 14, 8, 12, 15, 16]
            @test Bcube.dof(dhl, 4) == [6, 11, 8, 12, 17, 18, 19, 20]

            #---- Mesh with 2 cubes side by side
            path = joinpath(tempdir, "mesh.msh")
            Bcube._gen_2cubes_mesh(path)
            mesh = read_msh(path)

            dhl = Bcube.DofHandler(mesh, FunctionSpace(:Lagrange, 2), 1, true)
            @test Bcube.dof(dhl, 1) == collect(1:27)
            @test Bcube.dof(dhl, 2) == [
                3,
                28,
                29,
                6,
                30,
                31,
                9,
                32,
                33,
                12,
                34,
                35,
                15,
                36,
                37,
                18,
                38,
                39,
                21,
                40,
                41,
                24,
                42,
                43,
                27,
                44,
                45,
            ]
        end
    end
end
