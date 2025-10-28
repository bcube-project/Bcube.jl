@testset "projection" begin
    @testset "1D_Line" begin
        mesh = line_mesh(11; xmin = -1.0, xmax = 1.0)
        Ω = CellDomain(mesh)
        dΩ = Measure(CellDomain(mesh), 2)

        U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
        u = FEFunction(U)

        f = PhysicalFunction(x -> 2 * x[1] + 3)
        projection_l2!(u, f, dΩ)

        function f_element(cInfo)
            _u = materialize(u, cInfo)
            _f = materialize(f, cInfo)
            s = shape(celltype(cInfo))

            # Loop over line vertices
            for ξ in get_coords(s)
                cPoint = CellPoint(ξ, cInfo, ReferenceDomain())
                @test materialize(_u, cPoint) ≈ materialize(_f, cPoint)
            end
        end

        nelts_1 = 0
        for cInfo in DomainIterator(Ω)
            nelts_1 += 1
            f_element(cInfo)
        end

        nelts_2 = 0
        foreach_element(Ω) do cInfo
            nelts_2 += 1
            f_element(cInfo)
        end

        @test nelts_1 == nelts_2 == ncells(mesh)

        # f(x) = c*x
        #- build mesh and variable
        xmin = 2.0
        xmax = 7.0
        mesh = one_cell_mesh(:line; xmin, xmax)
        Ω = CellDomain(mesh)
        dΩ = Measure(CellDomain(mesh), 2)

        cInfo = CellInfo(mesh, 1)
        ctype = celltype(cInfo)
        cnodes = nodes(cInfo)
        xc = center(ctype, cnodes).x # coordinates in PhysicalDomain

        fs = FunctionSpace(:Taylor, 1)
        U = TrialFESpace(fs, mesh, :discontinuous)
        u = FEFunction(U)

        #- create test function and project on solution space
        f = PhysicalFunction(x -> 2 * x[1] + 4.0)
        projection_l2!(u, f, dΩ)

        f_cInfo = materialize(f, cInfo)
        _f = x -> materialize(f_cInfo, CellPoint(x, cInfo, PhysicalDomain()))

        #- compare the result vector + the resulting interpolation function
        @test all(
            get_dof_values(u) .≈
            [_f(xc), ForwardDiff.derivative(_f, xc)[1] * (xmax - xmin)],
        ) # u = [f(x0), f'(x0) dx]
    end

    @testset "2D_Triangle" begin
        mesh = read_mesh(
            joinpath(@__DIR__, "..", "assets", "rectangle-mesh-tri-nx3-ny4.msh22");
            warn = false,
        )
        Ω = CellDomain(mesh)
        dΩ = Measure(Ω, 2)

        fSpace = FunctionSpace(:Lagrange, 1)
        U = TrialFESpace(fSpace, mesh, :continuous)
        u = FEFunction(U)
        f = PhysicalFunction(x -> x[1] + x[2])

        projection_l2!(u, f, dΩ)

        _count_cell = 0
        foreach_element(Ω) do cInfo
            _count_cell += 1
            _u = materialize(u, cInfo)
            _f = materialize(f, cInfo)

            # Corresponding shape
            s = shape(celltype(cInfo))

            # Loop over triangle vertices
            for ξ in get_coords(s)
                cPoint = CellPoint(ξ, cInfo, ReferenceDomain())
                @test materialize(_u, cPoint) ≈ materialize(_f, cPoint)
            end
        end
        @test ncells(mesh) == _count_cell
    end

    @testset "misc" begin
        mesh = one_cell_mesh(:quad)
        T, N = Bcube.get_return_type_and_codim(PhysicalFunction(x -> x[1]), mesh)
        @test N == (1,)
        @test T == Float64
        T, N = Bcube.get_return_type_and_codim(PhysicalFunction(x -> 1), mesh)
        @test N == (1,)
        @test T == Int
        T, N = Bcube.get_return_type_and_codim(PhysicalFunction(x -> x), mesh)
        @test N == (2,)
        @test T == Float64
    end

    @testset "var_on_vertices" begin
        mesh = line_mesh(3)

        # Test 1
        fs = FunctionSpace(:Lagrange, 0)
        U = TrialFESpace(fs, mesh)

        x = [1.0, 1.0]
        u = FEFunction(U, x)

        values = var_on_vertices(u, mesh)
        @test values == [1.0, 1.0, 1.0]

        # Test 2
        fs = FunctionSpace(:Lagrange, 1)
        U = TrialFESpace(fs, mesh)

        x = [1.0, 2.0, 3.0]
        u = FEFunction(U, x)

        values = var_on_vertices(u, mesh)
        @test values == x

        # Test 3
        fs = FunctionSpace(:Lagrange, 1)
        U = TrialFESpace(fs, mesh; size = 2)

        f = PhysicalFunction(x -> [x[1], -1.0 - 2 * x[1]])
        u = FEFunction(U)
        projection_l2!(u, f, mesh)

        values = var_on_vertices(u, mesh)
        @test isapprox_arrays(values[:, 1], [0.0, 0.5, 1.0]; rtol = 1e-15)
        @test isapprox_arrays(values[:, 2], [-1.0, -2.0, -3.0]; rtol = 1e-15)
    end
end
