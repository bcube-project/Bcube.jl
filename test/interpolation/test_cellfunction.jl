function test_CellPoint(cellinfo, x_ref, x_phys)
    p_ref = Bcube.CellPoint(x_ref, cellinfo, Bcube.ReferenceDomain())
    @test Bcube.change_domain(p_ref, Bcube.ReferenceDomain()) == p_ref

    p_phys = Bcube.change_domain(p_ref, Bcube.PhysicalDomain())
    @test p_phys == Bcube.CellPoint(x_phys, cellinfo, Bcube.PhysicalDomain())

    @test Bcube.change_domain(p_phys, Bcube.PhysicalDomain()) == p_phys

    p_ref_bis = Bcube.change_domain(p_phys, Bcube.ReferenceDomain())
    @test p_ref_bis == p_ref

    nothing
end

@testset "CellFunction" begin
    lx, ly = 20.0, 10.0
    mesh = one_cell_mesh(:quad; xmin = 0.0, xmax = lx, ymin = 0.0, ymax = ly)
    x = [lx / 2, ly / 2]
    cellinfo = CellInfo(mesh, 1)

    x1_ref  = SA[0.0, 0.0]
    x1_phys = SA[lx / 2, ly / 2]

    x2_ref  = SA[-3 / 4, 1.0]
    x2_phys = SA[lx / 8, ly]

    @testset "CellPoint" begin
        @testset "One point" begin
            test_CellPoint(cellinfo, x1_ref, x1_phys)
            test_CellPoint(cellinfo, x2_ref, x2_phys)
        end
        @testset "Tuple of points" begin
            test_CellPoint(cellinfo, (x1_ref, x2_ref), (x1_phys, x2_phys))
        end
        @testset "Vector of points" begin
            test_CellPoint(cellinfo, SA[x1_ref, x2_ref], SA[x1_phys, x2_phys])
        end
    end

    @testset "PhysicalFunction" begin
        a = Bcube.CellPoint(SA[x1_ref, x2_ref], cellinfo, Bcube.ReferenceDomain())
        b = Bcube.CellPoint(SA[x1_phys, x2_phys], cellinfo, Bcube.PhysicalDomain())
        _f = x -> x[1] + x[2]
        f = PhysicalFunction(_f)
        @test f(a) == _f.(SA[x1_phys, x2_phys])
        @test f(b) == _f.(SA[x1_phys, x2_phys])

        a = Bcube.CellPoint((x1_ref, x2_ref), cellinfo, Bcube.ReferenceDomain())
        b = Bcube.CellPoint((x1_phys, x2_phys), cellinfo, Bcube.PhysicalDomain())
        @test f(a) == _f.((x1_phys, x2_phys))
        @test f(b) == _f.((x1_phys, x2_phys))
    end

    @testset "CellFunction" begin
        a = Bcube.CellPoint((x1_ref, x2_ref), cellinfo, Bcube.ReferenceDomain())
        b = Bcube.CellPoint((x1_phys, x2_phys), cellinfo, Bcube.PhysicalDomain())
        _f = x -> x[1] + x[2]

        λref = Bcube.CellFunction(_f, Bcube.ReferenceDomain())
        @test λref(a) == _f.((x1_ref, x2_ref))

        λphys = Bcube.CellFunction(_f, Bcube.PhysicalDomain())
        @test λphys(a) == _f.((x1_phys, x2_phys))
    end
end
