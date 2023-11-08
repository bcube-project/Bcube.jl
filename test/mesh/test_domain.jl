@testset "domain" begin
    nx = 10
    ny = 10
    l = 1.0
    tmp_path = "tmp.msh"
    gen_rectangle_mesh(
        tmp_path,
        :quad;
        nx = nx,
        ny = ny,
        lx = l,
        ly = l,
        xc = 0.0,
        yc = 0.0,
    )
    mesh = read_msh(tmp_path)
    rm(tmp_path)

    @test topodim(mesh) === 2
    periodicBCType_x = PeriodicBCType(Translation(SA[-l, 0.0]), ("East",), ("West",))
    periodicBCType_y = PeriodicBCType(Translation(SA[0.0, l]), ("South",), ("North",))
    Γ_perio_x = BoundaryFaceDomain(mesh, periodicBCType_x)
    Γ_perio_y = BoundaryFaceDomain(mesh, periodicBCType_y)

    # Testing subdomain
    mesh = line_mesh(3; xmin = 0.0, xmax = 2) # mesh with 2 cells
    dΩ = Measure(CellDomain(mesh, 1:1), 2) # only first cell

    #- new api
    cInfo = CellInfo(mesh, 1)
    res = Bcube.integrate_on_ref(PhysicalFunction(x -> 2 * x), cInfo, Quadrature(Val(2)))
    @test res[1] == 1.0
end
