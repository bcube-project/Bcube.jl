@testset "domain" begin
    nx = 10
    ny = 10
    l = 1.0
    path = joinpath(tempdir, "mesh.msh")
    gen_rectangle_mesh(path, :quad; nx = nx, ny = ny, lx = l, ly = l, xc = 0.0, yc = 0.0)
    mesh = read_msh(path)

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
