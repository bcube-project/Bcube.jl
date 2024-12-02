@testset "domain" begin
    nx = 10
    ny = 10
    l = 1.0
    mesh = rectangle_mesh(
        nx,
        ny;
        xmin = -l / 2,
        xmax = l / 2,
        ymin = -l / 2,
        ymax = l / 2,
        bnd_names = ("West", "East", "South", "North"),
    )

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
    res = integrate_on_ref_element(PhysicalFunction(x -> 2 * x), cInfo, Quadrature(Val(2)))
    @test res[1] == 1.0

    @testset "domain-to-mesh" begin
        mesh = rectangle_mesh(3, 4)

        f(x) = norm(x) > 0.5
        ind = Bcube.identify_cells(mesh, f)
        Ω = CellDomain(mesh, ind)

        new_mesh = Bcube.domain_to_mesh(Ω)

        @test nnodes(new_mesh) == 8
        @test ncells(new_mesh) == 3
        @test Bcube.inner_faces(new_mesh) == [3, 6]
        @test Bcube.outer_faces(new_mesh) == [1, 2, 4, 5, 7, 8, 9, 10]
        @test Bcube.boundary_faces(new_mesh, "CLIPPED_BND") == [1, 4, 5]
        @test Bcube.boundary_faces(new_mesh, "xmin") == [8]
        @test Bcube.boundary_faces(new_mesh, "xmax") == [2, 9]
        @test "ymin" ∉ values(boundary_names(new_mesh))
        @test Bcube.boundary_faces(new_mesh, "ymax") == [7, 10]
    end
end
