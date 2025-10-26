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

    @testset "subdomains" begin
        @testset "CellDomain" begin
            mesh = read_mesh(
                joinpath(
                    @__DIR__,
                    "..",
                    "assets",
                    "rectangle-mesh-tri-quad-nx10-ny10.msh22",
                );
                warn = false,
            )
            Ω = CellDomain(mesh)
            subdomains = Bcube.get_subdomains(Ω)
            @test length(subdomains) == 2
            @test sum(sdom -> length(Bcube.get_indices(sdom)), subdomains) == ncells(mesh)
            # subdomain #1
            @test Bcube.get_elementtype(subdomains[1]) == Bcube.Tri3_t()
            @test length(Bcube.get_indices(subdomains[1])) == 112
            @test all(Bcube.get_indices(subdomains[1])[1:5] .== [1, 2, 3, 4, 5])
            @test all(
                Bcube.get_indices(subdomains[1])[(end - 5):end] .==
                [107, 108, 109, 110, 111, 112],
            )
            # subdomain #2
            @test Bcube.get_elementtype(subdomains[2]) == Bcube.Quad4_t()
            @test length(Bcube.get_indices(subdomains[2])) == 45
            @test all(Bcube.get_indices(subdomains[2])[1:5] .== [113, 114, 115, 116, 117])
            @test all(
                Bcube.get_indices(subdomains[2])[(end - 5):end] .==
                [152, 153, 154, 155, 156, 157],
            )
        end

        @testset "FaceDomain" begin
            mesh = read_mesh(
                joinpath(
                    @__DIR__,
                    "..",
                    "assets",
                    "rectangle-mesh-tri-quad-nx10-ny10.msh22",
                );
                warn = false,
            )
            Γ = InteriorFaceDomain(mesh)
            subdomains = Bcube.get_subdomains(Γ)
            @test length(subdomains) == 3
            @test sum(sdom -> length(Bcube.get_indices(sdom)), subdomains) ==
                  length(inner_faces(mesh))
            # subdomain #1
            @test Bcube.get_elementtype(subdomains[1]) ==
                  (Bcube.Bar2_t(), Bcube.Tri3_t(), Bcube.Tri3_t())
            @test length(Bcube.get_indices(subdomains[1])) == 154
            @test all(Bcube.get_indices(subdomains[1])[1:5] .== [1, 2, 3, 4, 5])
            @test all(
                Bcube.get_indices(subdomains[1])[(end - 5):end] .==
                [177, 178, 179, 180, 181, 182],
            )
            # subdomain #2
            @test Bcube.get_elementtype(subdomains[2]) ==
                  (Bcube.Bar2_t(), Bcube.Tri3_t(), Bcube.Quad4_t())
            @test length(Bcube.get_indices(subdomains[2])) == 9
            @test all(Bcube.get_indices(subdomains[2])[1:5] .== [41, 43, 65, 76, 84])
            @test all(
                Bcube.get_indices(subdomains[2])[(end - 5):end] .==
                [76, 84, 91, 93, 135, 141],
            )
            # subdomain #3
            @test Bcube.get_elementtype(subdomains[3]) ==
                  (Bcube.Bar2_t(), Bcube.Quad4_t(), Bcube.Quad4_t())
            @test length(Bcube.get_indices(subdomains[3])) == 76
            @test all(Bcube.get_indices(subdomains[3])[1:5] .== [183, 184, 186, 187, 189])
            @test all(
                Bcube.get_indices(subdomains[3])[(end - 5):end] .==
                [265, 266, 269, 271, 273, 275],
            )
        end
    end
end
