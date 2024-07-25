@testset "vtk" begin
    @testset "write_vtk" begin
        mesh = rectangle_mesh(10, 10; xmax = 2π, ymax = 2π)
        val_sca = var_on_vertices(PhysicalFunction(x -> cos(x[1]) * sin(x[2])), mesh)
        val_vec = var_on_vertices(PhysicalFunction(x -> SA[cos(x[1]), sin(x[2])]), mesh)
        basename = "write_vtk_rectangle"
        write_vtk(
            joinpath(tempdir, basename),
            1,
            0.0,
            mesh,
            Dict(
                "u" => (val_sca, WriteVTK.VTKPointData()),
                "v" => (transpose(val_vec), WriteVTK.VTKPointData()),
            ),
        )
        fname = Bcube._build_fname_with_iterations(basename, 1) * ".vtu"
        @test fname2sum[fname] == bytes2hex(open(sha1, joinpath(tempdir, fname)))

        basename = "write_vtk_mesh"
        write_vtk(joinpath(tempdir, basename), basic_mesh())
        fname = Bcube._build_fname_with_iterations(basename, 1) * ".vtu"
        @test fname2sum[fname] == bytes2hex(open(sha1, joinpath(tempdir, fname)))
    end

    @testset "write_vtk_lagrange" begin
        mesh = rectangle_mesh(6, 7; xmin = -1, xmax = 1.0, ymin = -1, ymax = 1.0)
        u = FEFunction(TrialFESpace(FunctionSpace(:Lagrange, 4), mesh))
        projection_l2!(u, PhysicalFunction(x -> x[1]^2 + x[2]^2), mesh)

        vars = Dict("u" => u, "grad_u" => ∇(u))

        # bmxam: for some obscur reason, order 3 and 5 lead to different sha1sum
        # when running in standard mode or in test mode...
        for mesh_degree in (1, 2, 4)
            basename = "write_vtk_lagrange_deg$(mesh_degree)"
            Bcube.write_vtk_lagrange(
                joinpath(tempdir, basename),
                vars,
                mesh;
                mesh_degree,
                discontinuous = false,
                vtkversion = v"1.0",
            )

            # Check
            fname = basename * ".vtu"
            @test fname2sum[fname] == bytes2hex(open(sha1, joinpath(tempdir, fname)))
        end

        # add var MeshCellData :
        quad = Quadrature(4)
        dΩ = Measure(CellDomain(mesh), quad)
        vars["umean"] = Bcube.cell_mean(u, dΩ)
        basename = "write_vtk_lagrange_deg4_with_mean"
        Bcube.write_vtk_lagrange(
            joinpath(tempdir, basename),
            vars,
            mesh;
            mesh_degree = 4,
            discontinuous = false,
            vtkversion = v"1.0",
        )

        # Check
        fname = basename * ".vtu"
        @test fname2sum[fname] == bytes2hex(open(sha1, joinpath(tempdir, fname)))

        basename = "write_vtk_lagrange_deg4_dg_with_mean"
        Bcube.write_vtk_lagrange(
            joinpath(tempdir, basename),
            vars,
            mesh;
            mesh_degree = 4,
            discontinuous = true,
            vtkversion = v"1.0",
        )

        # Check
        fname = basename * ".vtu"
        @test fname2sum[fname] == bytes2hex(open(sha1, joinpath(tempdir, fname)))
    end
end
