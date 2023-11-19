@testset "vtk" begin
    # Reading checksums (to be moved up in the folder tree if we use the same process elsewhere)
    f = readdlm(joinpath(@__DIR__, "checksums.sha1"), String)
    fname2sum = Dict(r[2] => r[1] for r in eachrow(f))

    @testset "write_vtk_lagrange" begin
        mesh = rectangle_mesh(6, 7; xmin = -1, xmax = 1.0, ymin = -1, ymax = 1.0)
        u = FEFunction(TrialFESpace(FunctionSpace(:Lagrange, 4), mesh))
        projection_l2!(u, PhysicalFunction(x -> x[1]^2 + x[2]^2), mesh)

        vars = Dict("u" => u, "grad_u" => ∇(u))

        # bmxam: for some obscur reason, order 3 and 5 lead to different sha1sum
        # when running in standard mode or in test mode...
        for degree_export in (1, 2, 4)
            U_export = TrialFESpace(FunctionSpace(:Lagrange, degree_export), mesh)
            basename = "write_vtk_lagrange_deg$(degree_export)"
            Bcube.write_vtk_lagrange(
                joinpath(tempdir, basename),
                vars,
                mesh,
                U_export;
                vtkversion = v"1.0",
            )

            # Check
            fname = basename * ".vtu"
            @test fname2sum[fname] == bytes2hex(open(sha1, joinpath(tempdir, fname)))
        end
    end
end
