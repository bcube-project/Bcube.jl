@testset "vtk" begin
    # Reading checksums (to be moved up in the folder tree if we use the same process elsewhere)
    f = readdlm(joinpath(@__DIR__, "checksums.sha1"), String)
    fname2sum = Dict(r[2] => r[1] for r in eachrow(f))

    tempdir = mktempdir()

    @testset "write_vtk_lagrange" begin
        mesh = rectangle_mesh(6, 7; xmin = -1, xmax = 1.0, ymin = -1, ymax = 1.0)
        u = FEFunction(TrialFESpace(FunctionSpace(:Lagrange, 4), mesh))
        projection_l2!(u, PhysicalFunction(x -> x[1]^2 + x[2]^2), mesh)

        vars = Dict("u" => u, "grad_u" => ∇(u))

        for degree_export in 1:5
            U_export = TrialFESpace(FunctionSpace(:Lagrange, degree_export), mesh)
            basename = "write_vtk_lagrange_$(degree_export)"
            Bcube.write_vtk_lagrange(joinpath(tempdir, basename), vars, mesh, U_export)

            # Check
            fname = Bcube._build_fname_with_iterations(basename, 0) * ".vtu"
            @test fname2sum[fname] == bytes2hex(open(sha1, joinpath(tempdir, fname)))
        end
    end
end
