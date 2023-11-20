@testset "projection" begin
    @testset "1D_Line" begin
        mesh = line_mesh(11; xmin = -1.0, xmax = 1.0)

        fSpace = FunctionSpace(:Lagrange, 1)
        fes = FESpace(fSpace, :continuous)
        u = CellVariable(:u, mesh, fes)

        projector = L2_projector(u)

        f = x -> 2 * x[1] + 3
        F = projection(f, projector)

        c2n = connectivities_indices(mesh, :c2n)
        cellTypes = cells(mesh)

        for icell in 1:ncells(mesh)
            # Alias for cell type
            ct = cellTypes[icell]

            # Alias for nodes
            n = get_nodes(mesh, c2n[icell])

            S1 = n[1].x[1]
            S2 = n[2].x[1]
            S = [S1, S2]

            # Corresponding shape
            s = shape(ct)

            # Loop over line vertices
            for i in 1:2
                @test isapprox(f(S[i]), F[dof(u, icell, 1, i)], atol = 1e-8)
            end
        end

        # f(x) = c*x
        #- build mesh and variable
        xmin = 2.0
        xmax = 7.0
        mesh = one_cell_mesh(:line; xmin, xmax)
        cnodes = get_nodes(mesh)
        ct = cells(mesh)[1]
        Finv = mapping_inv(cnodes, ct)
        xc = center(cnodes, ct)
        fs = FunctionSpace(:Taylor, 1)
        fes = FESpace(fs, :discontinuous)
        u2 = CellVariable(:u, mesh, fes)

        #- create test function and project on solution space
        f(x) = 2 * x[1] + 4.0
        q = projection(f, L2_projector(u2))
        #- compare the result vector + the resulting interpolation function
        @test all(q .≈ [f(xc), ForwardDiff.gradient(f, xc)[1] * (xmax - xmin)]) # q = [f(x0), f'(x0) dx]
        λ = x -> shape_functions(fs, shape(ct), Finv(x))
        uᵢ = interpolate(λ, q)
        @test all(uᵢ([x]) .≈ f([x]) for x in range(xmin, xmax; length = 5))
    end

    @testset "2D_Triangle" begin
        path = joinpath(tempdir, "mesh.msh")
        mesh = read_msh(path)
        fSpace = FunctionSpace(:Lagrange, 1)
        fes = FESpace(fSpace, :continuous)
        u = CellVariable(:u, mesh, fes)
        f = x -> x[1] + x[2]

        c2n = connectivities_indices(mesh, :c2n)
        cellTypes = cells(mesh)

        projector = L2_projector(u)
        F = projection(f, projector)

        for icell in 1:ncells(mesh)
            # Alias for cell type
            ct = cellTypes[icell]

            # Alias for nodes
            n = get_nodes(mesh, c2n[icell])

            S1 = [n[1].x[1], n[1].x[2]]
            S2 = [n[2].x[1], n[2].x[2]]
            S3 = [n[3].x[1], n[3].x[2]]
            S = [S1, S2, S3]

            # Corresponding shape
            s = shape(ct)

            λ = x -> shape_functions(fSpace, s, x)

            # Loop over triangle vertices
            for i in 1:3
                @test isapprox(f(S[i]), F[dof(u, icell, 1, i)], atol = 1e-8)
            end
        end
    end
end
