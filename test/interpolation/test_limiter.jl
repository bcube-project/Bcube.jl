@testset "Limiter" begin
    @testset "LinearScalingLimiter" begin
        Lx = 3.0
        Ly = 1.0
        mesh = rectangle_mesh(4, 2; xmax = Lx, ymax = Ly)

        function f(k, x)
            if x[1] < 1.0
                return 0.0
            elseif x[1] > 2.0
                return 1.0
            else
                return k * (x[1] - 1.5) + 0.5
            end
        end
        degree = 1
        fs = FunctionSpace(Bcube.Lagrange(:Legendre), degree + 1)
        fes = TrialFESpace(fs, mesh, :discontinuous; size = 1) # DG, scalar
        Ω = CellDomain(mesh)
        dΩ = Measure(Ω, 2 * degree + 1)
        dΓ = Measure(InteriorFaceDomain(mesh), 2 * degree + 1)

        for k in [1, 2]
            u = FEFunction(fes, mesh, PhysicalFunction(x -> f(k, x)))
            ũ = cell_mean(u, dΩ)
            @test get_values(ũ) ≈ [0.0, 0.5, 1.0]

            # default target measures are (dΩ, dΓ, dΓbc): check that the
            # three ways of providing (or not) the target measures agree
            lim₁, u_lim = linear_scaling_limiter(u, ũ, Ω)
            lim₂, _ = linear_scaling_limiter(u, ũ, Ω; targetMeasures = (dΩ, dΓ))
            lim₃, _ = linear_scaling_limiter(u, ũ, dΩ)
            @test get_values(lim₁) ≈ [0.0, 1.0 / k, 0.0]
            @test get_values(lim₂) ≈ get_values(lim₁)
            @test get_values(lim₃) ≈ get_values(lim₁)

            # `u_lim` is a lazy expression: check its type, project it, and
            # verify the local maximum principle. The admissible windows are
            # deduced from the cell means (0.0, 0.5, 1.0) of the 3-cell mesh.
            @test u_lim isa Bcube.AbstractLazy
            u_limited = FEFunction(fes)
            projection_l2!(u_limited, u_lim, mesh)
            windows = [(0.0, 0.5), (0.0, 1.0), (0.5, 1.0)]
            foreach_element(Ω) do cInfo, _, _
                i = cellindex(cInfo)
                uᵢ = materialize(u_limited, cInfo)
                quadrule = QuadratureRule(shape(celltype(cInfo)), 2 * degree + 1)
                values = [
                    uᵢ(CellPoint(ξ, cInfo, ReferenceDomain())) for ξ in get_nodes(quadrule)
                ]
                @test minimum(values) ≥ windows[i][1] - 1e-12
                @test maximum(values) ≤ windows[i][2] + 1e-12
            end

            # test that `bounds` doesn't affect the result when values
            # are not restrictive
            limᵤ, u_lim = linear_scaling_limiter(u, ũ, Ω; bounds = (0, 1))
            @test get_values(ũ) ≈ [0.0, 0.5, 1.0]
            @test get_values(limᵤ) ≈ [0.0, 1.0 / k, 0.0]

            # test that `bounds` affect the result when values
            # are restrictive
            limᵤ, u_lim = linear_scaling_limiter(u, ũ, Ω; bounds = (0.4, 0.9))
            @test get_values(ũ) ≈ [0.0, 0.5, 1.0]
            @test get_values(limᵤ) ≈ [0.0, (0.4 - 0.5) / (k * (1 - 1.5)), 0.0]

            # test that limiter reduces to a 1st-order scheme if `bounds`
            # imposes constraints that are not even satisfied by
            # cell mean values
            limᵤ, u_lim = linear_scaling_limiter(u, ũ, Ω; bounds = (10, 20))
            @test get_values(ũ) ≈ [0.0, 0.5, 1.0]
            @test get_values(limᵤ) ≈ [0.0, 0.0, 0.0]

            # with bounds that are not even satisfied by the cell means, the
            # limiter degenerates to a first-order (piecewise constant) field
            u_limited = FEFunction(fes)
            projection_l2!(u_limited, u_lim, mesh)
            means = get_values(ũ)
            foreach_element(Ω) do cInfo, _, _
                i = cellindex(cInfo)
                uᵢ = materialize(u_limited, cInfo)
                quadrule = QuadratureRule(shape(celltype(cInfo)), 2 * degree + 1)
                values = [
                    uᵢ(CellPoint(ξ, cInfo, ReferenceDomain())) for ξ in get_nodes(quadrule)
                ]
                @test all(values .≈ means[i])
            end

            # `coefmax` caps the limiter coefficient
            limᵤ, _ = linear_scaling_limiter(u, ũ, Ω; coefmax = 0.3)
            @test get_values(limᵤ) ≈ [0.0, 0.3, 0.0]

            # `DMPrelax` widens the admissible window: the middle cell window
            # [0, 1] becomes [-relax, 1+relax] around the mean 0.5, so the
            # limiter saturates at 1 when 2relax ≥ k*0.5 - 0.5
            limᵤ, _ = linear_scaling_limiter(u, ũ, Ω; DMPrelax = 0.25)
            @test get_values(limᵤ) ≈ [1.0, min(1.0, (1 + 2 * 0.25) / k), 1.0]
        end

        # `checkvalues=true` is a safety net: it errors when a cell mean is
        # outside the in-cell min/max or outside the admissible window. Both
        # ranges being initialized from the cell means themselves, this cannot
        # trigger through the public API with well-defined values; test the
        # internal check directly, and that the flag doesn't alter the result
        u = FEFunction(fes, mesh, PhysicalFunction(x -> f(1, x)))
        ũ = cell_mean(u, dΩ)
        @test_throws ErrorException Bcube._compute_scalar_limiter(
            0.5, # v̅ᵢ outside [mᵢ, Mᵢ] = [0.6, 0.8]
            0.6,
            0.8,
            0.0,
            1.0,
            1.0,
            true,
        )
        @test Bcube._compute_scalar_limiter(0.5, 0.4, 0.6, 0.0, 1.0, 1.0, true) ≈ 1.0
        limᵤ, _ = linear_scaling_limiter(u, ũ, Ω; checkvalues = true)
        @test get_values(limᵤ) ≈ [0.0, 1.0, 0.0]

        # inconsistent `u_mean` (not the mean of `u`): no error, the limiter
        # only relies on the provided values
        ũ_bad = MeshCellData([5.0, 0.5, 1.0])
        limᵤ, _ = linear_scaling_limiter(u, ũ_bad, Ω)
        @test get_values(limᵤ) == [0.0, 0.0, 0.0]

        # error for a continuous FESpace
        fes_c = TrialFESpace(fs, mesh; size = 1)
        u_c = FEFunction(fes_c, mesh, PhysicalFunction(x -> f(1, x)))
        @test_throws AssertionError linear_scaling_limiter(u_c, ũ, Ω)

        # error for negative DMPrelax
        @test_throws AssertionError linear_scaling_limiter(u, ũ, Ω; DMPrelax = -0.1)

        # error when periodic BCs are provided
        Γp = BoundaryFaceDomain(
            mesh,
            PeriodicBCType(Translation(SA[3.0, 0.0]), "xmin", "xmax"),
        )
        @test_throws ErrorException linear_scaling_limiter(u, ũ, Ω; periodicBCs = (Γp,))

        # error when a target measure is defined on a periodic domain
        dΓp = Measure(Γp, 2 * degree + 1)
        @test_throws ErrorException linear_scaling_limiter(
            u,
            ũ,
            Ω;
            targetMeasures = (dΩ, dΓp),
        )
    end

    @testset "DefaultTargetMeasures" begin
        Lx = 3.0
        Ly = 1.0
        mesh = rectangle_mesh(4, 2; xmax = Lx, ymax = Ly)
        degree = 1
        fs = FunctionSpace(Bcube.Lagrange(:Legendre), degree + 1)
        fes = TrialFESpace(fs, mesh, :discontinuous; size = 1)
        Ω = CellDomain(mesh)
        dΩ = Measure(Ω, 2 * degree + 1)
        u = FEFunction(fes, mesh, PhysicalFunction(x -> x[1]))

        # from a domain: (cells, interior faces, boundary faces), the cell
        # measure quadrature type follows the Lagrange space and its degree
        # is the degree of the function space
        tms = Bcube.default_target_measures(u, Ω)
        @test length(tms) == 3
        @test get_domain(tms[1]) === Ω
        @test get_domain(tms[2]) isa InteriorFaceDomain
        @test get_domain(tms[3]) isa BoundaryFaceDomain
        @test Bcube.get_quadrature(tms[1]) isa Quadrature{<:QuadratureLegendre}
        @test get_degree(Bcube.get_quadrature(tms[1])) == get_degree(fs)

        # from a measure: the cell measure is kept as is and the face measures
        # share its quadrature
        tms = Bcube.default_target_measures(u, dΩ)
        @test tms[1] === dΩ
        @test Bcube.get_quadrature(tms[2]) == Bcube.get_quadrature(dΩ)
        @test Bcube.get_quadrature(tms[3]) == Bcube.get_quadrature(dΩ)

        # the quadrature type follows the Lagrange type of the space
        u_lob = FEFunction(
            TrialFESpace(
                FunctionSpace(Bcube.Lagrange(:Lobatto), degree + 1),
                mesh,
                :discontinuous;
                size = 1,
            ),
            mesh,
            PhysicalFunction(x -> x[1]),
        )
        @test Bcube.get_quadrature(Bcube.default_target_measures(u_lob, Ω)[1]) isa
              Quadrature{<:QuadratureLobatto}
        u_uni = FEFunction(
            TrialFESpace(
                FunctionSpace(Bcube.Lagrange(:Uniform), degree + 1),
                mesh,
                :discontinuous;
                size = 1,
            ),
            mesh,
            PhysicalFunction(x -> x[1]),
        )
        @test Bcube.get_quadrature(Bcube.default_target_measures(u_uni, Ω)[1]) isa
              Quadrature{<:QuadratureUniform}

        # on a mesh without boundary names, only (cells, interior faces) are built
        mesh_nobc = Mesh(
            get_nodes(mesh),
            Bcube.cells(mesh),
            Bcube.connectivities(mesh, :c2n).indices,
        )
        u_nobc = FEFunction(fes, mesh_nobc, PhysicalFunction(x -> x[1]))
        Ω_nobc = CellDomain(mesh_nobc)
        tms_nobc = Bcube.default_target_measures(u_nobc, Ω_nobc)
        @test length(tms_nobc) == 2
        limᵤ, _ = linear_scaling_limiter(
            u_nobc,
            cell_mean(u_nobc, Measure(Ω_nobc, 2 * degree + 1)),
            Ω_nobc,
        )
        @test get_values(limᵤ) ≈ [0.0, 1.0, 0.0]

        # boundary faces are required to "see" the extrema reached on the cell
        # boundaries: u = c(x) + 2(y - 1/2) is linear in y, invisible to the
        # Legendre nodes of dΩ (strictly inside the cells) and dΓ (in x only),
        # but caught on the y-boundary faces
        c(x) = x[1] < 1 ? 0.0 : (x[1] > 2 ? 1.0 : 0.5)
        u_y = FEFunction(fes, mesh, PhysicalFunction(x -> c(x) + 2 * (x[2] - 0.5)))
        ũ_y = cell_mean(u_y, dΩ)
        dΓ = Measure(InteriorFaceDomain(mesh), 2 * degree + 1)
        lim_bc, _ = linear_scaling_limiter(u_y, ũ_y, Ω) # default: includes dΓbc
        lim_no, _ = linear_scaling_limiter(u_y, ũ_y, Ω; targetMeasures = (dΩ, dΓ))
        @test get_values(lim_bc) ≈ [0.0, 0.5, 0.0]
        @test get_values(lim_no) ≈ [0.0, sqrt(3) / 2, 0.0]
        @test get_values(lim_bc)[2] < get_values(lim_no)[2]
    end
end
