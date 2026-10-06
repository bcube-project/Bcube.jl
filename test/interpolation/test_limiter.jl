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
        dΓ = Measure(InteriorFaceDomain(mesh), 2*degree+1)

        for k in [1, 2]
            u = FEFunction(fes, mesh, PhysicalFunction(x -> f(k, x)))
            ũ = cell_mean(u, dΩ)
            limᵤ, u_lim = linear_scaling_limiter(u, ũ, Ω, (dΩ, dΓ))
            @test get_values(ũ) ≈ [0.0, 0.5, 1.0]
            @test get_values(limᵤ) ≈ [0.0, 1.0 / k, 0.0]

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
            limᵤ, u_lim = linear_scaling_limiter(u, ũ, Ω, (dΩ, dΓ); bounds = (0, 1))
            @test get_values(ũ) ≈ [0.0, 0.5, 1.0]
            @test get_values(limᵤ) ≈ [0.0, 1.0 / k, 0.0]

            # test that `bounds` affect the result when values
            # are restrictive
            limᵤ, u_lim = linear_scaling_limiter(u, ũ, Ω, (dΩ, dΓ); bounds = (0.4, 0.9))
            @test get_values(ũ) ≈ [0.0, 0.5, 1.0]
            @test get_values(limᵤ) ≈ [0.0, (0.4-0.5) / (k*(1-1.5)), 0.0]

            # test that limiter reduces to a 1st-order scheme if `bounds`
            # imposes constraints that are not even satisfied by
            # cell mean values
            limᵤ, u_lim = linear_scaling_limiter(u, ũ, Ω, (dΩ, dΓ); bounds = (10, 20))
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
        end
    end
end
