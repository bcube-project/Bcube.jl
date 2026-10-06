@testset "Limiter" begin
    @testset "LinearScalingLimiter" begin
        Lx = 3.0
        Ly = 1.0
        mesh = rectangle_mesh(4, 2; xmax = Lx, ymax = Ly)

        c2n = connectivities_indices(mesh, :c2n)
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
        dΩ = Measure(CellDomain(mesh), 2 * degree + 1)
        dΓ = Measure(InteriorFaceDomain(mesh), 2*degree+1)

        for k in [1, 2]
            u = FEFunction(fes, mesh, PhysicalFunction(x -> f(k, x)))
            limᵤ, u_lim, ũ = linear_scaling_limiter(u, dΩ; otherMeasures = (dΓ,))
            @test get_values(ũ) ≈ [0.0, 0.5, 1.0]
            @test get_values(limᵤ) ≈ [0.0, 1.0 / k, 0.0]

            # test that `bounds` doesn't affect the result when values
            # are not restrictive
            limᵤ, u_lim, ũ =
                linear_scaling_limiter(u, dΩ; bounds = (0, 1), otherMeasures = (dΓ,))
            @test get_values(ũ) ≈ [0.0, 0.5, 1.0]
            @test get_values(limᵤ) ≈ [0.0, 1.0 / k, 0.0]

            # test that `bounds` affect the result when values
            # are restrictive
            limᵤ, u_lim, ũ =
                linear_scaling_limiter(u, dΩ; bounds = (0.4, 0.9), otherMeasures = (dΓ,))
            @test get_values(ũ) ≈ [0.0, 0.5, 1.0]
            @test get_values(limᵤ) ≈ [0.0, (0.4-0.5) / (k*(1-1.5)), 0.0]

            # test that limiter reduces to a 1st-order scheme if `bounds`
            # imposes constraints that are not even satisfied by
            # cell mean values
            limᵤ, u_lim, ũ = linear_scaling_limiter(u, dΩ; bounds = (10, 20))
            @test get_values(ũ) ≈ [0.0, 0.5, 1.0]
            @test get_values(limᵤ) ≈ [0.0, 0.0, 0.0]
        end
    end
end
