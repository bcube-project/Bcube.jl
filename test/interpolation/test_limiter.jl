@testset "Limiter" begin
    @testset "LinearScalingLimiter" begin
        tmp_path = "tmp.msh"
        Lx = 3.0
        Ly = 1.0
        gen_rectangle_mesh(
            tmp_path,
            :quad;
            nx = 4,
            ny = 2,
            lx = Lx,
            ly = Ly,
            xc = Lx / 2,
            yc = Ly / 2,
        )
        mesh = read_msh(tmp_path, 2) # '2' indicates the space dimension (3 by default)
        rm(tmp_path)

        c2n = connectivities_indices(mesh, :c2n)
        f = (k, x) -> begin
            if x[1] < 1.0
                return 0.0
            elseif x[1] > 2.0
                return 1.0
            else
                return k * (x[1] - 1.5) + 0.5
            end
        end
        degree = 1
        fs = FunctionSpace(:Taylor, degree)
        fes = FESpace(fs, :discontinuous; size = 1) # DG, scalar
        u = CellVariable(:u, mesh, fes)

        for k in [1, 2]
            set_values!(u, x -> f(k, x))
            @test mean_values(u, Val(2 * degree + 1)) ≈ [0.0, 0.5, 1.0]
            limᵤ, ũ = linear_scaling_limiter(u, 2 * degree + 1)
            @test get_values(limᵤ) ≈ [0.0, 1.0 / k, 0.0]
        end
    end
end
