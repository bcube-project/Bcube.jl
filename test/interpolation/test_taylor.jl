import Bcube: shape_functions, ∂λξ_∂ξ

@testset "Taylor" begin

    # TODO : add more test for gradients, waiting for Ghislain's commit

    @testset "Line" begin
        shape = Line()

        fs = FunctionSpace(:Taylor, 0)
        λ = ξ -> shape_functions(fs, shape, ξ)
        ∇λ = ξ -> ∂λξ_∂ξ(fs, shape, ξ)
        @test λ(rand())[1] ≈ 1.0
        @test ∇λ(rand())[1] ≈ 0.0
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [0.0])
        @test λ == [1.0]

        fs = FunctionSpace(:Taylor, 1)
        λ = ξ -> shape_functions(fs, shape, ξ)
        ∇λ = ξ -> ∂λξ_∂ξ(fs, shape, ξ)
        @test λ(-1) ≈ [1.0, -0.5]
        @test λ(1) ≈ [1.0, 0.5]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [0.0])
        @test λ == [1.0, 0.0]
    end

    @testset "Square" begin
        shape = Square()

        fs = FunctionSpace(:Taylor, 0)
        λ = ξ -> shape_functions(fs, shape, ξ)
        ∇λ = ξ -> ∂λξ_∂ξ(fs, shape, ξ)
        @test λ(rand(2))[1] ≈ 1.0
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [0.0, 0.0])
        @test λ == [1.0]

        fs = FunctionSpace(:Taylor, 1)
        λ = ξ -> shape_functions(fs, shape, ξ)
        ∇λ = ξ -> ∂λξ_∂ξ(fs, shape, ξ)
        @test λ([0.0, 0.0]) ≈ [1.0, 0.0, 0.0]
        @test λ([1.0, 0.0]) ≈ [1.0, 0.5, 0.0]
        @test λ([0.0, 1.0]) ≈ [1.0, 0.0, 0.5]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [0.0, 0.0])
        @test λ == [1.0, 0.0, 0.0]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape)
        @test λ[1]([0.0, 0.0]) == 1.0
        @test λ[2]([0.0, 0.0]) == 0.0
        @test λ[3]([0.0, 0.0]) == 0.0
        λ = Bcube.shape_functions_vec(fs, Val(2), shape)
        @test λ[1]([0.0, 0.0]) == [1.0, 0.0]
        @test λ[2]([0.0, 0.0]) == [0.0, 0.0]
        @test λ[3]([0.0, 0.0]) == [0.0, 0.0]
        @test λ[4]([0.0, 0.0]) == [0.0, 1.0]
        @test λ[5]([0.0, 0.0]) == [0.0, 0.0]
        @test λ[6]([0.0, 0.0]) == [0.0, 0.0]
    end
end
