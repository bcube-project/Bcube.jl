import Bcube: shape_functions, grad_shape_functions

@testset "Taylor" begin

    # TODO : add more test for gradients, waiting for Ghislain's commit

    @testset "Line" begin
        shape = Line()

        interp = FunctionSpace(:Taylor, 0)
        λ = x -> shape_functions(interp, shape, x)
        ∇λ = x -> grad_shape_functions(interp, shape, x)
        @test λ(rand())[1] ≈ 1.0
        @test ∇λ(rand())[1] ≈ 0.0
        λ = Bcube.shape_functions_vec(interp, Val(1), shape, [0.0])
        @test λ == [1.0]

        interp = FunctionSpace(:Taylor, 1)
        λ = x -> shape_functions(interp, shape, x)
        ∇λ = x -> grad_shape_functions(interp, shape, x)
        @test λ(-1) ≈ [1.0, -0.5]
        @test λ(1) ≈ [1.0, 0.5]
        λ = Bcube.shape_functions_vec(interp, Val(1), shape, [0.0])
        @test λ == [1.0, 0.0]
    end

    @testset "Square" begin
        shape = Square()

        interp = FunctionSpace(:Taylor, 0)
        λ = x -> shape_functions(interp, shape, x)
        ∇λ = x -> grad_shape_functions(interp, shape, x)
        @test λ(rand(2))[1] ≈ 1.0
        λ = Bcube.shape_functions_vec(interp, Val(1), shape, [0.0, 0.0])
        @test λ == [1.0]

        interp = FunctionSpace(:Taylor, 1)
        λ = x -> shape_functions(interp, shape, x)
        ∇λ = x -> grad_shape_functions(interp, shape, x)
        @test λ([0.0, 0.0]) ≈ [1.0, 0.0, 0.0]
        @test λ([1.0, 0.0]) ≈ [1.0, 0.5, 0.0]
        @test λ([0.0, 1.0]) ≈ [1.0, 0.0, 0.5]
        λ = Bcube.shape_functions_vec(interp, Val(1), shape, [0.0, 0.0])
        @test λ == [1.0, 0.0, 0.0]
        λ = Bcube.shape_functions_vec(interp, Val(1), shape)
        @test λ[1]([0.0, 0.0]) == 1.0
        @test λ[2]([0.0, 0.0]) == 0.0
        @test λ[3]([0.0, 0.0]) == 0.0
        λ = Bcube.shape_functions_vec(interp, Val(2), shape)
        @test λ[1]([0.0, 0.0]) == [1.0, 0.0]
        @test λ[2]([0.0, 0.0]) == [0.0, 0.0]
        @test λ[3]([0.0, 0.0]) == [0.0, 0.0]
        @test λ[4]([0.0, 0.0]) == [0.0, 1.0]
        @test λ[5]([0.0, 0.0]) == [0.0, 0.0]
        @test λ[6]([0.0, 0.0]) == [0.0, 0.0]
    end
end
