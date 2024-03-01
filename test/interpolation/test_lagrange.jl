import Bcube: ndofs, shape_functions, ∂λξ_∂ξ, ∂λξ_∂x, Mesh, Cube, Prism, Tetra
Σ = sum

"""
    test_lagrange_shape_function(shape, degree)

Test if the lagrange shape functions are zeros on all nodes except on "their" corresponding node.
"""
function test_lagrange_shape_function(shape, degree)
    @testset "Degree=$degree" begin
        fs = FunctionSpace(:Lagrange, degree)
        expected = zeros(ndofs(fs, shape))
        for (i, ξ) in enumerate(coords(fs, shape))
            expected .= 0.0
            expected[i] = 1.0
            result = shape_functions(fs, shape, ξ)
            !all(isapprox.(result, expected, atol = 5eps())) &&
                (@show degree, result, expected)
            @test all(isapprox.(result, expected, atol = 5eps()))
        end

        if typeof(shape) ∈ (Line, Square, Cube)
            # `QuadratureLobatto` is only available for `Line`, `Square`, `Cube` for now
            quad = QuadratureRule(shape, Quadrature(QuadratureLobatto(), Val(2)))
            ξquad = get_nodes(quad)
            λref = [shape_functions(fs, shape, ξ) for ξ in ξquad]
            λtest = shape_functions(fs, quad)
            @test all(map(≈, λtest, λref))

            ∇λref = [∂λξ_∂ξ(fs, shape, ξ) for ξ in ξquad]
            ∇λtest = ∂λξ_∂ξ(fs, quad)
            @test all(map(≈, ∇λtest, ∇λref))

            @test isa(Bcube.is_collocated(fs, quad), Bcube.IsNotCollocatedStyle)
            if degree > 0
                @test isa(
                    Bcube.is_collocated(
                        fs,
                        QuadratureRule(shape, Quadrature(QuadratureUniform(), Val(degree))),
                    ),
                    Bcube.IsCollocatedStyle,
                )
            end
        end
    end
end

@testset "Lagrange" begin
    @testset "Line" begin
        shape = Line()

        @test coords(FunctionSpace(:Lagrange, 0), shape) == (SA[0.0],)
        @test coords(FunctionSpace(:Lagrange, 1), shape) == (SA[-1.0], SA[1.0])
        @test coords(FunctionSpace(:Lagrange, 2), shape) == (SA[-1.0], SA[0.0], SA[1.0])

        for deg in 0:2
            test_lagrange_shape_function(shape, deg)
        end

        fs = FunctionSpace(:Lagrange, 0)
        λ = shape_functions(fs, shape)
        ∇λ = ∂λξ_∂ξ(fs, shape)
        @test λ(π)[1] ≈ 1.0
        @test ∇λ(π)[1] ≈ 0.0
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [0.0])
        @test λ == [1.0]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape)
        @test λ[1]([0.0]) == 1.0
        λ = Bcube.shape_functions_vec(fs, Val(2), shape)
        @test λ[1]([0.0]) == [1.0, 0.0]
        @test λ[2]([0.0]) == [0.0, 1.0]

        fs = FunctionSpace(:Lagrange, 1)
        λ = shape_functions(fs, shape)
        ∇λ = ∂λξ_∂ξ(fs, shape)
        @test λ(-1) ≈ [1.0, 0.0]
        @test λ(1) ≈ [0.0, 1.0]
        @test Σ(λ(π)) ≈ 1.0
        @test ∇λ(π) ≈ [-1.0 / 2.0, 1.0 / 2.0]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [0.0])
        @test λ == [0.5, 0.5]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape)
        @test λ[1]([0.0]) == 0.5
        @test λ[2]([0.0]) == 0.5

        fs = FunctionSpace(:Lagrange, 2)
        λ = shape_functions(fs, shape)
        ∇λ = ∂λξ_∂ξ(fs, shape)
        @test λ(-1) ≈ [1.0, 0.0, 0.0]
        @test λ(0) ≈ [0.0, 1.0, 0.0]
        @test λ(1) ≈ [0.0, 0.0, 1.0]
        @test Σ(λ(π)) ≈ 1.0
        @test ∇λ(0.0) ≈ [-1.0 / 2.0, 0.0, 1.0 / 2.0]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [0.0])
        @test λ == [0.0, 1.0, 0.0]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape)
        @test λ[1]([0.0]) == 0.0
        @test λ[2]([0.0]) == 1.0
        @test λ[3]([0.0]) == 0.0

        # Tests for gradients of shape functions expressed on local element
        # TODO: the part below should be moved to test_ref2loc
        mesh = Mesh([Node([0.0]), Node([1.0])], [Bar2_t()], Connectivity([2], [1, 2]))
        cellTypes = cells(mesh)
        ctype = cellTypes[1]
        c2n = connectivities_indices(mesh, :c2n)
        cnodes = get_nodes(mesh, c2n[1])
        Finv = Bcube.mapping_inv(ctype, cnodes)

        fs = FunctionSpace(:Lagrange, 1)
        ∇λ = x -> ∂λξ_∂x(fs, Val(1), ctype, cnodes, Finv(x))
        @test ∇λ(0.5) ≈ reshape([-1.0, 1.0], (2, 1))
    end

    @testset "Triangle" begin
        shape = Triangle()
        @test coords(FunctionSpace(:Lagrange, 0), shape) ≈ (SA[1.0 / 3, 1.0 / 3],)
        @test coords(FunctionSpace(:Lagrange, 1), shape) ==
              (SA[0.0, 0.0], SA[1.0, 0.0], SA[0.0, 1.0])
        @test coords(FunctionSpace(:Lagrange, 2), shape) == (
            SA[0.0, 0.0],
            SA[1.0, 0.0],
            SA[0.0, 1.0],
            SA[0.5, 0.0],
            SA[0.5, 0.5],
            SA[0.0, 0.5],
        )
        @test coords(FunctionSpace(:Lagrange, 3), shape) == (
            SA[0.0, 0.0],
            SA[1.0, 0.0],
            SA[0.0, 1.0],
            SA[1 / 3, 0.0],
            SA[2 / 3, 0.0],
            SA[2 / 3, 1 / 3],
            SA[1 / 3, 2 / 3],
            SA[0.0, 2 / 3],
            SA[0.0, 1 / 3],
            SA[1 / 3, 1 / 3],
        )

        fs = FunctionSpace(:Lagrange, 0)
        λ = x -> shape_functions(fs, shape, x)
        ∇λ = x -> ∂λξ_∂ξ(fs, shape, x)
        @test λ([π, ℯ])[1] ≈ 1.0
        @test ∇λ([π, ℯ]) ≈ [0.0 0.0]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [1.0 / 3.0, 1.0 / 3.0])
        @test λ == [1.0]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape)
        @test λ[1]([1.0 / 3.0, 1.0 / 3.0]) == 1.0

        fs = FunctionSpace(:Lagrange, 1)
        λ = x -> shape_functions(fs, shape, x)
        ∇λ = x -> ∂λξ_∂ξ(fs, shape, x)
        @test λ([0, 0]) ≈ [1.0, 0.0, 0.0]
        @test λ([1, 0]) ≈ [0.0, 1.0, 0.0]
        @test λ([0, 1]) ≈ [0.0, 0.0, 1.0]
        @test Σ(λ([π, ℯ])) ≈ 1.0
        @test ∇λ([π, ℯ]) ≈ [
            -1.0 -1.0
            1.0 0.0
            0.0 1.0
        ]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [1.0 / 3.0, 1.0 / 3.0])
        @test λ ≈ [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape)
        @test λ[1]([1.0 / 3.0, 1.0 / 3.0]) ≈ 1.0 / 3.0
        @test λ[2]([1.0 / 3.0, 1.0 / 3.0]) ≈ 1.0 / 3.0
        @test λ[3]([1.0 / 3.0, 1.0 / 3.0]) ≈ 1.0 / 3.0

        fs = FunctionSpace(:Lagrange, 2)
        λ = x -> shape_functions(fs, shape, x)
        ∇λ = x -> ∂λξ_∂ξ(fs, shape, x)
        @test λ([0, 0])[1:3] ≈ [1.0, 0.0, 0.0]
        @test λ([1, 0])[1:3] ≈ [0.0, 1.0, 0.0]
        @test λ([0, 1])[1:3] ≈ [0.0, 0.0, 1.0]
        @test Σ(λ([π, ℯ])) ≈ 1.0
        @test ∇λ([0, 0]) ≈ [
            -3.0 -3.0
            -1.0 0.0
            0.0 -1.0
            4.0 0.0
            0.0 0.0
            0.0 4.0
        ]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [1.0 / 3.0, 1.0 / 3])
        @test λ ≈ [
            -0.11111111111111112,
            -0.11111111111111112,
            -0.11111111111111112,
            0.44444444444444453,
            0.4444444444444444,
            0.44444444444444453,
        ]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape)
        @test λ[1]([1.0 / 3.0, 1.0 / 3.0]) ≈ -0.111111111111111
        @test λ[2]([1.0 / 3.0, 1.0 / 3.0]) ≈ -0.111111111111111
        @test λ[3]([1.0 / 3.0, 1.0 / 3.0]) ≈ -0.111111111111111
        @test λ[4]([1.0 / 3.0, 1.0 / 3.0]) ≈ 0.444444444444444
        @test λ[5]([1.0 / 3.0, 1.0 / 3.0]) ≈ 0.444444444444444
        @test λ[6]([1.0 / 3.0, 1.0 / 3.0]) ≈ 0.444444444444444

        for deg in 0:3
            test_lagrange_shape_function(shape, deg)
        end

        # Tests for gradients of shape functions expressed on local element
        # Mesh with only one triangle of order 1 : S1,S2,S3
        S1 = [-1.0, -1.0]
        S2 = [1.0, -2.0]
        S3 = [-1.0, 1.0]

        mesh =
            Mesh([Node(S1), Node(S2), Node(S3)], [Tri3_t()], Connectivity([3], [1, 2, 3]))
        cellTypes = cells(mesh)
        ctype = cellTypes[1]
        c2n = connectivities_indices(mesh, :c2n)
        n = get_nodes(mesh, c2n[1])

        vol =
            0.5 * abs((S2[1] - S1[1]) * (S3[2] - S1[2]) - (S3[1] - S1[1]) * (S2[2] - S1[1]))

        fs = FunctionSpace(:Lagrange, 1)
        ∇λ = x -> ∂λξ_∂x(fs, Val(1), ctype, n, x)
        @test ∇λ([0, 0])[1, :] ≈ (0.5 / vol) * ([S2[2] - S3[2], S3[1] - S2[1]])
        @test ∇λ([0, 0])[2, :] ≈ (0.5 / vol) * ([S3[2] - S1[2], S1[1] - S3[1]])
        @test ∇λ([0, 0])[3, :] ≈ (0.5 / vol) * ([S1[2] - S2[2], S2[1] - S1[1]])
    end

    @testset "Square" begin
        shape = Square()
        @test coords(FunctionSpace(:Lagrange, 0), shape) == (SA[0.0, 0.0],)
        @test coords(FunctionSpace(:Lagrange, 1), shape) ==
              (SA[-1.0, -1.0], SA[1.0, -1.0], SA[-1.0, 1.0], SA[1.0, 1.0])
        @test coords(FunctionSpace(:Lagrange, 2), shape) == (
            SA[-1.0, -1.0],
            SA[0.0, -1.0],
            SA[1.0, -1.0],
            SA[-1.0, 0.0],
            SA[0.0, 0.0],
            SA[1.0, 0.0],
            SA[-1.0, 1.0],
            SA[0.0, 1.0],
            SA[1.0, 1.0],
        )

        for deg in 0:3
            test_lagrange_shape_function(shape, deg)
        end

        fs = FunctionSpace(:Lagrange, 0)
        λ = x -> shape_functions(fs, shape, x)
        ∇λ = x -> ∂λξ_∂ξ(fs, shape, x)
        @test λ([π, ℯ])[1] ≈ 1.0
        @test ∇λ([π, ℯ])[1, :] ≈ [0.0, 0.0]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [0.0, 0.0])
        @test λ == [1.0]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape)
        @test λ[1]([0.0, 0.0]) == 1.0

        fs = FunctionSpace(:Lagrange, 1)
        λ = ξ -> shape_functions(fs, shape, ξ)
        ∇λ = ξ -> ∂λξ_∂ξ(fs, shape, ξ)
        @test λ([-1, -1])[1] ≈ 1.0
        @test λ([1, -1])[2] ≈ 1.0
        @test λ([1, 1])[4] ≈ 1.0
        @test λ([-1, 1])[3] ≈ 1.0
        @test Σ(λ([π, ℯ])) ≈ 1.0
        @test ∇λ([0, 0])[1, :] ≈ [-1.0, -1.0] ./ 4.0
        @test ∇λ([0, 0])[2, :] ≈ [1.0, -1.0] ./ 4.0
        @test ∇λ([0, 0])[4, :] ≈ [1.0, 1.0] ./ 4.0
        @test ∇λ([0, 0])[3, :] ≈ [-1.0, 1.0] ./ 4.0
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [0.0, 0.0])
        @test λ == [0.25, 0.25, 0.25, 0.25]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape)
        @test λ[1]([0.0, 0.0]) == 0.25
        @test λ[2]([0.0, 0.0]) == 0.25
        @test λ[3]([0.0, 0.0]) == 0.25
        @test λ[4]([0.0, 0.0]) == 0.25
        λ = Bcube.shape_functions_vec(fs, Val(2), shape)
        @test λ[1]([0.0, 0.0]) == [0.25, 0.0]
        @test λ[2]([0.0, 0.0]) == [0.25, 0.0]
        @test λ[3]([0.0, 0.0]) == [0.25, 0.0]
        @test λ[4]([0.0, 0.0]) == [0.25, 0.0]
        @test λ[5]([0.0, 0.0]) == [0.0, 0.25]
        @test λ[6]([0.0, 0.0]) == [0.0, 0.25]
        @test λ[7]([0.0, 0.0]) == [0.0, 0.25]
        @test λ[8]([0.0, 0.0]) == [0.0, 0.25]

        fs = FunctionSpace(:Lagrange, 2)
        λ = ξ -> shape_functions(fs, shape, ξ)
        @test λ([-1, -1])[1] ≈ 1.0
        @test λ([1, -1])[3] ≈ 1.0
        @test λ([1, 1])[9] ≈ 1.0
        @test λ([-1, 1])[7] ≈ 1.0
        @test Σ(λ([π, ℯ])) ≈ 1.0
        λ = Bcube.shape_functions_vec(fs, Val(1), shape, [0.0, 0.0])
        @test λ == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        λ = Bcube.shape_functions_vec(fs, Val(1), shape)
        @test λ[1]([0.0, 0.0]) == 0.0
        @test λ[2]([0.0, 0.0]) == 0.0
        @test λ[3]([0.0, 0.0]) == 0.0
        @test λ[4]([0.0, 0.0]) == 0.0
        @test λ[5]([0.0, 0.0]) == 1.0
        @test λ[6]([0.0, 0.0]) == 0.0
        @test λ[7]([0.0, 0.0]) == 0.0
        @test λ[8]([0.0, 0.0]) == 0.0
        @test λ[9]([0.0, 0.0]) == 0.0
    end

    @testset "Cube" begin
        for deg in 0:2
            test_lagrange_shape_function(Cube(), deg)
        end
    end

    @testset "Tetra" begin
        for deg in 0:1
            test_lagrange_shape_function(Tetra(), deg)
        end
    end

    @testset "Prism" begin
        for deg in 0:1
            test_lagrange_shape_function(Prism(), deg)
        end
    end
end
