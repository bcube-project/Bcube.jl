@testset "Shape function generator" begin
    @testset "Line" begin
        shape = Line()

        fs = FunctionSpace(:LagrangeGenerated, 1)
        λ = ξ -> shape_functions(fs, shape, ξ)
        ∇λ = ξ -> ∂λξ_∂ξ(fs, shape, ξ)
        @test λ([-1.0]) ≈ [1.0, 0.0]
        @test λ([1.0]) ≈ [0.0, 1.0]
        @test ∇λ([0.0]) ≈ [-0.5, 0.5]

        fs = FunctionSpace(:HermiteGenerated, 3)
        λ = ξ -> shape_functions(fs, shape, ξ)
        ∇λ = ξ -> ∂λξ_∂ξ(fs, shape, ξ)
        @test λ([-1.0]) ≈ [1.0, 0.0, 0.0, 0.0]
        @test λ([1.0]) ≈ [0.0, 0.0, 1.0, 0.0]
        @test ∇λ([-1.0]) ≈ [0.0, 1.0, 0.0, 0.0]
        @test ∇λ([1.0]) ≈ [0.0, 0.0, 0.0, 1.0]

        a(u, v) = ∫(∇(u) ⋅ ∇(v))dΩ

        mesh = one_cell_mesh(:line)
        fs = FunctionSpace(:LagrangeGenerated, 1)
        U = TrialFESpace(fs, mesh)
        V = TestFESpace(U)
        dΩ = Measure(CellDomain(mesh), 3)
        A = assemble_bilinear(a, U, V)
        @test A ≈ [
            0.5 -0.5
            -0.5 0.5
        ]
    end

    @testset "Triangle" begin
        shape = Triangle()

        fs = FunctionSpace(:LagrangeGenerated, 1)
        λ = ξ -> shape_functions(fs, shape, ξ)
        ∇λ = ξ -> ∂λξ_∂ξ(fs, shape, ξ)
        @test λ([0.0, 0.0]) ≈ [1.0, 0.0, 0.0]
        @test λ([0.0, 1.0]) ≈ [0.0, 1.0, 0.0]
        @test λ([1.0, 0.0]) ≈ [0.0, 0.0, 1.0]

        @test ∇λ([π, ℯ]) ≈ [
            -1.0 -1.0
            0.0 1.0
            1.0 0.0
        ]

        a(u, v) = ∫(∇(u) ⋅ ∇(v))dΩ

        mesh = one_cell_mesh(
            :tri;
            xmin = 0.0,
            xmax = 1.0,
            ymin = 0.0,
            ymax = 1.0,
            zmin = -1.0,
            zmax = 1.0,
        )
        fs = FunctionSpace(:LagrangeGenerated, 1)
        U = TrialFESpace(fs, mesh)
        V = TestFESpace(U)
        dΩ = Measure(CellDomain(mesh), 3)
        A = assemble_bilinear(a, U, V)
        @test A ≈ [
            1.0 -0.5 -0.5
            -0.5 0.5 0.0
            -0.5 0.0 0.5
        ]
    end

    # TO DO
    # @testset "Square" begin
    #     shape = Square()

    # end
end
