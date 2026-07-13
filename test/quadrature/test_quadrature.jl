@testset "Quadrature" begin
    @testset "Quadrature type and degree" begin
        q = Quadrature(2)
        @test q isa Quadrature
        @test Bcube.get_degree(q) == 2
        @test Bcube.get_quadtype(q) isa QuadratureLegendre

        q_lob = Quadrature(QuadratureLobatto(), 3)
        @test Bcube.get_degree(q_lob) == 3
        @test Bcube.get_quadtype(q_lob) isa QuadratureLobatto

        q_uni = Quadrature(QuadratureUniform(), 1)
        @test Bcube.get_degree(q_uni) == 1
        @test Bcube.get_quadtype(q_uni) isa QuadratureUniform

        q_val = Quadrature(Val(4))
        @test Bcube.get_degree(q_val) == 4

        q0 = Quadrature(0)
        @test Bcube.get_degree(q0) == 0
    end

    @testset "QuadratureRule - Line" begin
        qr = QuadratureRule(Line(), Quadrature(2))
        @test length(qr) == 2
        @test sum(Bcube.get_weights(qr)) ≈ 2.0

        qr1 = QuadratureRule(Line(), Quadrature(1))
        @test length(qr1) == 1
        @test sum(Bcube.get_weights(qr1)) ≈ 2.0

        qr_lob = QuadratureRule(Line(), Quadrature(QuadratureLobatto(), 2))
        @test length(qr_lob) == 3
        @test sum(Bcube.get_weights(qr_lob)) ≈ 2.0

        qr_uni = QuadratureRule(Line(), Quadrature(QuadratureUniform(), 2))
        @test length(qr_uni) == 3
        @test sum(Bcube.get_weights(qr_uni)) ≈ 1.0

        qr_int = QuadratureRule(Line(), 3)
        @test length(qr_int) == 2
    end

    @testset "QuadratureRule - Square" begin
        qr = QuadratureRule(Square(), Quadrature(2))
        @test length(qr) == 4
        @test sum(Bcube.get_weights(qr)) ≈ 4.0
        @test all(length.(Bcube.get_nodes(qr)) .== 2)
    end

    @testset "QuadratureRule - Cube" begin
        qr = QuadratureRule(Cube(), Quadrature(2))
        @test length(qr) == 8
        @test sum(Bcube.get_weights(qr)) ≈ 8.0
        @test all(length.(Bcube.get_nodes(qr)) .== 3)
    end

    @testset "QuadratureRule - Triangle" begin
        for d in 1:8
            qr = QuadratureRule(Triangle(), Quadrature(d))
            @test sum(Bcube.get_weights(qr)) ≈ 0.5
        end
    end

    @testset "QuadratureRule - Tetra" begin
        for d in 1:4
            qr = QuadratureRule(Tetra(), Quadrature(d))
            @test sum(Bcube.get_weights(qr)) ≈ 1.0 / 6.0
        end
    end

    @testset "QuadratureRule - Prism" begin
        for d in 1:5
            qr = QuadratureRule(Prism(), Quadrature(d))
            @test sum(Bcube.get_weights(qr)) ≈ 1.0
        end
    end

    @testset "QuadratureRule - Pyramid" begin
        for d in 1:5
            qr = QuadratureRule(Pyramid(), Quadrature(d))
            @test length(Bcube.get_weights(qr)) == length(Bcube.get_nodes(qr))
        end
    end

    @testset "QuadratureRule - Point" begin
        qr = QuadratureRule(Bcube.Point(), Quadrature(1))
        @test length(qr) == 1
        @test Bcube.get_weights(qr)[1] ≈ 1.0
        @test Bcube.get_nodes(qr)[1] ≈ 0.0
    end

    @testset "quadrature_rule_bary - Triangle" begin
        w, bary = Bcube.quadrature_rule_bary(1, Triangle(), Val(1))
        @test length(w) == 1
        w2, bary2 = Bcube.quadrature_rule_bary(1, Triangle(), Val(2))
        @test length(w2) == 2
        w3, bary3 = Bcube.quadrature_rule_bary(1, Triangle(), Val(3))
        @test length(w3) == 3
    end

    @testset "quadrature_rule_bary - Square" begin
        for d in 1:3
            w, bary = Bcube.quadrature_rule_bary(1, Square(), Val(d))
            @test length(w) == d
        end
    end

    @testset "quadrature_rule_bary - Tetra" begin
        w1, bary1 = Bcube.quadrature_rule_bary(1, Tetra(), Val(1))
        @test length(w1) == 1
        w2, bary2 = Bcube.quadrature_rule_bary(1, Tetra(), Val(2))
        @test length(w2) == 3
    end

    @testset "QuadratureNode" begin
        qr = QuadratureRule(Line(), Quadrature(2))
        quadnodes = Bcube.get_quadnodes(qr)
        @test length(quadnodes) == length(qr)

        qn = quadnodes[1]
        @test Bcube.get_index(qn) == 1
        @test Bcube.get_coords(qn) isa StaticArrays.SVector
        @test qn[1] == Bcube.get_coords(qn)[1]
        @test Bcube.get_quadrature_rule(qn) isa QuadratureRule
        @test Bcube.evalquadnode(identity, qn) == Bcube.get_coords(qn)
    end

    @testset "is_collocated" begin
        q1 = Quadrature(2)
        q2 = Quadrature(2)
        q3 = Quadrature(3)
        @test Bcube.is_collocated(q1, q2) isa Bcube.IsCollocatedStyle
        @test Bcube.is_collocated(q1, q3) isa Bcube.IsNotCollocatedStyle
    end

    @testset "get_num_nodes_per_dim" begin
        qr_line = QuadratureRule(Line(), Quadrature(2))
        @test Bcube._get_num_nodes_per_dim(qr_line) == length(qr_line)

        qr_sq = QuadratureRule(Square(), Quadrature(2))
        @test Bcube._get_num_nodes_per_dim(qr_sq) == (2, 2)

        qr_cube = QuadratureRule(Cube(), Quadrature(2))
        @test Bcube._get_num_nodes_per_dim(qr_cube) == (2, 2, 2)
    end

    @testset "Lobatto triangle quadrature" begin
        w, x = Bcube.quadrature_points(Triangle(), Val(4), QuadratureLobatto())
        @test length(w) == 22
        @test sum(w) ≈ 0.5
    end

    @testset "get_coords on Number and Array" begin
        @test Bcube.get_coords(3.14) == 3.14
        @test Bcube.get_coords([1.0, 2.0]) == [1.0, 2.0]
    end

    @testset "QuadratureRule with NTuple" begin
        w = (1.0, 1.0)
        x = (SA[-1.0], SA[1.0])
        qr = QuadratureRule(Line(), Quadrature(1), w, x)
        @test length(qr) == 2
        @test Bcube.get_weights(qr) isa StaticArrays.SVector
    end

    @testset "Line Lobatto exactness" begin
        qr = QuadratureRule(Line(), Quadrature(QuadratureLobatto(), 3))
        w = Bcube.get_weights(qr)
        x = Bcube.get_nodes(qr)
        @test sum(w .* x .^ 2) ≈ 2.0 / 3.0
    end

    @testset "QuadratureNode size" begin
        qr = QuadratureRule(Line(), Quadrature(2))
        qn = Bcube.get_quadnodes(qr)[1]
        @test size(qn) == size(Bcube.get_coords(qn))
    end
end