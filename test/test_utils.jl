import Bcube: densify, densify!, rawcat

@testset "utils" begin
    @testset "densify" begin
        a = [i for i in 1:5]
        densify!(a)
        @test a == [i for i in 1:5]

        a = [1, 3, 4, 5, 2, 3]
        densify!(a)
        @test a == [1, 2, 3, 4, 5, 2]

        a = [1, 2, 4, 10, 6, 10, 2]
        densify!(a)
        @test a == [1, 2, 3, 4, 5, 4, 2]

        a = [1, 2, 4, 10, 6, 10, 2]
        remap = [1, 2, 0, 3, 4, 5, 0, 0, 0, 4]
        _a, _remap = densify(a; permute_back = true)
        @test all(_remap[i] == remap[i] for i in keys(_remap))
    end

    @testset "otimes" begin
        a = [1, 2]
        @test otimes(a, a) == [1 2; 2 4]
    end

    @testset "dcontract" begin
        A = zeros(2, 2, 2)
        A[1, :, :] .= [1 2; 3 4]
        A[2, :, :] .= [-1 -1; 0 0]

        B = zeros(2, 2)
        B .= [1 2; 3 4]
        @test A ⊡ B == [30.0; -3.0]
    end

    @testset "rawcat" begin
        a = [[1, 2], [3, 4, 5], [6, 7]]
        @test rawcat(a) == [1, 2, 3, 4, 5, 6, 7]

        b = [SA[1, 2], SA[3, 4, 5], SA[6, 7]]
        @test rawcat(b) == [1, 2, 3, 4, 5, 6, 7]

        c = [[1 2; 10 20], [3 4 5; 30 40 50], [6 7; 60 70]]
        @test rawcat(c) == [1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60, 7, 70]

        d = [SA[1 2; 10 20], SA[3 4 5; 30 40 50], SA[6 7; 60 70]]
        @test rawcat(d) == [1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60, 7, 70]
        @test isa(rawcat(d), Vector)

        x = [1, 2, 3]
        @test rawcat(x) == x
    end

    @testset "matrix_2_vector_of_SA" begin
        a = [
            1 2 3
            4 5 6
        ]
        b = Bcube.matrix_2_vector_of_SA(a)
        @test b[1] == SA[1, 4]
        @test b[2] == SA[2, 5]
        @test b[3] == SA[3, 6]
    end
end
