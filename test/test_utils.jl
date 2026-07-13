import Bcube: densify, densify!, rawcat
import Bcube:
    soft_max,
    soft_min,
    soft_abs,
    soft_extrema,
    WiddenAsUnion,
    unwrap,
    map_and_widden_as_union,
    myfindfirst,
    tuplemap,
    cumsum_exclusive,
    convert_to_vector_of_union

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

    @testset "soft functions" begin
        @test soft_max(1.0, 2.0) == 2.0
        @test soft_max(-3.0, 5.0) == 5.0
        @test soft_min(1.0, 2.0) == 1.0
        @test soft_min(-3.0, 5.0) == -3.0
        @test soft_abs(3.0) == 3.0
        @test soft_abs(-3.0) == 3.0
        @test soft_abs(0.0) == 0.0
        @test soft_extrema([1, 3, 2]) == (1, 3)
        @test soft_extrema([5]) == (5, 5)
        @test_throws ArgumentError soft_extrema(Int[])
        @test soft_extrema(abs, [-2, 3, -1]) == (1, 3)
    end

    @testset "WiddenAsUnion" begin
        a = WiddenAsUnion(1)
        @test unwrap(a) == 1
        @test unwrap(WiddenAsUnion{Int}) == Int
        @test unwrap(Union{WiddenAsUnion{Int}, Float64}) == Union{Int, Float64}
    end

    @testset "map_and_widden_as_union" begin
        result = map_and_widden_as_union(x -> x * 2, [1, 2, 3])
        @test result == [2, 4, 6]
        @test eltype(result) == Int

        result = map_and_widden_as_union(x -> x * 2, [1, 2, [3, 4]])
        @test result[1] == 2
        @test result[2] == 4
        @test result[3] == [6, 8]
    end

    @testset "myfindfirst" begin
        @test myfindfirst(x -> x > 2, (1, 2, 3, 4)) == 3
        @test myfindfirst(x -> x > 10, (1, 2, 3)) === nothing
        @test myfindfirst(x -> x == 1, (1,)) == 1
    end

    @testset "tuplemap" begin
        @test tuplemap(+, (1, 2, 3), (4, 5, 6)) == (5, 7, 9)
        @test tuplemap(*, (1, 2), (3, 4)) == (3, 8)
        @test tuplemap(x -> x * 2, (1, 2, 3)) == (2, 4, 6)
        @test tuplemap(+, (), ()) == ()
    end

    @testset "cumsum_exclusive" begin
        @test cumsum_exclusive([1, 2, 3, 4]) == [0, 1, 3, 6]
        @test cumsum_exclusive([5]) == [0]
        @test cumsum_exclusive((1, 2, 3)) == (0, 1, 3)
        @test cumsum_exclusive(()) == ()
    end

    @testset "convert_to_vector_of_union" begin
        a = Any[1, 2.0, "hello"]
        result = convert_to_vector_of_union(a)
        @test result == a
        @test eltype(result) != Any  # should be a Union type

        b = [1, 2, 3]
        result2 = convert_to_vector_of_union(b)
        @test result2 === b
    end
end
