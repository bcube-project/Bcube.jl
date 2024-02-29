import Bcube: Connectivity, minsize, maxsize, inverse_connectivity

@testset "connectivity" begin
    ne = 4
    numIndices = [2, 4, 3, 1]
    indices = [10, 9, 7, 4, 6, 9, 1, 3, 5, 12]
    c = Connectivity(numIndices, indices)

    @test length(c) === 4
    @test size(c) === (4,)
    @test length(c, 2) === 4
    @test size(c, 4) === (1,)
    @test axes(c) === 1:4
    @test axes(c, 1) === 1:1:2
    @test axes(c, 2) === 3:1:6
    @test c[1] == [10, 9]
    @test c[2] == [7, 4, 6, 9]
    @test c[3] == [1, 3, 5]
    @test c[4] == [12]
    @test c[-2] == reverse(c[2])
    @test c[2][3] === 6
    @test minsize(c) === 1
    @test maxsize(c) === 4
    for (i, a) in enumerate(c)
        @test a == c[i]
    end

    invc, _keys = inverse_connectivity(c)
    _test = Bool[]
    for (i, a) in enumerate(invc)
        for b in a
            push!(_test, _keys[i] ∈ c[b])
        end
    end
    @assert length(_test) > 0 && all(_test)
end
