@testset "LazyOperators" begin
    args = ((1, 2, 3), (3, 4, 4))
    @test Bcube.LazyOperators._map_over(+, args...) == (4, 6, 7)

    args = (("a", "b", "c", "d"), ("e", "f", "g", "h"))
    @test Bcube.LazyOperators._map_over(*, args...) == ("ae", "bf", "cg", "dh")
end