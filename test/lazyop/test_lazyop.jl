@testset "LazyOperators" begin
    args = ((1, 2, 3), (3, 4, 4))
    @test Bcube.LazyOperators._map_over(+, args...) == (4, 6, 7)

    args = (("a", "b", 5, "d"), ("e", "f", 4, "h"))
    @test Bcube.LazyOperators._map_over(*, args...) == ("ae", "bf", 20, "dh")
end
