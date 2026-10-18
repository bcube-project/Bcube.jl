@testset "LazyOperators" begin
    args = ((1, 2, 3), (3, 4, 4))
    @test Bcube.LazyOperators._map_over(+, args...) == (4, 6, 7)

    args = (("a", "b", 5, "d"), ("e", "f", 4, "h"))
    @test Bcube.LazyOperators._map_over(*, args...) == ("ae", "bf", 20, "dh")

    @testset "LazyWrap" begin
        import Bcube.LazyOperators: LazyWrap, get_args, materialize, unwrap

        lw = LazyWrap(1, 2, 3)
        @test get_args(lw) === (1, 2, 3)
        @test unwrap(lw) === (1, 2, 3)
        @test materialize(lw) === (1, 2, 3)

        lw1 = LazyWrap(42)
        @test materialize(lw1) === 42

        @test materialize(LazyWrap(1, 2), "extra") === (1, 2)
    end

    @testset "LazyOperator" begin
        import Bcube.LazyOperators:
            LazyOperator, get_args, get_operator, materialize, get_type_operator

        op = LazyOperator(+, 1, 2)
        @test get_operator(op) === +
        @test get_args(op) === (1, 2)
        @test materialize(op) == 3

        op2 = LazyOperator(*, LazyOperator(+, 1, 2), 3)
        @test materialize(op2) == 9

        f = (x, y) -> x - y
        op3 = LazyOperator(f, 10, 3)
        @test materialize(op3) == 7

        # Calling the operator materializes with extra args
        @test op(5, 6) == 3

        @test get_type_operator(typeof(op)) == typeof(+)
    end

    @testset "NullOperator" begin
        import Bcube.LazyOperators: NullOperator, materialize, get_operator, get_args

        n = NullOperator()
        @test get_operator(n) === nothing
        @test get_args(n) === (nothing,)
        @test materialize(n, 1.0) === n

        @test map(x -> 42, n) == 42
    end

    @testset "pretty_name" begin
        import Bcube.LazyOperators: pretty_name

        @test pretty_name(42) == "42"
        @test pretty_name(3.14) == "3.14"
        @test pretty_name(nothing) == ""
        @test pretty_name([1, 2]) == "Vector{Int64}"
    end

    @testset "may_unwrap_tuple" begin
        import Bcube.LazyOperators: may_unwrap_tuple

        @test may_unwrap_tuple((1, 2, 3)) === (1, 2, 3)
        @test may_unwrap_tuple((42,)) === 42
    end

    @testset "materialize on tuples" begin
        import Bcube.LazyOperators: materialize, LazyWrap

        result = materialize((1, 2, 3))
        @test result isa LazyWrap
        @test result.args === (1, 2, 3)
    end

    @testset "lazy_compose materialize_op" begin
        import Bcube.LazyOperators: LazyOperator, materialize, lazy_compose, MapOver

        op = LazyOperator(lazy_compose, (x -> x + 1), 5)
        @test materialize(op) == 6

        # `MapOver` arguments : the function is applied on each leaf
        # (`>` is not defined on `MapOver`)
        f(x, y) = ifelse(x > 1, x * y, -1)
        op = lazy_compose(f, (MapOver(1, 2, 3), 10))
        @test materialize(op) === MapOver(-1, 20, 30)
    end

    @testset "map_over_leaves" begin
        import Bcube.LazyOperators: map_over_leaves, MapOver, NullOperator

        f(x, y) = x * y + 1
        @test map_over_leaves(f, 2, 3) === 7
        @test map_over_leaves(f, MapOver(1, 2, 3), 2) === MapOver(3, 5, 7)
        @test map_over_leaves(f, MapOver(1, 2), MapOver(3, 4)) === MapOver(4, 9)

        # nested `MapOver`s
        a = MapOver(MapOver(1, 2), MapOver(3, 4))
        @test map_over_leaves(f, a, 2) === MapOver(MapOver(3, 5), MapOver(7, 9))
        # the nested `MapOver`s must not prevent the inference (it did with julia 1.10)
        f3(x, y, z) = x * y + z
        @test @inferred(map_over_leaves(f3, a, 2, 1)) ===
              MapOver(MapOver(3, 5), MapOver(7, 9))

        # leaves that are all `NullOperator` are collapsed
        g(x, y) = x * y
        @test map_over_leaves(g, MapOver(1, 2), NullOperator()) === NullOperator()
        b = MapOver(MapOver(1, 2), NullOperator())
        @test map_over_leaves(g, b, 3) === MapOver(MapOver(3, 6), NullOperator())

        # a function returning a `Tuple` gives a `Tuple` of `MapOver`s
        h(x) = (x, 2 * x, x * NullOperator())
        @test map_over_leaves(h, MapOver(1, 2)) ===
              (MapOver(1, 2), MapOver(2, 4), NullOperator())
        @test map_over_leaves(h, a) === (
            MapOver(MapOver(1, 2), MapOver(3, 4)),
            MapOver(MapOver(2, 4), MapOver(6, 8)),
            NullOperator(),
        )
    end
end
