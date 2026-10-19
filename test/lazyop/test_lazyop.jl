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
        # ... and is inferred, even when the components of the `Tuple` have different types
        h2(x) = (x, 2.0 * x, x * NullOperator())
        @test @inferred(map_over_leaves(h2, a)) === (
            MapOver(MapOver(1, 2), MapOver(3, 4)),
            MapOver(MapOver(2.0, 4.0), MapOver(6.0, 8.0)),
            NullOperator(),
        )

        # the other subtypes of `AbstractMapOver` are evaluated operator by operator
        struct OtherMapOver{A <: Tuple} <: Bcube.LazyOperators.AbstractMapOver{A}
            args::A
        end
        Bcube.LazyOperators.get_basetype(::Type{<:OtherMapOver}) = OtherMapOver
        o = OtherMapOver((3, 4))
        @test map_over_leaves(f, o, 2) === OtherMapOver((7, 9))
        @test map_over_leaves(f, o, MapOver(1, 2)) === OtherMapOver((4, 9))

        # above `MAX_UNROLLED_LEAVES` leaves, `f` is evaluated operator by operator
        which_path(x, _) = x isa MapOver ? :operator_by_operator : :leaves
        @test map_over_leaves(which_path, MapOver(1, 2), 2) === MapOver(:leaves, :leaves)
        n = isqrt(Bcube.LazyOperators.MAX_UNROLLED_LEAVES) + 1
        big = MapOver(ntuple(i -> MapOver(ntuple(j -> i + j, n)), n))
        @test map_over_leaves(which_path, big, 2) === :operator_by_operator
        @test map_over_leaves(f, big, 2) == f(big, 2)

        # The generator of `_map_over_leaves` must find its helpers when `LazyOperators` is
        # loaded from source: the pkgimage of Bcube hides a wrong definition order
        # ("method too new", or an undefined binding for julia >= 1.12). It runs in a new
        # process: loading `LazyOperators` again here would overwrite the methods it defines
        # on functions of `Base`. The new process has no project: `LazyOperators` only
        # depends on standard libraries.
        path = joinpath(pkgdir(Bcube), "src", "LazyOperators", "LazyOperators.jl")
        code = """
        include($(repr(path)))
        r = LazyOperators.map_over_leaves(*, LazyOperators.MapOver(1, 2), 2)
        r === LazyOperators.MapOver(2, 4) || exit(1)
        """
        @test success(`$(Base.julia_cmd()) --startup-file=no -e $code`)
    end

    @testset "DofMap" begin
        import Bcube.LazyOperators:
            DofMap, BlockMap, NullOperator, dof_copy, dofwise, unwrap
        ∅ = NullOperator()

        a = DofMap((1.0, 2.0, 3.0))
        @test eltype(a) === Float64
        @test unwrap(a) === (1.0, 2.0, 3.0)
        @test_throws DimensionMismatch DofMap{(2, 2)}((1, 2, 3))
        @test_throws ArgumentError DofMap{3}((1, 2, 3))
        @test_throws ArgumentError DofMap{(-1, -3)}((1, 2, 3))
        @test_throws ArgumentError DofMap{(1, 1, 3)}((1, 2, 3))

        # the operators are lazy and fused: a single `copy` at the end
        @test (2a + a) isa Base.Broadcast.Broadcasted
        @test dof_copy(2a + a) === DofMap((3.0, 6.0, 9.0))
        @test copy(-a) === DofMap((-1.0, -2.0, -3.0))
        @test Base.materialize(a * a) === DofMap((1.0, 4.0, 9.0))

        # trial (1, n) ⊕ test (m, 1) gives (m, n), stored column-major
        u = DofMap{(1, 3)}((1, 2, 3))
        v = DofMap{(2, 1)}((10, 20))
        @test dof_copy(u * v) === DofMap{(2, 3)}((10, 20, 20, 40, 30, 60))
        @test_throws DimensionMismatch dof_copy(u * DofMap{(1, 2)}((1, 2)))

        # the other arguments are passed unchanged to each dof
        g = DofMap((SA[1.0, 2.0], SA[3.0, 4.0]))
        @test dof_copy(g ⋅ SA[1.0, 1.0]) === DofMap((3.0, 7.0))
        @test dof_copy(g ⊗ SA[1.0, 0.0]) ===
              DofMap((SA[1.0 0.0; 2.0 0.0], SA[3.0 0.0; 4.0 0.0]))

        # `NullOperator`
        @test a * ∅ === ∅
        @test dot(∅, a) === ∅
        @test g ⊗ ∅ === ∅
        @test a + ∅ === a
        @test dof_copy(∅ - a) === DofMap((-1.0, -2.0, -3.0))
        @test max(a, ∅) === a
        @test dof_copy(dofwise((x, y) -> y, a, ∅)) === ∅ # values of type `NullOperator`

        # `BlockMap`
        b = BlockMap(a, ∅)
        @test dof_copy(b * 2) === BlockMap(DofMap((2.0, 4.0, 6.0)), ∅)
        @test dof_copy(b + BlockMap(a, a)) === BlockMap(DofMap((2.0, 4.0, 6.0)), a)
        @test b * ∅ === ∅
        @test BlockMap(∅, ∅) * 2 === ∅ # collapsed
        @test dof_copy(BlockMap(dofwise((x, y) -> y, a, ∅), ∅)) === ∅ # collapsed after copy
        @test_throws DimensionMismatch b * BlockMap(a, a, a)

        # composition
        f(x, y) = ifelse(x > y, -x, 10x)
        @test dof_copy(dofwise(f, a, 1.5)) === DofMap((10.0, -2.0, -3.0))
        @test dof_copy(dofwise((x, y) -> x * y + x, b, 1.5)) ===
              BlockMap(DofMap((2.5, 5.0, 7.5)), ∅)
        @test dofwise(f, 1, 2) === 10
        @test_throws DimensionMismatch dofwise(+, b, BlockMap(a, a, a))

        # a function returning a `Tuple` gives a `Tuple` of `DofMap`s
        t(x) = (x, 2x, x * ∅)
        @test dof_copy(dofwise(t, a)) === (a, DofMap((2.0, 4.0, 6.0)), ∅)
        @test @inferred(dof_copy(dofwise(t, b))) ===
              (BlockMap(a, ∅), BlockMap(DofMap((2.0, 4.0, 6.0)), ∅), ∅)

        # values of a non-concrete type (failed inference): unrolled fallback
        @test unwrap(dof_copy(dofwise(x -> x > 1.5 ? 1 : 2.0, a))) === (2.0, 1, 1)

        # 81 × 81 dofs: inferred, no allocation (loop, not unrolled)
        n = 81
        U = DofMap{(1, n)}(ntuple(i -> SA[i 1.0; 2.0 i], n))
        V = DofMap{(n, 1)}(ntuple(i -> SA[1.0 i; i 0.0], n))
        h(U, V) = dof_copy((U ⊡ V) * 2.0 - tr(U) * tr(V) + dofwise(f, tr(U), 1.0))
        r = @inferred h(U, V)
        @test r isa DofMap{(n, n), Float64}
        @test unwrap(r)[2 + (3 - 1) * n] ≈
              (SA[3 1.0; 2.0 3] ⊡ SA[1.0 2; 2 0.0]) * 2 - 6 * 1 - 6
        h_alloc(U, V) = @allocated h(U, V)
        h_alloc(U, V)
        @test h_alloc(U, V) == 0

        # same for a linear form (1D axes) with blocks
        W = BlockMap(DofMap(ntuple(i -> SA[i, 1.0], n)), ∅)
        k(W) = dof_copy(W ⋅ SA[1.0, 2.0] * 3.0 + dofwise((w, x) -> w ⋅ x, W, SA[0.0, 1.0]))
        r = @inferred k(W)
        @test unwrap(r)[1] isa DofMap{(n,), Float64}
        @test unwrap(unwrap(r)[1])[4] ≈ (4 + 2) * 3 + 1
        k_alloc(W) = @allocated k(W)
        k_alloc(W)
        @test k_alloc(W) == 0

        # no method ambiguity with the types of Bcube (`MapOver` will be removed, and
        # `max`/`min` with `Missing` are not relevant)
        amb = Test.detect_ambiguities(Bcube.LazyOperators, Bcube)
        is_new(m) = occursin(r"DofMap|BlockMap|FixedArg", string(m.sig))
        is_skipped(m) =
            occursin(r"MapOver|Missing", replace(string(m.sig), "LazyMapOver" => ""))
        @test isempty(filter(p -> any(is_new, p) && !any(is_skipped, p), amb))

        # The generator of `dofwise` must find its helpers when `LazyOperators` is loaded
        # from source (see the `map_over_leaves` testset)
        path = joinpath(pkgdir(Bcube), "src", "LazyOperators", "LazyOperators.jl")
        code = """
        include($(repr(path)))
        L = LazyOperators
        r = L.dof_copy(L.dofwise(*, L.BlockMap(L.DofMap((1, 2)), L.NullOperator()), 2))
        r === L.BlockMap(L.DofMap((2, 4)), L.NullOperator()) || exit(1)
        """
        @test success(`$(Base.julia_cmd()) --startup-file=no -e $code`)
    end
end
