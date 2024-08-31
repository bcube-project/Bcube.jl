@testset "issue #130" begin
    struct Parameters
        x::Int
    end

    function _run()
        mesh = one_cell_mesh(:line)
        dΩ = Measure(CellDomain(mesh), 1)
        U = TrialFESpace(FunctionSpace(:Lagrange, 0), mesh)
        V = TestFESpace(U)

        params = Parameters(1)

        f1(v) = _f1 ∘ (v,)
        _f1(v) = v
        l1(v) = ∫(f1(v))dΩ
        a1 = assemble_linear(l1, V)

        # Two tests of composition with not-only-`AbstractLazy` args :

        # As the first tuple arg `v` is an `AbstractLazy`,
        # operator `∘` automatically builds a `LazyOp`
        f2(v, params) = _f2 ∘ (v, params)
        _f2(v, params) = v
        l2(v) = ∫(f2(v, params))dΩ
        a2 = assemble_linear(l2, V)

        # As the first tuple arg `params` is not an `AbstractLazy`,
        # one must explicitly use `lazy_compose` to build a `LazyOp`
        f3(params, v) = lazy_compose(_f3, (params, v))
        _f3(params, v) = v
        l3(v) = ∫(f3(params, v))dΩ
        a3 = assemble_linear(l3, V)

        @test a1 == a2 == a3
    end

    _run()
end
