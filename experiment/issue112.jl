module issue112
using Bcube

mesh = one_cell_mesh(:line)
dΩ = Measure(CellDomain(mesh), 1)
U = TrialFESpace(FunctionSpace(:Lagrange, 0), mesh)
V = TestFESpace(U)

struct Parameters
    x::Int
end
params = Parameters(1)

# OK
f1(v) = _f1 ∘ (v,)
_f1(v) = v
l1(v) = ∫(f1(v))dΩ
a1 = assemble_linear(l1, V)

# KO because attempting to materialize params::Parameters on CellInfo
f2(v, params) = _f2 ∘ (v, params)
_f2(v, params) = v
@show "cas 2===="
l2(v) = ∫(f2(v, params))dΩ
a2 = assemble_linear(l2, V)

# KO first arg params is not AbstractLazy
function f3(params, v)
    a = lazy_compose(_f3, (params, v))
    Bcube.show_lazy_operator(a)
    return a
end
@show "cas 3===="
_f3(params, v) = v
l3(v) = ∫(f3(params, v))dΩ
a3 = assemble_linear(l3, V)

@show a1 == a2 == a3

end
