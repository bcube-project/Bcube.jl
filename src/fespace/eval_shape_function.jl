"""
    shape_functions(fs::AbstractFunctionSpace, n::Val{N}, quadrule::AbstractQuadratureRule) where {N}
    shape_functions(fs::AbstractFunctionSpace, n::Val{N}, quadnode::QuadratureNode) where {N}

Return values of shape functions corresponding to a function space `fs` evaluated
at a given `quadnode` position or at all quadrature nodes of a given `quadrule`.
"""
function _shape_functions_on_quadrule_gen(
    fs::AbstractFunctionSpace,
    n::Val{N},
    quadrule::AbstractQuadratureRule,
) where {N}
    shape = get_shape(quadrule)()
    λ = [shape_functions(fs, n, shape, ξ) for ξ in get_nodes(quadrule)]
    return :(SA[$(λ...)])
end

@generated function _shape_functions_on_quadrule(
    fs::AbstractFunctionSpace,
    ::Val{N},
    ::AbstractQuadratureRule{S, Q},
) where {N, S, Q}
    _quadrule = QuadratureRule(S(), Q())
    _shape_functions_on_quadrule_gen(fs(), Val(N), _quadrule)
end

function shape_functions(
    fs::AbstractFunctionSpace,
    n::Val{N},
    quadnode::QuadratureNode,
) where {N}
    quadrule = get_quadrature_rule(quadnode)
    _shape_functions_on_quadrule(fs, n, quadrule)[get_index(quadnode)]
end

function shape_functions(
    fs::AbstractFunctionSpace,
    n::Val{N},
    quadrule::AbstractQuadratureRule,
) where {N}
    _shape_functions_on_quadrule(fs, n, quadrule)
end

# Alias for scalar case
function shape_functions(fs::AbstractFunctionSpace, quadrule::AbstractQuadratureRule)
    shape_functions(fs, Val(1), quadrule)
end

"""
    ∂λξ_∂ξ(::AbstractFunctionSpace, n::Val{N}, quadnode::QuadratureNode) where N

Gradient of shape functions for any function space. The result is an array whose values
are the gradient of each shape functions evaluated at `quadnode` position.
`N` is the size of the finite element space.

# Implementation

Default version using automatic differentiation. Specialize to increase performance.
"""
function ∂λξ_∂ξ(fs::AbstractFunctionSpace, n::Val{N}, quadnode::QuadratureNode) where {N}
    quadrule = get_quadrature_rule(quadnode)
    ∂λξ_∂ξ(fs, n, quadrule)[get_index(quadnode)]
end

function _∂λξ_∂ξ_gen(
    fs::AbstractFunctionSpace,
    ::Val{N},
    quadrule::AbstractQuadratureRule,
) where {N}
    shape = get_shape(quadrule)()
    ξ = get_nodes(quadrule)
    ∇λ = map(_ξ -> ∂λξ_∂ξ(fs, Val(N), shape, _ξ), ξ)
    if isa(∇λ[1], SArray)
        expr_∇λ = [:($(typeof(_∇λ))($(_∇λ...))) for _∇λ in ∇λ]
    else
        expr_∇λ = [:($_∇λ) for _∇λ in ∇λ]
    end
    return :(SA[$(expr_∇λ...)])
end

@generated function _∂λξ_∂ξ(
    fs::AbstractFunctionSpace,
    ::Val{N},
    ::AbstractQuadratureRule{S, Q},
) where {N, S, Q}
    _quadrule = QuadratureRule(S(), Q())
    _∂λξ_∂ξ_gen(fs(), Val(N), _quadrule)
end

function ∂λξ_∂ξ(
    fs::AbstractFunctionSpace,
    n::Val{N},
    quadrule::AbstractQuadratureRule{S, Q},
) where {N, S, Q}
    _∂λξ_∂ξ(fs, n, quadrule)
end

# fix ambiguity
function ∂λξ_∂ξ(
    fs::AbstractFunctionSpace,
    n::Val{1},
    quadrule::AbstractQuadratureRule{S, Q},
) where {S, Q}
    _∂λξ_∂ξ(fs, n, quadrule)
end

# alias for scalar case
function ∂λξ_∂ξ(
    fs::AbstractFunctionSpace,
    quadrule::AbstractQuadratureRule{S, Q},
) where {S, Q}
    ∂λξ_∂ξ(fs, Val(1), quadrule)
end
