"""
The `GradientStyle` helps distinguishing between "classic" gradient and a tangentiel gradient.

# Implementation
Note that the whole point of the `GradientStyle` is to allow two distinct symbols (∇ and ∇ₛ) for
the classic and tangential gradient. Otherwise, the dispatch between classic / tangential could
be done only in `ref2phys.jl`.
"""
abstract type AbstractGradientStyle end
struct VolumicGradientStyle <: AbstractGradientStyle end
struct TangentialGradientStyle <: AbstractGradientStyle end

"""
`GS` is the `AbstractGradientStyle`
"""
struct Gradient{O, A, GS} <: AbstractLazyOperator{O, A}
    args::A # `args` represent a Tuple containing the function whom gradient is computed
end
LazyOperators.get_args(lOp::Gradient) = lOp.args
LazyOperators.get_operator(lOp::Gradient) = lOp
Gradient(args::Tuple, gs::AbstractGradientStyle) = Gradient{Nothing, typeof(args), gs}(args)
Gradient(f, gs::AbstractGradientStyle) = Gradient((f,), gs)
Gradient(::NullOperator, ::AbstractGradientStyle) = NullOperator()
Gradient(f) = Gradient(f, VolumicGradientStyle())

@inline gradient_style(::Gradient{O, A, GS}) where {O, A, GS} = GS

const ∇ = Gradient

TangentialGradient(f) = Gradient(f, TangentialGradientStyle())
const ∇ₛ = TangentialGradient

"""
Materialization of a `Gradient` on a `CellPoint`. Only valid for a function and a `CellPoint` defined on
the reference domain.

# Implementation
The user writes mathematical expressions in the PhysicalDomain. So the gradient always represents the derivation
with respect to the physical domain spatial coordinates, even if evaluated on a point expressed in the ReferenceDomain.

The current Gradient implementation consists in applying ForwardDiff on the given operator 'u', on a point in the
ReferenceDomain. That is to say, we compute ForwarDiff.derivative(ξ -> u ∘ F, ξ) (where F is the identity if u is defined
is the ReferenceDomain). This gives ∇(u ∘ F)(ξ) = t∇(F)(ξ) * ∇(u)(F(x)).
However, we only want ∇(u)(F(x)) : that's why a multiplication by the transpose of the inverse mapping jacobian is needed.

An alternative approach would be to apply ForwardDiff in the PhysicalDomain : ForwarDiff.derivative(x -> u ∘ F^-1, x). The
problem is that the inverse mapping F^-1 is not always defined.

# Maths notes
We use the following convention for any function f:R^n->R^p : ∇f is a tensor/matrix equal to ∂fi/∂xj. However when f
is a scalar function, i.e f:R^n -> R, then ∇f is a column vector (∂f/∂x1, ∂f/∂x2, ...).

That being said:
    - if λ is a scalar function, then ∇λ = transpose(J^(-1)) ̂∇̂λ
    - if λ is a vector function, then ∇λ = ̂∇̂λ J^(-1)

Rq : note that the two formulae are different because we decided to break our own convention by writing, for a scalar function,
∇f as a column vector instead of a row vector.

## Proof (to be also written in Latex)
x are the physical coords, X the reference coords. F(X) = x the mapping ref -> phys
To compute an integral such as ∫g(x)dx, we map the integral on a ref element:
∫g(x)dx = ∫g∘F(X)dX


1) vector case : λ is a vector function
if g(x) = ∇λ(x); i.e g(x) = ∂λi/∂xj, then we are looking for ∂λi/∂xj ∘ F. We now that λi = ̂λi ∘ F^-1,
so (∂λi/∂xj)(x) = (∂(̂λi∘F^-1)/∂xj)(x) = (∂Fk^-1/∂xj)(x)*(∂̂λi/∂Xk)(X), hence
∂λi/∂xj = (∂Fk^-1/∂xj) * (∂̂λi/∂Xk ∘ F^-1)
So if we compose with `F` to obtain the seeked term:
(∂λi/∂xj) ∘ F = [(∂Fk^-1/∂xj) ∘ F] * (∂̂λi/∂Xk)
Now, we define (J^-1)_kj as [(∂Fk^-1/∂xj) ∘ F]. Note that J^-1 is a function of X, whereas ∂Fk^-1/∂xj is
a function of x...
For the matrix-matrix product to be realised, we have to commute (J^-1)_kj and ∂̂λi/∂Xk
(which is possible since we are working on scalar quantities thanks to Einstein convention):
(∂λi/∂xj) ∘ F = (∂̂λi/∂Xk) * [(∂Fk^-1/∂xj) ∘ F]
∇λ_ij ∘ F = ̂∇̂λ_ik (J^-1)_kj

To conclude, when we want to integrate ∇λ we need to calculate ∇λ ∘ F and this term is equal to
̂∇̂λ J^-1

2) scalar case : λ is a scalar function
What has been said in 1 remains true (except i = 1 only). So we have:
(∂λ/∂xj) ∘ F = [(∂Fk^-1/∂xj) ∘ F] * (∂̂λ/∂Xk),
which could be written as
∇λ_j ∘ F = (J^-1)_kj * ∇λ_k

However, because of the convention retained for gradient of a scalar function, (∂̂λ/∂Xk) is a column-vector so
to perform a matrix-(column-vector) product, we need to transpose J^-1:
∇λ_j ∘ F = transpose(J^-1)_jk * ∇λ_k

# Dev notes
* The signature used to be `lOp::Gradient{O,<:Tuple{AbstractCellFunction{ReferenceDomain}}}`, but for some
reason I don't understand, it didn't work for vector-valued shape functions.
* We have an `if` because the formulae is sligthly different for a scalar function or a vector function (see maths section)

TODO:
* improve formulae with a reshape
* Specialize for a ShapeFunction to use the hardcoded version instead of ForwardDiff
"""
function gradient(
    op::AbstractLazy,
    cPoint::CellPoint{ReferenceDomain},
    gs::AbstractGradientStyle,
)
    f(_ξ) = op(CellPoint(_ξ, get_cellinfo(cPoint), ReferenceDomain()))
    f = Base.Fix2(_op_cpoint, (op, cPoint))
    ξ = get_coord(cPoint)
    valS = _size_codomain(f, ξ)
    cInfo = get_cellinfo(cPoint)
    cnodes = nodes(cInfo)
    ctype = celltype(cInfo)
    return ∂fξ_∂x(gs, f, valS, ctype, cnodes, ξ)

    # ForwarDiff applied in the PhysicalDomain : not viable because
    # the inverse mapping is not always known.
    # cPoint_phys = change_domain(cPoint, PhysicalDomain())
    # f(ξ) = op(CellPoint(ξ, get_cellinfo(cPoint_phys), PhysicalDomain()))
    # fx = f(get_coord(cPoint_phys))
    # return _gradient_or_jacobian(Val(length(fx)), f, get_coord(cPoint_phys))
end

function _op_cpoint(ξ, (op, cPoint))
    op(CellPoint(ξ, get_cellinfo(cPoint), ReferenceDomain()))
end

# Hack to fix type inference in julia 1.10
function ForwardDiffExt.static_dual_eval(
    ::Type{T},
    f::Base.Fix2{typeof(_op_cpoint)},
    x::StaticArray,
) where {T}
    @noinline f(@noinline ForwardDiffExt.dualize(T, x))
end
# Hack to fix type inference in julia 1.10
@noinline function ForwardDiff.jacobian(f::Base.Fix2{typeof(_op_cpoint)}, x::StaticArray)
    @noinline ForwardDiff.vector_mode_jacobian(f, x)
end
# Hack to fix type inference in julia 1.10
function ForwardDiff.vector_mode_jacobian(f::Base.Fix2{typeof(_op_cpoint)}, x::StaticArray)
    T = typeof(ForwardDiff.Tag(f, eltype(x)))
    return ForwardDiffExt.extract_jacobian(T, ForwardDiffExt.static_dual_eval(T, f, x), x)
end

∂fξ_∂x(::VolumicGradientStyle, args...) = ∂fξ_∂x(args...)
∂fξ_∂x(::TangentialGradientStyle, args...) = ∂fξ_∂x_hypersurface(args...)

_size_codomain(f, x) = Val(length(f(x)))
_size_codomain(f::AbstractCellFunction, x) = Val(get_size(f))

function gradient(
    op::AbstractLazy,
    cPoint::CellPoint{PhysicalDomain},
    ::VolumicGradientStyle,
)
    # Fow now this version is "almost" never used because we never evaluate functions
    # in the physical domain (at least in (bi)linear forms).
    f(x) = op(CellPoint(x, cPoint.cellinfo, PhysicalDomain()))
    valS = _size_codomain(f, get_coord(cPoint))
    return _gradient_or_jacobian(valS, f, get_coord(cPoint))
end

# dispatch on codomain size (this is only used for functions evaluated on the physical domain)
_gradient_or_jacobian(::Val{1}, f, x) = ForwardDiff.gradient(f, x)
_gradient_or_jacobian(::Val{S}, f, x) where {S} = ForwardDiff.jacobian(f, x)

function gradient(
    cellFunction::AbstractCellShapeFunctions{<:ReferenceDomain},
    cPoint::CellPoint{ReferenceDomain},
    gs::AbstractGradientStyle,
)
    cnodes = get_cellnodes(cPoint)
    ctype = get_celltype(cPoint)
    ξ = get_coord(cPoint)
    fs = get_function_space(cellFunction)
    n = Val(get_size(cellFunction))
    MapOver(_grad_shape_functions(gs, fs, n, ctype, cnodes, ξ))
end

∂λξ_∂x(::VolumicGradientStyle, args...) = ∂λξ_∂x(args...)
∂λξ_∂x(::TangentialGradientStyle, args...) = ∂λξ_∂x_hypersurface(args...)

function _grad_shape_functions(
    gs::AbstractGradientStyle,
    fs::AbstractFunctionSpace,
    n::Val{1},
    ctype,
    cnodes,
    ξ,
)
    grad = ∂λξ_∂x(gs, fs, n, ctype, cnodes, ξ)
    _reshape_gradient_shape_function_impl(grad, n)
end

@generated function _reshape_gradient_shape_function_impl(
    a::SMatrix{Ndof, Ndim},
    ::Val{1},
) where {Ndof, Ndim}
    _exprs = [[:(a[$i, $j]) for j in 1:Ndim] for i in 1:Ndof]
    exprs = [:(SA[$(_expr...)]) for _expr in _exprs]
    return :(tuple($(exprs...)))
end

function _grad_shape_functions(
    gs::AbstractGradientStyle,
    fs::AbstractFunctionSpace,
    n::Val{N},
    ctype,
    cnodes,
    ξ,
) where {N}
    # Note that the code below is identical to _grad_shape_functions(gs, fs, ::Val{1}, (...))
    # However, for some space we might want a different implementation (for a different function space)
    # and this specific version (::Val{n}) will be the one to specialize. Whereas the scalar one will
    # always be true.
    grad = ∂λξ_∂x(gs, fs, Val(1), ctype, cnodes, ξ)
    _reshape_gradient_shape_function_impl(grad, n)
end

@generated function _reshape_gradient_shape_function_impl(
    a::SMatrix{Ndof_sca, Ndim, T},
    ::Val{Ncomp},
) where {Ndof_sca, Ndim, T, Ncomp}
    z = zero(T)
    exprs_tot = []
    exprs = Matrix{Any}(undef, Ncomp, Ndim)
    for icomp in 1:Ncomp
        for idof_sca in 1:Ndof_sca
            for i in 1:Ncomp
                for j in 1:Ndim
                    exprs[i, j] = i == icomp ? :(a[$idof_sca, $j]) : :($z)
                end
            end
            push!(exprs_tot, :(SMatrix{$Ncomp, $Ndim, $T}($(exprs...))))
        end
    end
    :(tuple($(exprs_tot...)))
end

""" Materialization of a Gradient on a cellinfo is itself a Gradient, but with its function materialized """
function LazyOperators.materialize(lOp::Gradient, cInfo::CellInfo)
    Gradient(LazyOperators.materialize_args(get_args(lOp), cInfo), gradient_style(lOp))
end

function LazyOperators.materialize(
    lOp::Gradient{O, <:Tuple{Vararg{AbstractCellFunction}}},
    cPoint::CellPoint,
) where {O}
    f, = get_args(lOp)
    gradient(f, cPoint, gradient_style(lOp))
end

function LazyOperators.materialize(
    lOp::Gradient{O, <:Tuple{AbstractLazy}},
    cPoint::CellPoint,
) where {O}
    f, = get_args(lOp)
    gradient(f, cPoint, gradient_style(lOp))
end

function LazyOperators.materialize(
    lOp::Gradient{O, <:Tuple},
    sideInfo::AbstractSide,
) where {O}
    arg = LazyOperators.materialize_args(get_args(lOp), sideInfo)
    return Gradient(arg, gradient_style(lOp))
end

"""
Using MultiFESpace, we may want to compute Gradient(v) where v = (v1,v2,...). So `v` is a
Tuple of LazyMapOver since v1, v2 are LazyMapOver (i.e they represent all the shape functions at once)
"""
function LazyOperators.materialize(
    lOp::Gradient{O, <:Tuple{LazyMapOver}},
    cPoint::CellPoint,
) where {O}
    return MapOver(
        materialize(
            x -> materialize(Gradient(x, gradient_style(lOp)), cPoint),
            get_args(lOp)...,
        ),
    )
end

# Specialize the previous method when `Gradient` is applied
# to two levels of LazyMapOver in order to help compiler inference
# to deal with recursion.
# This may be used by `bilinear_assemble` when a bilinear form
# containing a `Gradient` is evaluated at a `CellPoint`
function LazyOperators.materialize(
    lOp::Gradient{O, <:Tuple{LazyMapOver{<:NTuple{N, LazyMapOver}}}},
    cPoint::CellPoint,
) where {O, N}
    return MapOver(
        materialize(
            x -> materialize(Gradient(x, gradient_style(lOp)), cPoint),
            get_args(lOp)...,
        ),
    )
end

function LazyOperators.materialize(
    lOp::Gradient{O, <:Tuple{LazyMapOver{<:CellShapeFunctions}}},
    cPoint::CellPoint,
) where {O}
    return materialize(Gradient(get_args(get_args(lOp)...), gradient_style(lOp)), cPoint)
end

function LazyOperators.materialize(
    lOp::Gradient{O, <:Tuple{LazyMapOver{<:CellShapeFunctions}}},
    sidePoint::AbstractSide{Nothing, <:Tuple{FacePoint}},
) where {O}
    op_side = get_operator(sidePoint)
    cellPoint = op_side(get_args(sidePoint)...)
    return materialize(Gradient(get_args(get_args(lOp)...), gradient_style(lOp)), cellPoint)
end

function LazyOperators.materialize(
    lOp::Gradient{O, <:Tuple{LazyMapOver}},
    sidePoint::AbstractSide{Nothing, <:Tuple{FacePoint}},
) where {O}
    ∇x = tuplemap(
        x -> materialize(Gradient(x, gradient_style(lOp)), sidePoint),
        get_args(get_args(lOp)...),
    )
    return MapOver(∇x)
end

function LazyOperators.materialize(
    lOp::Gradient{O, <:Tuple{Vararg{AbstractCellFunction}}},
    sidePoint::AbstractSide{Nothing, <:Tuple{FacePoint}},
) where {O}
    op_side = get_operator(sidePoint)
    cellPoint = op_side(get_args(sidePoint)...)
    materialize(lOp, cellPoint)
end
