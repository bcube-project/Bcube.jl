struct Gradient{O, A} <: AbstractLazyOperator{O, A}
    args::A # `args` represent a Tuple containing the function whom gradient is computed
end
LazyOperators.get_args(lOp::Gradient) = lOp.args
LazyOperators.get_operator(lOp::Gradient) = lOp
Gradient(args::Tuple) = Gradient{Nothing, typeof(args)}(args)
Gradient(f) = Gradient((f,))
const ∇ = Gradient
∇(::NullOperator) = NullOperator()

""" Materialization of a Gradient on a cellinfo is itself a Gradient, but with its function materialized """
function LazyOperators.materialize(lOp::Gradient, cInfo::CellInfo)
    Gradient(LazyOperators.materialize_args(get_args(lOp), cInfo))
end

function LazyOperators.materialize(
    lOp::Gradient{O, <:Tuple{Vararg{AbstractCellFunction}}},
    cPoint::CellPoint,
) where {O}
    f, = get_args(lOp)
    Gradient(f, cPoint)
end

"""
Materialization of a `Gradient` on a `CellPoint`. Only valid for a function and a `CellPoint` defined on
the reference domain.

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
function Gradient(
    cellFunction::AbstractCellFunction{<:ReferenceDomain},
    cPoint::CellPoint{ReferenceDomain},
)
    cnodes = get_cellnodes(cPoint)
    ctype = get_celltype(cPoint)
    ξ = get_coord(cPoint)
    f = get_function(cellFunction)
    m = mapping_jacobian_inv(cnodes, ctype, ξ)
    _Gradient(cellFunction(cPoint), f, ξ, m)
end
_Gradient(::Real, f, ξ::AbstractArray, m) = transpose(m) * ForwardDiff.gradient(f, ξ)
_Gradient(::AbstractArray, f, ξ::AbstractArray, m) = ForwardDiff.jacobian(f, ξ) * m

function Gradient(
    cellFunction::AbstractCellShapeFunctions{<:ReferenceDomain},
    cPoint::CellPoint{ReferenceDomain},
)
    cnodes = get_cellnodes(cPoint)
    ctype = get_celltype(cPoint)
    ξ = get_coord(cPoint)
    fs = get_function_space(cellFunction)
    n = Val(get_size(cellFunction))
    MapOver(grad_shape_functionsNA(fs, n, ctype, cnodes, ξ))
end

function Gradient(cellFunction::AbstractCellFunction{<:PhysicalDomain}, cPoint::CellPoint)
    f = get_function(cellFunction)
    x = change_domain(cPoint, PhysicalDomain()) # transparent if already in PhysicalDomain
    return ForwardDiff.gradient(f, get_coord(x))
end

function grad_shape_functionsNA(fs::AbstractFunctionSpace, n::Val{1}, ctype, cnodes, ξ)
    grad = grad_shape_functions(fs, n, ctype, cnodes, ξ)
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

function grad_shape_functionsNA(
    fs::AbstractFunctionSpace,
    n::Val{N},
    ctype,
    cnodes,
    ξ,
) where {N}
    grad = grad_shape_functions(fs, Val(1), ctype, cnodes, ξ)
    _reshape_gradient_shape_function_impl(grad, n)
end

@generated function _reshape_gradient_shape_function_impl(
    a::SMatrix{Ndof_sca, Ndim, T},
    ::Val{Ncomp},
) where {Ndof_sca, Ndim, T, Ncomp}
    z = zero(T)
    expr_0 = [:($z) for j in 1:Ndim]
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

"""
Using MultiFESpace, we may want to compute Gradient(v) where v = (v1,v2,...). So `v` is a
Tuple of LazyMapOver since v1, v2 are LazyMapOver (i.e they represent all the shape functions at once)
"""
function LazyOperators.materialize(
    lOp::Gradient{O, <:Tuple{LazyMapOver}},
    cPoint::CellPoint,
) where {O}
    return MapOver(materialize(Base.Fix2(materialize, cPoint) ∘ Gradient, get_args(lOp)...))
end

function LazyOperators.materialize(
    lOp::Gradient{O, <:Tuple{LazyMapOver{<:CellShapeFunctions}}},
    cPoint::CellPoint,
) where {O}
    return materialize(Gradient(get_args(get_args(lOp)...)), cPoint)
end
