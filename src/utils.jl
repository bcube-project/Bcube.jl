"""
    densify(a::AbstractVector, [permute_back=false])

Remap values of `a`, i.e if `length(a) = n`, make sure that every value of `a`
belongs to [1,n] (in other words, eliminate holes in values of `a`).
"""
function densify(a::AbstractVector{T}; permute_back::Bool = false) where {T <: Integer}
    _a = unique(a)
    remap = Dict{eltype(a), eltype(a)}(_a[i] => i for i in 1:length(_a)) #zeros(eltype(a), maximum(_a))
    dense_a = [remap[aᵢ] for aᵢ in a]
    permute_back ? (return dense_a, remap) : (return dense_a)
end

function densify!(a::AbstractVector{T}) where {T <: Integer}
    a .= densify(a)
end

"""
    myrand(n, xmin = 0., xmax = 1.)

Create a random vector of size `n` with `xmin` and `xmax` as bounds. Use `xmin`
and `xmax` either as scalars or as vectors of size `n`.
"""
myrand(n, xmin = 0.0, xmax = 1.0) = (xmax .- xmin) .* rand(n) .+ xmin

"""
    rawcat(x)

Equivalent to `reduce(vcat,vec.(x))`
"""
function rawcat(
    x::Union{NTuple{N, A}, Vector{A}},
) where {N, A <: Union{AbstractArray{T}, Tuple{Vararg{T}}}} where {T}
    isempty(x) && return T[]
    n = sum(length, x)
    y = Vector{T}(undef, n)
    i = 0
    for x1 in x, x2 in x1
        i += 1
        @inbounds y[i] = x2
    end
    @assert i == n "Dimension mismatch"
    return y
end
rawcat(x::Vector{T}) where {T} = x

"""
    matrix_2_vector_of_SA(a)

"Reshape" a matrix of size (m,n) into a `Vector` (of size n) of `StaticVector`` (of size m)
"""
matrix_2_vector_of_SA(a) = vec(reinterpret(SVector{size(a, 1), eltype(a)}, a))

"""
    convert_to_vector_of_union(a::Vector{T}) where T

Convert a vector 'a', whose the element type is abstract, to
a vector whose the element type is a 'Union' of concrete types (if it is possible)
"""
function convert_to_vector_of_union(a::Vector{T}) where {T}
    if !isconcretetype(T)
        types = unique(typeof, a)
        a_ = Union{typeof.(types)...}[a...]
        return a_
    else
        return a
    end
end

function _soft_max(x, y, k)
    m = min(x, y)
    M = max(x, y)
    return M + log(1.0 + exp((m - M) * k)) / k
end
#soft_max(x::ForwardDiff.Dual, y::ForwardDiff.Dual, k=10) = _soft_max(x,y,k)
#soft_max(x, y, k=10) = _soft_max(x,y,k)
soft_max(x, y) = max(x, y) #default

_soft_min(x, y, k) = -log(exp(-k * x) + exp(-k * y)) / k
#soft_min(x::ForwardDiff.Dual, y::ForwardDiff.Dual, k=10) = min(x,y)#_soft_min(x,y,k)
#soft_min(x,y, k=10) = _soft_min(x,y,k)
soft_min(x, y) = min(x, y) #default

_soft_abs(x, k) = √(x^2 + k^2)
#soft_abs(x, k=0.001) =  abs(x) #_soft_abs(x,k)
#soft_abs(x::ForwardDiff.Dual, k=0.001) = _soft_abs(x,k)# x*tanh(x/k)
soft_abs(x) = abs(x)

soft_extrema(itr) = soft_extrema(identity, itr)

function soft_extrema(f, itr)
    y = iterate(itr)
    y === nothing && throw(ArgumentError("collection must be non-empty"))
    (v, s) = y
    vmin = vmax = f(v)
    while true
        y = iterate(itr, s)
        y === nothing && break
        (x, s) = y
        fx = f(x)
        vmax = soft_max(fx, vmax)
        vmin = soft_min(fx, vmin)
    end
    return (vmin, vmax)
end

# warning : it relies on julia internal!
raw_unzip(a) = a.is

"""
    WiddenAsUnion{T}

Type used internally by `map_and_widden_as_union`.
"""
struct WiddenAsUnion{T}
    a::T
end

function Base.promote_typejoin(
    ::Type{WiddenAsUnion{T1}},
    ::Type{WiddenAsUnion{T2}},
) where {T1, T2}
    return Union{WiddenAsUnion{T1}, WiddenAsUnion{T2}}
end
function Base.promote_typejoin(
    ::Type{Union{WiddenAsUnion{T1}, T2}},
    ::Type{WiddenAsUnion{T3}},
) where {T1, T2, T3}
    return Union{WiddenAsUnion{T1}, T2, WiddenAsUnion{T3}}
end

unwrap(x::WiddenAsUnion) = x.a
unwrap(::Type{WiddenAsUnion{T}}) where {T} = T
unwrap(::Type{Union{WiddenAsUnion{T1}, T2}}) where {T1, T2} = Union{unwrap(T1), unwrap(T2)}

"""
    map_and_widden_as_union(f, c...)

Transform collection `c` by applying `f` to each element. For multiple collection arguments,
apply `f` elementwise, and stop when any of them is exhausted.
The difference with [`map`](@ref) comes from the type of the result:
`map_and_widden_as_union` uses `Union` to widden the type of the resulting
collection and potentially reduce type instabilities.

# Examples
```jldoctest
julia> map_and_widden_as_union(x -> x * 2, [1, 2, [3,4]])
3-element Vector{Union{Int64, Vector{Int64}}}:
 2
 4
  [6, 8]
```
instead of:
```jldoctest
julia> map(x -> x * 2, [1, 2, [3,4]])
3-element Vector{Any}:
 2
 4
  [6, 8]
```

# Implementation

When `f` is applied to one element of `c`, the result
is wrapped in a type `WrapAsUnion` on which specific
`promote_typejoin` rules are applied.
When `map_and_widden_as_union` is applied
to collections of heterogeneous elements, these rules
help  to infer the type of the resulting collection
as a `Union` of different types instead of a
widder (abstract) type.
"""
function map_and_widden_as_union(f, c...)
    g(x...) = WiddenAsUnion(f(x...))
    a = map(g, c...)
    T = unwrap(eltype(a))
    return T[unwrap(x) for x in a]
end

"""
    myfindfirst(predicate::Function, t::Tuple)

Function equivalent to `Base.findfirst(predicate::Function, A)`.
This version is optimized to work better on `Tuple` by avoiding
type instabilities.

# source:
https://discourse.julialang.org/t/why-does-findfirst-t-on-a-tuple-of-typed-only-constant-fold-for-the-first/68893/3
"""
myfindfirst(predicate::Function, t::Tuple) = _findfirst(predicate, 1, t)
_findfirst(f, i, x) = f(first(x)) ? i : _findfirst(f, i + 1, Base.tail(x))
_findfirst(f, i, ::Tuple{}) = nothing

# see : https://discourse.julialang.org/t/why-doesnt-the-compiler-infer-the-type-of-this-function/35005/3
@inline function tuplemap(f, t1::Tuple{Vararg{Any, N}}, t2::Tuple{Vararg{Any, N}}) where {N}
    (f(first(t1), first(t2)), tuplemap(f, Base.tail(t1), Base.tail(t2))...)
end
@inline function tuplemap(f, t::Tuple{Vararg{Any, N}}) where {N}
    (f(first(t)), tuplemap(f, Base.tail(t))...)
end
@inline tuplemap(f, ::Tuple{}) = ()
@inline tuplemap(f, ::Tuple{}, ::Tuple{}) = ()

# temporary solution (API needed ?)
function _solve!(x, A, b, backend::Bcube.AbstractBcubeBackend)
    __solve!(x, A, b, get_backend(backend))
end
function __solve!(x, A, b, backend)
    x .= A \ b
    return nothing
end

"""
    cumsum_exclusive(a::AbstractVector)
    cumsum_exclusive(a::Tuple)

Return the cumlative sum, excluding the current element.
"""
function cumsum_exclusive(a::AbstractVector)
    b = cumsum(a)[1:(end - 1)]
    return vcat(zero(eltype(a)), b)
end

function cumsum_exclusive(a::Tuple)
    b = Base.front(cumsum(a))
    return (zero(eltype(a)), b...)
end
