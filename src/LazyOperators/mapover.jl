"""
    AbtractLazyMapOver{A} <: AbstractLazyWrap{A}

Subtypes must implement:
- `get_args(lmap::AbtractLazyMapOver)`
and optionally:
- `pretty_name(a::AbtractLazyMapOver)`
- `pretty_name_style(::AbtractLazyMapOver)`
- `show_lazy_operator(a::AbtractLazyMapOver; level=1, indent=4, islast=(true,))`
"""
abstract type AbstractLazyMapOver{A} <: AbstractLazyWrap{A} end

"""
    LazyMapOver{A} <: AbstractLazyMapOver{A}

Type used to wrap data (of type `A`) on which functions must
be mapped over it.
"""
struct LazyMapOver{A} <: AbstractLazyMapOver{A}
    args::A
end

LazyMapOver(args::Tuple) = LazyMapOver{typeof(args)}(args)
LazyMapOver(args...) = LazyMapOver{typeof(args)}(args)

get_args(a::LazyMapOver) = a.args

pretty_name(::LazyMapOver) = "LazyMapOver"
pretty_name_style(::LazyMapOver) = Dict(:color => :blue)

materialize(a::LazyMapOver, x) = LazyMapOver(lazy_map_over(Base.Fix2(materialize, x), a))
materialize(f::F, a::LazyMapOver) where {F <: Function} = lazy_map_over(f, a)

(a::LazyMapOver)(x::Vararg{Any, N}) where {N} = materialize(a, x...)

function lazy_map_over(f::F, a::Vararg{LazyMapOver, N}) where {F <: Function, N}
    _tuplemap(f, _tuplemap(get_args, a)...)
end

# Specialize the previous method for two levels of LazyMapOver
# in order to help compiler inference to deal with recursion.
function lazy_map_over(
    f::F,
    a::Vararg{LazyMapOver{<:NTuple{N1, LazyMapOver}}, N},
) where {F <: Function, N, N1}
    _tuplemap(f, _tuplemap(get_args, a)...)
end

"""
    _tuplemap(f::F, t::Vararg{NTuple{N1, Any}, N})

Transform (multi-)tuple `t` by applying `f` (elementwise) to each element, similarly to `Base.map(f, t...)`.
This method is implemented recursively to help inference and improve performance.
"""
@inline function _tuplemap(f::F, t::Vararg{NTuple{N1, Any}, N}) where {F, N, N1}
    @inline
    heads = map(first, t)
    tails = map(Base.tail, t)
    (f(heads...), _tuplemap(f, tails...)...)
end
@inline _tuplemap(f::F, ::Vararg{Tuple{}, N}) where {F, N} = ()

# _first(a::NTuple{N}, b::Vararg{NTuple{N}}) where {N} = (first(a), _first(b)...)
# _tail(a::NTuple{N}, b::Vararg{NTuple{N}}) where {N} = (first(a), _first(b)...)
# function _tuplemap(f::F, t1::Tuple{Vararg{Any, N}}, t2::Tuple{Vararg{Any, N}}) where {F, N}
#     (f(first(t1), first(t2)), _tuplemap(f, Base.tail(t1), Base.tail(t2))...)
# end
# function _tuplemap(f::F, t::Tuple{Vararg{Any, N}}) where {F, N}
#     (f(first(t)), _tuplemap(f, Base.tail(t))...)
# end
# _tuplemap(f, ::Tuple{}) = ()
# _tuplemap(f, ::Tuple{}, ::Tuple{}) = ()

unwrap(a::Tuple{Vararg{Any, N}}) where {N} = _unwrap_tuple(a)
_unwrap_tuple(a::Tuple{}) = ()
_unwrap_tuple(a::Tuple{T}) where {T} = (unwrap(first(a)),)
_unwrap_tuple(a::Tuple) = (unwrap(first(a)), _unwrap_tuple(Base.tail(a))...)
