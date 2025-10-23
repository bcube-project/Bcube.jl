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

abstract type AbstractMapOver{A} end

get_basetype(::Type{<:T}) where {T <: AbstractMapOver} = error("to be defined")
get_basetype(a::AbstractMapOver) = get_basetype(typeof(a))

unwrap(a::AbstractMapOver) = a.args
unwrap(a::Tuple{Vararg{Any, N}}) where {N} = _unwrap_map_over(a)
_unwrap_map_over(a::Tuple{}) = ()
_unwrap_map_over(a::Tuple{T}) where {T} = (unwrap(first(a)),)
_unwrap_map_over(a::Tuple) = (unwrap(first(a)), _unwrap_map_over(Base.tail(a))...)

(a::AbstractMapOver)(x::Vararg{Any, N}) where {N} = evaluate(a, x...)
function evaluate(a::T, x::Vararg{Any, N}) where {T <: AbstractMapOver, N}
    map_over(Base.Fix2(evaluate, x), a)
end
evaluate(a, x::AbstractMapOver) = map_over(Base.Fix1(evaluate, a), x)

Base.map(f::F, a::Vararg{AbstractMapOver, N}) where {F <: Function, N} = map_over(f, a...)

for f in LazyBinaryOp
    @eval ($f)(a::AbstractMapOver, b::AbstractMapOver) = map_over($f, a, b)
    @eval ($f)(a::AbstractMapOver, b) = map_over(Base.Fix2($f, b), a)
    @eval ($f)(a, b::AbstractMapOver) = map_over(Base.Fix1($f, a), b)
    #fix ambiguity
    #@eval ($f)(a::AbstractMapOver, b::AbstractLazy) = map_over(Base.Fix2($f, b), a)
    #@eval ($f)(a::AbstractLazy, b::AbstractMapOver) = map_over(Base.Fix1($f, a), b)
end

for f in LazyUnaryOp
    @eval ($f)(a::AbstractMapOver) = map_over($f, a)
end

# remove ambiguity:
Base.:*(::AbstractMapOver, ::NullOperator) = NullOperator()
Base.:*(::NullOperator, ::AbstractMapOver) = NullOperator()
Base.:/(::AbstractMapOver, ::NullOperator) = error("Invalid operation")
Base.:/(::NullOperator, b::AbstractMapOver) = NullOperator()
Base.:+(a::AbstractMapOver, ::NullOperator) = a
Base.:+(::NullOperator, b::AbstractMapOver) = b
Base.:-(a::AbstractMapOver, ::NullOperator) = a
Base.:-(::NullOperator, b::AbstractMapOver) = -b
Base.:^(::AbstractMapOver, ::NullOperator) = error("Undefined")
Base.:^(::NullOperator, ::AbstractMapOver) = NullOperator()
Base.max(a::AbstractMapOver, ::NullOperator) = a
Base.max(::NullOperator, b::AbstractMapOver) = b
Base.min(a::AbstractMapOver, ::NullOperator) = a
Base.min(::NullOperator, b::AbstractMapOver) = b
LinearAlgebra.dot(::AbstractMapOver, ::NullOperator) = NullOperator()
LinearAlgebra.dot(::NullOperator, ::AbstractMapOver) = NullOperator()

"""
    map_over(f, args::AbstractMapOver...)

Similar to `Base.map(f, args...)`. To help inference and improve performance,
this method is implemented recursively and is based on method `_tuplemap`
"""
function map_over(
    f::F,
    args::Vararg{AbstractMapOver{<:Tuple{Vararg{Number}}}, N},
) where {F <: Function, N}
    f_a = _tuplemap(f, _tuplemap(unwrap, args)...)
    T = get_basetype(typeof(first(args)))
    T(f_a)
end
function map_over(f::F, args::Vararg{AbstractMapOver, N}) where {F <: Function, N}
    f_a = _map_over(f, _tuplemap(unwrap, args)...)
    T = get_basetype(typeof(first(args)))
    T(f_a)
end
"""
    gen_map_over(f, M, N, args...)

The purpose is to build an `Expr` corresponding to the application of `f` on each
"line" of the `Tuple`s `args`. For instance if `f = +` and `args = ((a,b,c), (d,e,f))`,
we want to build the `Expr`` of the `Tuple` `(a+d, b+e, c+f)`
"""
function gen_map_over_two_tuples(f, N, a, b)
    exprs = [:(f(a[$i], b[$i])) for i in 1:N]
    return :(($(exprs...),))
end
@generated function _map_over(f::F, a::NTuple{N}, b::NTuple{N}) where {F <: Function, N}
    gen_map_over_two_tuples(f, N, a, b)
end
function _map_over(f::F, a::Vararg{Tuple, N}) where {F <: Function, N}
    _heads = Base.heads(a...)
    _tails = Base.tails(a...)
    (__map_over(f, _heads...), _map_over(f, _tails...)...)
end
_map_over(f::F, a::Vararg{Tuple{}, N}) where {F <: Function, N} = ()

function _map_over(::Val{0}, f::F, a::Vararg{Tuple, N}) where {F <: Function, N}
    _tuplemap(f, a...)
end

__map_over(f::F, a::Vararg{Any, N}) where {F, N} = f(a...)
__map_over(f::F, a::Vararg{AbstractMapOver, N}) where {F, N} = map_over(f, a...)

pretty_name(a::AbstractMapOver) = string(get_basetype(a))
pretty_name_style(a::AbstractMapOver) = Dict(:color => :blue)
function show_lazy_operator(a::AbstractMapOver; level = 1, indent = 4, islast = (true,))
    print_tree_prefix(level, indent, islast)
    printstyled(pretty_name(a) * ": "; pretty_name_style(a)...)
    printstyled(pretty_name(a) * "  \n"; pretty_name_style(a)...)
    args = unwrap(a)
    for (i, arg) in enumerate(args)
        _islast = (islast..., i == length(args))
        arg ≠ nothing && show_lazy_operator(arg; level = (level + 1), islast = _islast)
    end
end

"""
    MapOver{A} <: AbstractMapOver{A}

A container used to wrap data for which all materialized
operators on that data must be map over it.
This corresponds to the non-lazy version of `LazyMapOver`.
"""
struct MapOver{A <: Tuple} <: AbstractMapOver{A}
    args::A
end

MapOver(args::Vararg{Any, N}) where {N} = MapOver{typeof(args)}(args)
get_basetype(::Type{<:MapOver}) = MapOver
