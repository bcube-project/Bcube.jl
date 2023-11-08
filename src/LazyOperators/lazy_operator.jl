"""
    abstract type AbstractLazy end

Subtypes must implement:
- `materialize(a::AbstractLazy, x)`
and optionally:
- `pretty_name(a::AbstractLazy)`
- `show_lazy_operator(a::AbstractLazy; level=1, indent=4, islast=(true,))`
"""
abstract type AbstractLazy end

function materialize(a::AbstractLazy, x)
    error("`materialize` is not defined for:\n $(typeof(a)) \n and:\n $(typeof(x))")
end

# default rule on tuple is to apply materialize on each element of the tuple
materialize(t::Tuple, x::Vararg{Any, N}) where {N} = LazyWrap(_materialize(t, x...))
function _materialize(t::Tuple, x::Vararg{Any, N}) where {N}
    (materialize(first(t), x...), _materialize(Base.tail(t), x...)...)
end
_materialize(::Tuple{}, ::Vararg{Any, N}) where {N} = ()

pretty_name(a)            = string(typeof(a))
pretty_name(a::Number)    = string(a)
pretty_name(a::Base.Fix1) = "Fix1: (f=" * pretty_name(a.f) * ", x=" * pretty_name(a.x) * ")"
pretty_name(a::Base.Fix2) = "Fix2: (f=" * pretty_name(a.f) * ", x=" * pretty_name(a.x) * ")"
pretty_name_style(a)      = Dict(:color => :normal)

function show_lazy_operator(
    a;
    level = 1,
    indent = 4,
    islast = (true,),
    printTupleOp::Bool = false,
)
    print_tree_prefix(level, indent, islast)
    printstyled(pretty_name(a) * "  \n"; pretty_name_style(a)...)
end

function print_tree_prefix(level, indent = 4, islast = (true,))
    _color = map(x -> x ? :light_black : :normal, islast)
    motif = "│" * join(fill(" ", max(0, indent - 1)))
    if level > 1
        for lev in 2:(level - 1)
            printstyled(motif; color = _color[lev])
        end
        prefix = "└" * join(fill("─", max(0, indent - 2))) * " "
        printstyled(prefix; color = :normal)
    end
end

function pretty_name(f::Function)
    name = string(typeof(f))
    delimiter = '"'
    if contains(name, delimiter)
        splitname = split(name, delimiter)
        name = splitname[1] * delimiter * splitname[2] * delimiter
    end
    return name
end
pretty_name_style(::Function) = Dict(:color => :light_green)
pretty_name(::Nothing) = ""

"""
    AbstractLazyWrap{A} <: AbstractLazy

Subtypes must implement:
- `get_args(a::AbstractLazyWrap)`
and optionally:
- `unwrap(a::AbstractLazyWrap)`
- `pretty_name(a::AbstractLazyWrap)`
- `pretty_name_style(::AbstractLazyWrap) `
- `show_lazy_operator(a::AbstractLazyWrap; level=1, indent=4, islast=(true,))`
"""
abstract type AbstractLazyWrap{A} <: AbstractLazy end

function get_args(a::AbstractLazyWrap)
    error("Function `get_args` is not defined for type $(typeof(a))")
end
function materialize(a::AbstractLazyWrap, x::Vararg{Any, N}) where {N}
    materialize_args(get_args(a), x...)
end
# Do not return a tuple when only one element is wrapped:
function materialize(a::AbstractLazyWrap{<:Tuple{T}}, x::Vararg{Any, N}) where {N, T}
    first(materialize_args(get_args(a), x...))
end
unwrap(a) = a
unwrap(a::AbstractLazyWrap) = get_args(a)

pretty_name(a::AbstractLazyWrap) = "$(typeof(a))"
pretty_name_style(::AbstractLazyWrap) = Dict(:color => :light_black)

function show_lazy_operator(
    a::AbstractLazyWrap;
    level = 1,
    indent = 4,
    islast = (true,),
    printTupleOp::Bool = false,
)
    level == 1 && println("\n---------------")
    print_tree_prefix(level, indent, islast)
    printstyled(pretty_name(a) * "  \n"; pretty_name_style(a)...)
    _islast = (islast..., true)
    show_lazy_operator(
        get_args(a);
        level = (level + 1),
        islast = _islast,
        printTupleOp = printTupleOp,
    )
    level == 1 && println("---------------")
end

struct LazyWrap{A <: Tuple} <: AbstractLazyWrap{A}
    args::A
end

LazyWrap(args...) = LazyWrap{typeof(args)}(args)
get_args(a::LazyWrap) = a.args
pretty_name(::LazyWrap) = "LazyWrap"

"""
Subtypes must implement:
- `get_args(op::AbstractLazyOperator)`
- `get_operator(op::AbstractLazyOperator)`
- `materialize(Op::AbstractLazyOperator, x)`

Subtypes can implement:
- `show_lazy_op`
"""
abstract type AbstractLazyOperator{O, A} <: AbstractLazy end

get_type_operator(::Type{<:AbstractLazyOperator{O}}) where {O} = O
get_type_operator(op::AbstractLazyOperator) = get_type_operator(typeof(op))

get_type_args(::Type{<:AbstractLazyOperator{O, A}}) where {O, A} = fieldtypes(a)
get_type_args(op::AbstractLazyOperator) = get_type_args(typeof(op))

function get_args(op::AbstractLazyOperator)
    error("Function `get_args` is not defined for type $(typeof(op))")
end
function get_operator(op::AbstractLazyOperator)
    error("Function `get_operator` is not defined for type $(typeof(op))")
end

pretty_name(::Type{<:AbstractLazyOperator}) = "AbstractLazyOperator"
pretty_name(op::AbstractLazyOperator) = string(nameof(typeof(op)))
pretty_name_style(::Type{<:AbstractLazyOperator}) = Dict(:color => :red)
pretty_name_style(op::AbstractLazyOperator) = pretty_name_style(typeof(op))

function materialize(lOp::AbstractLazyOperator, x)
    op = get_operator(lOp)
    args = materialize_args(get_args(lOp), x)
    materialize_op(op, args...)
end

# default
materialize_op(op::O, args::Vararg{Any, N}) where {O, N} = op(args...)

# specific operator materialization for composition:
# if `args` contains one `AbstractLazy` type at least then:
#      the result is still a lazy composition
# else:
#      the result is the application of the function (i.e args1)
#      to its args.
@inline function materialize_op(op::typeof(∘), args::Vararg{Any, N}) where {N}
    _materialize_op_compose(op, args...)
end
@inline function _materialize_op_compose(
    op::O,
    arg1::T,
    args::Vararg{Any, N},
) where {O, T, N}
    _materialize_op_compose(any_of_type(args..., AbstractLazy), op, arg1, args...)
end
@inline function _materialize_op_compose(
    ::Val{true},
    op::O,
    args::Vararg{Any, N},
) where {O, N}
    op(args...)
end
@inline function _materialize_op_compose(
    ::Val{false},
    op,
    arg1::T1,
    args::Vararg{Any, N},
) where {T1, N}
    _may_apply_on_splat(arg1, args...)
end
_may_apply_on_splat(f, a::Tuple) = f(a...)
_may_apply_on_splat(f, a) = f(a)

# avoid:
# @inline materialize_args(args::Tuple, x ) = map(Base.Fix2(materialize, x), args)
# as inference seems to fail rapidely
materialize_args(args::Tuple, x::Vararg{Any, N}) where {N} = _materialize_args(args, x...)
function _materialize_args(args::Tuple, x::Vararg{Any, N}) where {N}
    (materialize(first(args), x...), _materialize_args(Base.tail(args), x...)...)
end
_materialize_args(args::Tuple{}, x::Vararg{Any, N}) where {N} = ()

# @generated materialize_args(args::Tuple, x...) = _materialize_args_impl(args)
# function _materialize_args_impl(::Type{<:Tuple{Vararg{Any,N}}}) where {N}
#     exprs = [:(materialize(args[$i], x...)) for i in 1:N]
#     return :(tuple($(exprs...)))
# end

# function show_lazy_operator(op; level=1, indent=4, prefix="")
#     println(prefix*string(typeof(op)))
# end

function show_lazy_operator(
    op::AbstractLazyOperator;
    level = 1,
    indent = 4,
    islast = (true,),
)
    level == 1 && println("\n---------------")
    print_tree_prefix(level, indent, islast)
    printstyled(pretty_name(op) * ": "; pretty_name_style(op)...)
    printstyled(
        pretty_name(get_operator(op)) * "  \n";
        pretty_name_style(get_operator(op))...,
    )
    args = get_args(op)
    show_lazy_operator(args; level = (level + 1), islast = islast, printTupleOp = false)
    level == 1 && println("---------------")
end

_rm_first_character(a::String) = last(a, length(a) - 1)
function _select_character(a::String, range)
    a[collect(eachindex(a))[first(range):min(last(range), length(a))]]
end

pretty_name(::Tuple) = "Tuple:"
pretty_name_style(::Tuple) = Dict(:color => :light_black)
function show_lazy_operator(
    t::Tuple;
    level = 1,
    indent = 4,
    islast = (true,),
    printTupleOp::Bool = true,
)
    _show_lazy_operator(Val(printTupleOp), t; level = level, islast = islast)
end
function _show_lazy_operator(::Val{true}, t::Tuple; level = 1, indent = 4, islast = (true,))
    level == 1 && println("\n---------------")
    print_tree_prefix(level, indent, islast)
    printstyled(pretty_name(t) * ": \n"; pretty_name_style(t)...)
    _show_lazy_operator(
        Val(false),
        t;
        level = level + 1,
        indent = indent,
        islast = (islast...,),
    )
    level == 1 && println("---------------")
    nothing
end
function _show_lazy_operator(
    ::Val{false},
    t::Tuple;
    level = 1,
    indent = 4,
    islast = (true,),
)
    for (i, x) in enumerate(t)
        _islast = (islast..., i == length(t))
        show_lazy_operator(x; level = level, indent = indent, islast = _islast)
    end
    nothing
end

(lOp::AbstractLazyOperator)(x::Vararg{Any, N}) where {N} = materialize(lOp, x...)

struct LazyOperator{O, A <: Tuple} <: AbstractLazyOperator{O, A}
    operator::O
    args::A
end

LazyOperator(op, args...) = LazyOperator{typeof(op), typeof(args)}(op, args)
get_args(op::LazyOperator) = op.args
get_operator(op::LazyOperator) = op.operator

abstract type AbstractNullOperator <: AbstractLazy end
struct NullOperator <: AbstractNullOperator end

pretty_name(::NullOperator) = "∅"
get_operator(a::NullOperator) = nothing
get_args(a::NullOperator) = (nothing,)
materialize(a::NullOperator, x) = a

function show_lazy_operator(op::NullOperator; level = 1, indent = 4, islast = (true,))
    level == 1 && println("\n---------------")
    print_tree_prefix(level, indent, islast)
    printstyled(pretty_name(op) * "\n"; pretty_name_style(op)...)
    level == 1 && println("---------------")
end
