module LazyOperators

import Base:
    *,
    /,
    +,
    -,
    ^,
    max,
    min,
    sqrt,
    abs,
    tan,
    sin,
    cos,
    tanh,
    sinh,
    cosh,
    atan,
    asin,
    acos,
    zero,
    one,
    materialize

using LinearAlgebra
import LinearAlgebra: dot, transpose, tr

export AbstractLazy
export AbstractLazyOperator
export LazyOperator
export LazyWrap
export unwrap
export materialize
export show_lazy_operator
export print_tree_prefix
export pretty_name_style
export pretty_name
export NullOperator

export LazyMapOver
export MapOver

any_of_type(a::T, ::Type{T}) where {T} = Val(true)
any_of_type(a, ::Type{T}) where {T} = Val(false)
@generated function any_of_type(a::Tuple, ::Type{T}) where {T}
    _T = fieldtypes(a)
    if any(map(x -> isa(x, Type{<:T}), _T))
        return :(Val(true))
    else
        return :(Val(false))
    end
end

include("lazy_operator.jl")
include("algebra.jl")
include("mapover.jl")

end
