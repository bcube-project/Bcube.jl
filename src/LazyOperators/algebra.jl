###############################################################
# Define rules for binary operators
###############################################################

const LazyBinaryOp = (:*, :/, :+, :-, :^, :max, :min, :dot)

for f in LazyBinaryOp
    @eval ($f)(a::AbstractLazy, b::AbstractLazy) = LazyOperator($f, a, b)
    @eval ($f)(a::AbstractLazy, b) = ($f)(a, LazyWrap(b))
    @eval ($f)(a, b::AbstractLazy) = ($f)(LazyWrap(a), b)
end

###############################################################
# Define rules for unary operators
###############################################################

const LazyUnaryOp = (
    :+,
    :-,
    :transpose,
    :tr,
    :sqrt,
    :abs,
    :tan,
    :sin,
    :cos,
    :tanh,
    :sinh,
    :cosh,
    :atan,
    :asin,
    :acos,
    :zero,
    :one,
)

for f in LazyUnaryOp
    @eval ($f)(a::AbstractLazy) = LazyOperator($f, a)
    @eval ($f)(::NullOperator) = NullOperator()
end

###############################################################
# Define rules with `NullOperator`
###############################################################

# For binary `+` and `-`:
for f in (:+, :-)
    @eval ($f)(a, ::NullOperator) = a
    @eval ($f)(::NullOperator, b) = ($f)(b)
    @eval ($f)(::NullOperator, ::NullOperator) = NullOperator()
    @eval ($f)(a::AbstractLazy, ::NullOperator) = a
    @eval ($f)(::NullOperator, b::AbstractLazy) = ($f)(b)
end

# For binary `*` and `dot`:
for f in (:*, :dot)
    @eval ($f)(a, ::NullOperator) = NullOperator()
    @eval ($f)(::NullOperator, b) = NullOperator()
    @eval ($f)(::NullOperator, ::NullOperator) = NullOperator()
    @eval ($f)(::AbstractLazy, ::NullOperator) = NullOperator()
    @eval ($f)(::NullOperator, ::AbstractLazy) = NullOperator()
end

# For binary `/`:
Base.:/(a, ::NullOperator) = error("Division by an AbstractNullOperator is not allowed.")
Base.:/(a::NullOperator, b) = a
Base.:/(::NullOperator, ::NullOperator) = NullOperator()

# For binary `^`:
Base.:^(a, ::NullOperator) = error("Undefined")
Base.:^(a::NullOperator, b) = a
Base.:^(::NullOperator, ::NullOperator) = NullOperator() # or "I" ?
Base.:^(a::AbstractLazy, ::NullOperator) = error("Undefined")
Base.:^(a::NullOperator, b::AbstractLazy) = a

###############################################################
# Define rules with `broadcasted`
###############################################################

function Base.broadcasted(f, a::AbstractLazy...)
    f_broacasted(x...) = broadcast(f, x...)
    LazyOperator(f_broacasted, a...)
end
Base.broadcasted(f, a::AbstractLazy, b) = Base.broadcasted(f, a, LazyWrap(b))
Base.broadcasted(f, a, b::AbstractLazy) = Base.broadcasted(f, LazyWrap(a), b)

###############################################################
# Define rules with composition
###############################################################
lazy_compose(a, b...) = lazy_compose(a, b)
lazy_compose(a, b::Tuple) = LazyOperator(lazy_compose, LazyWrap(a), LazyWrap(b...))

# trigger lazy composition for `f∘tuple(a...)` if in the tuple
# there is an `AbtractLazy` at first position,
Base.:∘(a::Function, b::AbstractLazy) = lazy_compose(a, (b,))
function Base.:∘(a::Function, b::Tuple{AbstractLazy, Vararg{Any}})
    lazy_compose(a, b)
end
function Base.:∘(a::Function, b::Tuple{Tuple{AbstractLazy, Vararg{Any}}, Vararg{Any}})
    lazy_compose(a, b)
end
