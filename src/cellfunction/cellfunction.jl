"""
    AbstractCellFunction{DS,S}

Abstract type to represent a function defined in specific domain `DS`,
which could be a `ReferenceDomain` or a `PhysicalDomain`. `S` is the size
of the codomain (i.e `S=length(f(x))`) where `f` is the `AbstractCellFunction`.

# Subtypes should implement :
* `get_function(f::AbstractCellFunction)`
"""
abstract type AbstractCellFunction{DS, S} <: AbstractLazy end

"""
    DomainStyle(f::AbstractCellFunction)
"""
DomainStyle(f::AbstractCellFunction{DS}) where {DS} = DS()

get_size(::AbstractCellFunction{DS, S}) where {DS, S} = S

"""
    get_function(f::AbstractCellFunction)
"""
function get_function(f::AbstractCellFunction)
    error("`get_function` is not defined for $(typeof(f))")
end

"""
    (f::AbstractCellFunction)(x)

Make `AbstractCellFunction` callable. Return `f(x)` by calling `evaluate(f, x)`.
"""
(f::AbstractCellFunction)(x) = evaluate(f, x)

"""
    evaluate(f::AbstractCellFunction, x)

Return the value `f(x)` by mapping `x` to the domain of `f` if necessary.
"""
evaluate(f::AbstractCellFunction, x) = evaluate(f, x, same_domain(f, x))

function evaluate(f::AbstractCellFunction, x, samedomain::Val{false})
    evaluate(f, change_domain(x, DomainStyle(f)))
end
function evaluate(f::AbstractCellFunction, x, samedomain::Val{true})
    get_function(f)(x)
end
function evaluate(f::AbstractCellFunction, x::CellPoint, samedomain::Val{true})
    evaluate_at_cellpoint(get_function(f), x)
end

"""
When a CellFunction is directly applied on a `FacePoint`, (i.e without the use of the
`side_n` or `side_p` operators), it means that the side of the face does matter (for
instance for a PhysicalFunction).

Hence we just convert the FacePoint into a CellPoint (of the negative side of the face)
before evaluating it
"""
evaluate(f::AbstractCellFunction, x::FacePoint) = evaluate(f, side_n(x))

function LazyOperators.pretty_name(a::AbstractCellFunction)
    "CellFunction{" * pretty_name(DomainStyle(a)) * "," * pretty_name(get_function(a)) * "}"
end
LazyOperators.pretty_name_style(::AbstractCellFunction) = Dict(:color => :light_green)

"""
Implement function `materialize` of the `AbstractLazy` interface.

TODO : define default behavior when things are stabilized :
`LazyOperators.materialize(f::AbstractCellFunction, x) = f`
"""
LazyOperators.materialize(f::AbstractCellFunction, x::CellInfo) = f
LazyOperators.materialize(f::AbstractCellFunction, x::CellPoint) = f(x)

"""
    abstract type AbstractShapeFunction{DS,S,FS} <: AbstractCellFunction{DS,S} end

Abstract type to represent a shape function defined in specific domain `DS`,
which could be a `ReferenceDomain` or a `PhysicalDomain`. `S` is the size
of the codomain (i.e `S=length(λ(x))`) and `FS` is the type of `FunctionSpace`.

# Interface
Subtypes should implement `AbstractCellFunction`:
* `get_function(f::AbstractShapeFunction)`

and its own specitic interface: [empty]
"""
abstract type AbstractShapeFunction{DS, S, FS} <: AbstractCellFunction{DS, S} end

get_function_space(::AbstractShapeFunction{DS, S, FS}) where {DS, S, FS} = FS()

"""
    ShapeFunction{DS,S,F<:Function} <: AbstractShapeFunction{DS,S}

Subtype of [`AbstractShapeFunction`](@ref) used to wrap a function
defined on a domain of style` `DS` in the cell
"""
struct ShapeFunction{DS, S, FS, F <: Function} <: AbstractShapeFunction{DS, S, FS}
    f::F
end

"""
    ShapeFunction(f::Function, domainstyle::DomainStyle, ::Val{S}) where S
"""
function ShapeFunction(f::Function, ds::DomainStyle, ::Val{S}, fs::FunctionSpace) where {S}
    ShapeFunction{typeof(ds), S, typeof(fs), typeof(f)}(f)
end

get_function(f::ShapeFunction) = f.f

function LazyOperators.pretty_name(a::ShapeFunction)
    "ShapeFunction{" *
    pretty_name(DomainStyle(a)) *
    "," *
    string(get_size(a)) *
    "," *
    pretty_name(get_function(a)) *
    "}"
end
LazyOperators.pretty_name_style(a::ShapeFunction) = Dict(:color => :light_green)

"""
    abstract type AbstractMultiShapeFunction{N,DS} end

Abstract type to represent a "set" of `ShapeFunction`.

`N` is the number of `ShapeFunction` contained in this `AbstractMultiShapeFunction`.

Note that all shape functions must have the same domain style `DS`.
"""
abstract type AbstractMultiShapeFunction{N, DS} end

struct MultiShapeFunction{N, DS, F <: Tuple{Vararg{AbstractShapeFunction, N}}} <:
       AbstractMultiShapeFunction{N, DS}
    shapeFunctions::F
end

"""
    MultiShapeFunction(f::NTuple{N,AbstractShapeFunction{DS}}) where {N,DS}
"""
function MultiShapeFunction(
    shapeFunctions::Tuple{Vararg{AbstractShapeFunction{DS}, N}},
) where {N, DS}
    MultiShapeFunction{N, DS, typeof(shapeFunctions)}(shapeFunctions)
end

get_shape_functions(f::MultiShapeFunction) = f.shapeFunctions

abstract type AbstractCellShapeFunctions{DS, S, FS, CS} <: AbstractShapeFunction{DS, S, FS} end

get_cell_shape(::AbstractCellShapeFunctions{DS, S, FS, CS}) where {DS, S, FS, CS} = CS()

"""
    CellShapeFunctions{DS,S,FS,CS} <: AbstractCellShapeFunctions{DS,S,FS,CS}

Subtype of [`AbstractCellShapeFunctions`](@ref) used to wrap a shape functions
defined on a domain of style `DS` in the cell.
The function `f` (of type `F`) must return all shape functions `λᵢ`
associated to the function space of type `FS` defined on cell shape `CS`, so that:
- λᵢ(x) = f(x)[i]
and:
- f(x) = (λ₁(x), …, λ₁(x), …, λₙ(x))
"""
struct CellShapeFunctions{DS, S, FS, CS} <: AbstractCellShapeFunctions{DS, S, FS, CS} end

"""
    CellShapeFunctions(domainstyle::DomainStyle, ::Val{S}, fs::FunctionSpace, shape::Shape) where S
"""
function CellShapeFunctions(
    ds::DomainStyle,
    ::Val{S},
    fs::FunctionSpace,
    shape::AbstractShape,
) where {S}
    CellShapeFunctions{typeof(ds), S, typeof(fs), typeof(shape)}()
end

function get_function(f::CellShapeFunctions)
    valS = Val(get_size(f))
    λ =
        _reshape_cell_shape_functions ∘
        shape_functions(get_function_space(f), valS, get_cell_shape(f))
end

@generated function _reshape_cell_shape_functions(a::SMatrix{Ndof, Ndim}) where {Ndof, Ndim}
    _exprs = [[:(a[$i, $j]) for j in 1:Ndim] for i in 1:Ndof]
    exprs = [:(SA[$(_expr...)]) for _expr in _exprs]
    return :(tuple($(exprs...)))
end
_reshape_cell_shape_functions(a::SVector) = Tuple(a)

function LazyOperators.pretty_name(a::CellShapeFunctions)
    "CellShapeFunctions{" *
    pretty_name(DomainStyle(a)) *
    "," *
    string(get_size(a)) *
    "," *
    pretty_name(get_function(a)) *
    "}"
end
LazyOperators.pretty_name_style(a::CellShapeFunctions) = Dict(:color => :light_green)

"""
    CellFunction{DS,S,F<:Function} <: AbstractCellFunction{DS,S}

Subtype of [`AbstractCellFunction`](@ref) used to wrap a function
defined on a domain of style` `DS` in the cell
"""
struct CellFunction{DS, S, F <: Function} <: AbstractCellFunction{DS, S}
    f::F
end

"""
    CellFunction(f::Function, domainstyle::DomainStyle, ::Val{S}) where {S}

`S` is the codomain size of `f`.
"""
function CellFunction(f::Function, ds::DomainStyle, ::Val{S}) where {S}
    CellFunction{typeof(ds), S, typeof(f)}(f)
end

get_function(f::CellFunction) = f.f

"""
    PhysicalFunction(f::Function, [size::Union{Integer, Tuple{Integer, Vararg{Integer}}} = 1])
    PhysicalFunction(f::Function, ::Val{size}) where {size}

Return a [`CellFunction`](@ref) defined on a `PhysicalDomain`.
`size` is the size of the codomain of `f`.

## Note:
Using a `Val` to prescribe the size a of `PhysicalFunction` is
recommended to improve type-stability and performance.
"""
function PhysicalFunction(
    f::Function,
    size::Union{Integer, Tuple{Integer, Vararg{Integer}}} = 1,
)
    CellFunction(f, PhysicalDomain(), Val(size))
end

function PhysicalFunction(f::Function, s::Val{size}) where {size}
    CellFunction(f, PhysicalDomain(), s)
end

"""
    ReferenceFunction(f::Function, [size::Union{Integer, Tuple{Integer, Vararg{Integer}}} = 1])
    ReferenceFunction(f::Function, ::Val{size}) where {size}

Return a [`CellFunction`](@ref) defined on a `ReferenceDomain`.
`size` is the size of the codomain of `f`.

## Note:
Using a `Val` to prescribe the size a of `ReferenceFunction` is
recommended to improve type-stability and performance.
"""
function ReferenceFunction(
    f::Function,
    size::Union{Integer, Tuple{Integer, Vararg{Integer}}} = 1,
)
    CellFunction(f, ReferenceDomain(), Val(size))
end

function ReferenceFunction(f::Function, s::Val{size}) where {size}
    CellFunction(f, ReferenceDomain(), s)
end

function LazyOperators.materialize(::AbstractCellFunction, ::FaceInfo)
    error("Cannot integrate a `CellFunction` on a face : please select a `Side`")
end

## TEMPORARY : should be moved in files where each function space is defined !
DomainStyle(fs::FunctionSpace{<:Lagrange}) = ReferenceDomain()
DomainStyle(fs::FunctionSpace{<:Taylor}) = ReferenceDomain()

#specific rules:
LazyOperators.materialize(a::LazyMapOver{<:AbstractCellShapeFunctions}, x::CellInfo) = a
function LazyOperators.materialize(
    a::LazyMapOver{<:AbstractCellShapeFunctions},
    x::CellPoint,
)
    f = get_args(a)
    MapOver(f(x))
end
function LazyOperators.lazy_map_over(
    f::Function,
    a::LazyMapOver{<:AbstractCellShapeFunctions},
)
    f(get_args(a))
end
function LazyOperators.materialize(a::LazyMapOver, x::CellPoint)
    MapOver(LazyOperators.lazy_map_over(Base.Fix2(materialize, x), a))
end
#LazyOperators.materialize(a::LazyMapOver{<:CellShapeFunctions}, x::CellPoint) = a

"""
Represent the side a of face between two cells.

@ghislainb : to be moved to `domain.jl` ?
"""
side_p(t::Tuple) = map(side_p, t)
side_n(t::Tuple) = map(side_n, t)
side_p(fInfo::FaceInfo) = get_cellinfo_p(fInfo)
side_n(fInfo::FaceInfo) = get_cellinfo_n(fInfo)
const side⁺ = side_p
const side⁻ = side_n

jump(a) = side⁻(a) - side⁺(a)
jump(a::Tuple) = map(jump, a)
jump(a, n) = side⁺(a) * side⁺(n) + side⁻(a) * side⁻(n)

abstract type AbstractSide{O, A <: Tuple} <: AbstractLazyOperator{O, A} end
function LazyOperators.get_operator(a::AbstractSide)
    error("`get_operator` is not defined for $(typeof(a))")
end
LazyOperators.get_args(side::AbstractSide) = side.args

struct Side⁺{O, A} <: AbstractSide{O, A}
    args::A
end
Side⁺(args...) = Side⁺{Nothing, typeof(args)}(args)
LazyOperators.get_operator(::Side⁺) = side_p

struct Side⁻{O, A} <: AbstractSide{O, A}
    args::A
end
Side⁻(args...) = Side⁻{Nothing, typeof(args)}(args)
LazyOperators.get_operator(::Side⁻) = side_n

function LazyOperators.materialize(side::AbstractSide, fInfo::FaceInfo)
    op_side = get_operator(side)
    return op_side(materialize_args(get_args(side), wrap_side(side, fInfo))...)
end

# function LazyOperators.materialize(side::AbstractSide, fInfo::FaceInfo)
#     op_side = get_operator(side)
#     return op_side(materialize_args(get_args(side), wrap_side(side, fInfo))...)
# end
wrap_side(::AbstractSide, a::AbstractSide) = a
wrap_side(::Side⁻, a::Side⁺) = Side⁻(get_args(a)...)
wrap_side(::Side⁺, a::Side⁻) = Side⁺(get_args(a)...)
wrap_side(::Side⁺, a::FaceInfo) = Side⁺(a)
wrap_side(::Side⁻, a::FaceInfo) = Side⁻(a)

function LazyOperators.materialize(a::AbstractLazy, sidefInfo::AbstractSide)
    op_side = get_operator(sidefInfo)
    return materialize(a, op_side(get_args(sidefInfo)...))
end

function LazyOperators.materialize(side::AbstractSide, fPoint::FacePoint)
    op_side = get_operator(side)
    cPoint = op_side(fPoint)
    return materialize(get_args(side)..., cPoint)
end
function materialize(a::LazyMapOver, x::AbstractSide{Nothing, <:Tuple{FaceInfo}})
    LazyMapOver(LazyOperators.lazy_map_over(Base.Fix2(materialize, x), a))
end
materialize(::NullOperator, ::AbstractSide) = NullOperator()

# function LazyOperators.materialize(
#     side::AbstractSide,
#     sidefInfo::AbstractSide{Nothing, <:Tuple{FacePoint}},
# )
#     op_side = get_operator(sidefInfo)
#     return materialize(get_args(side)..., op_side(get_args(sidefInfo)...))
# end

side_p(a::AbstractLazy) = Side⁺(a)
side_n(a::AbstractLazy) = Side⁻(a)
side_p(::NullOperator) = NullOperator()
side_n(::NullOperator) = NullOperator()

"""
Represent the face normal of a face
"""
struct FaceNormal <: AbstractLazy end

side_p(n) = Side⁺(n)
side_n(n) = Side⁻(n)

"""
At the `CellInfo` level, the `FaceNormal` doesn't really have a materialization and stays lazy
@ghislainb : I wonder if there is a way to avoid declaring this function ?
"""
LazyOperators.materialize(n::FaceNormal, ::CellInfo) = n

function LazyOperators.materialize(
    ::Side⁺{O, Tuple{FaceNormal}},
    fPoint::FacePoint,
) where {O}
    fInfo = get_faceinfo(fPoint)
    ξface = get_coord(fPoint)

    cInfo = get_cellinfo_p(fInfo)
    kside = get_cell_side_p(fInfo)
    cnodes = nodes(cInfo)
    ctype = celltype(cInfo)

    return normal(cnodes, ctype, kside, ξface)
end

function LazyOperators.materialize(
    ::Side⁻{O, Tuple{FaceNormal}},
    fPoint::FacePoint,
) where {O}
    fInfo = get_faceinfo(fPoint)
    ξface = get_coord(fPoint)

    cInfo = get_cellinfo_n(fInfo)
    kside = get_cell_side_n(fInfo)
    cnodes = nodes(cInfo)
    ctype = celltype(cInfo)

    return normal(cnodes, ctype, kside, ξface)
end

LazyOperators.materialize(f::Function, ::CellInfo) = f
LazyOperators.materialize(f::Function, ::CellPoint) = f
LazyOperators.materialize(f::Function, ::FaceInfo) = f
LazyOperators.materialize(f::Function, ::FacePoint) = f

LazyOperators.materialize(a::AbstractArray{<:Number}, ::CellInfo) = a
LazyOperators.materialize(a::AbstractArray{<:Number}, ::FaceInfo) = a
LazyOperators.materialize(a::AbstractArray{<:Number}, ::CellPoint) = a
LazyOperators.materialize(a::AbstractArray{<:Number}, ::FacePoint) = a
LazyOperators.materialize(a::Number, ::CellInfo) = a
LazyOperators.materialize(a::Number, ::FaceInfo) = a
LazyOperators.materialize(a::Number, ::CellPoint) = a
LazyOperators.materialize(a::Number, ::FacePoint) = a

LazyOperators.materialize(a::LinearAlgebra.UniformScaling, ::CellInfo) = a
LazyOperators.materialize(a::LinearAlgebra.UniformScaling, ::FaceInfo) = a
LazyOperators.materialize(a::LinearAlgebra.UniformScaling, ::CellPoint) = a
LazyOperators.materialize(a::LinearAlgebra.UniformScaling, ::FacePoint) = a