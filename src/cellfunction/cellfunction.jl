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
wrap_side(::Side⁻, a::Side⁺) = error("invalid : cannot `Side⁺` with `Side⁻`")
wrap_side(::Side⁺, a::Side⁻) = error("invalid : cannot `Side⁻` with `Side⁺`")
wrap_side(::Side⁺, a) = Side⁺(a)
wrap_side(::Side⁻, a) = Side⁻(a)
wrap_side(::Side⁻, a::Side⁻) = a
wrap_side(::Side⁺, a::Side⁺) = a

function LazyOperators.materialize(a::AbstractCellFunction, sidefInfo::AbstractSide)
    op_side = get_operator(sidefInfo)
    return materialize(a, op_side(get_args(sidefInfo)...))
end

function materialize(a::LazyMapOver, x::AbstractSide{Nothing, <:Tuple{FaceInfo}})
    LazyMapOver(LazyOperators.lazy_map_over(Base.Fix2(materialize, x), a))
end
function materialize(a::LazyMapOver, x::AbstractSide{Nothing, <:Tuple{FacePoint}})
    MapOver(LazyOperators.lazy_map_over(Base.Fix2(materialize, x), a))
end
materialize(::NullOperator, ::AbstractSide) = NullOperator()

function LazyOperators.materialize(side::AbstractSide, x)
    a = materialize_args(get_args(side), wrap_side(side, x))
    return LazyOperators.may_unwrap_tuple(a)
end

side_p(a::AbstractLazy) = Side⁺(a)
side_n(a::AbstractLazy) = Side⁻(a)
side_p(a::Side⁺) = a
side_p(a::Side⁻) = error("invalid : cannot apply `side_p` on `Side⁻`")
side_n(a::Side⁺) = error("invalid : cannot apply `side_n` on `Side⁺`")
side_n(a::Side⁻) = a
side_p(::NullOperator) = NullOperator()
side_n(::NullOperator) = NullOperator()

side_p(n) = Side⁺(n)
side_n(n) = Side⁻(n)

"""
Represent the face normal of a face
"""
struct FaceNormal <: AbstractLazy end

"""
At the `CellInfo` level, the `FaceNormal` doesn't really have a materialization and stays lazy
@ghislainb : I wonder if there is a way to avoid declaring this function ?
"""
LazyOperators.materialize(n::FaceNormal, ::CellInfo) = n

function LazyOperators.materialize(
    n::FaceNormal,
    ::AbstractSide{Nothing, <:Tuple{<:FaceInfo}},
)
    n
end

function LazyOperators.materialize(
    ::FaceNormal,
    sideFacePoint::Side⁺{Nothing, <:Tuple{<:FacePoint}},
)
    fPoint, = get_args(sideFacePoint)
    fInfo = get_faceinfo(fPoint)
    ξface = get_coords(fPoint)

    cInfo = get_cellinfo_p(fInfo)
    kside = get_cell_side_p(fInfo)
    cnodes = nodes(cInfo)
    ctype = celltype(cInfo)

    return normal(ctype, cnodes, kside, ξface)
end

function LazyOperators.materialize(
    ::FaceNormal,
    sideFacePoint::Side⁻{Nothing, <:Tuple{<:FacePoint}},
)
    fPoint, = get_args(sideFacePoint)
    fInfo = get_faceinfo(fPoint)
    ξface = get_coords(fPoint)

    cInfo = get_cellinfo_n(fInfo)
    kside = get_cell_side_n(fInfo)
    cnodes = nodes(cInfo)
    ctype = celltype(cInfo)

    return normal(ctype, cnodes, kside, ξface)
end

"""
Represent the tangential projector associated to an hypersurface. Its expression is
```math
    P = I - \\nu \\otimes \\nu
```
where ``\\nu`` is the cell normal vector.
"""
struct TangentialProjector <: AbstractLazy end

LazyOperators.materialize(P::TangentialProjector, ::CellInfo) = P

function LazyOperators.materialize(
    ::TangentialProjector,
    cPoint::CellPoint{ReferenceDomain},
)
    cnodes = get_cellnodes(cPoint)
    ctype = get_celltype(cPoint)
    ξ = get_coords(cPoint)
    return _tangential_projector(cnodes, ctype, ξ)
end

function LazyOperators.materialize(
    P::TangentialProjector,
    ::AbstractSide{Nothing, <:Tuple{<:FaceInfo}},
)
    P
end

function LazyOperators.materialize(
    ::TangentialProjector,
    sideFacePoint::Side⁻{Nothing, <:Tuple{<:FacePoint}},
)
    fPoint, = get_args(sideFacePoint)
    fInfo = get_faceinfo(fPoint)
    cInfo = get_cellinfo_n(fInfo)
    cnodes = nodes(cInfo)
    ctype = celltype(cInfo)
    ξcell = get_coords(side_n(fPoint))

    return _tangential_projector(cnodes, ctype, ξcell)
end

function LazyOperators.materialize(
    ::TangentialProjector,
    sideFacePoint::Side⁺{Nothing, <:Tuple{<:FacePoint}},
)
    fPoint, = get_args(sideFacePoint)
    fInfo = get_faceinfo(fPoint)
    cInfo = get_cellinfo_p(fInfo)
    cnodes = nodes(cInfo)
    ctype = celltype(cInfo)
    ξcell = get_coords(side_p(fPoint))

    return _tangential_projector(cnodes, ctype, ξcell)
end

"""
Warning : `cell_normal` is a "cell" operator, not a "face" operator.
So ξ is expected to be in the reference domain of the cell.
"""
function _tangential_projector(cnodes, ctype, ξ)
    ν = cell_normal(ctype, cnodes, ξ)
    return I - (ν ⊗ ν)
end

"""
Hypersurface "face" operator that rotates around the face-axis to virtually
bring back the two adjacent cells in the same plane.
"""
struct CoplanarRotation <: AbstractLazy end

function LazyOperators.materialize(
    R::CoplanarRotation,
    ::AbstractSide{Nothing, <:Tuple{<:FaceInfo}},
)
    R
end

function _unpack_face_point(sideFacePoint)
    fPoint, = get_args(sideFacePoint)
    fInfo = get_faceinfo(fPoint)

    cInfo_n = get_cellinfo_n(fInfo)
    cnodes_n = nodes(cInfo_n)
    ctype_n = celltype(cInfo_n)
    ξ_n = get_coords(side_n(fPoint))

    cInfo_p = get_cellinfo_p(fInfo)
    cnodes_p = nodes(cInfo_p)
    ctype_p = celltype(cInfo_p)
    ξ_p = get_coords(side_p(fPoint))

    return cnodes_n, cnodes_p, ctype_n, ctype_p, ξ_n, ξ_p
end

function LazyOperators.materialize(
    ::CoplanarRotation,
    sideFacePoint::Side⁻{Nothing, <:Tuple{<:FacePoint}},
)
    cnodes_n, cnodes_p, ctype_n, ctype_p, ξ_n, ξ_p = _unpack_face_point(sideFacePoint)
    return _coplanar_rotation(cnodes_n, cnodes_p, ctype_n, ctype_p, ξ_n, ξ_p)
end

function LazyOperators.materialize(
    ::CoplanarRotation,
    sideFacePoint::Side⁺{Nothing, <:Tuple{<:FacePoint}},
)
    cnodes_n, cnodes_p, ctype_n, ctype_p, ξ_n, ξ_p = _unpack_face_point(sideFacePoint)
    return _coplanar_rotation(cnodes_p, cnodes_n, ctype_p, ctype_n, ξ_p, ξ_n)
end

function _coplanar_rotation(
    cnodes_n::AbstractArray{Node{2, T}, N},
    cnodes_p,
    ctype_n::AbstractEntityType{1},
    ctype_p,
    ξ_n,
    ξ_p,
) where {T, N}

    # We choose to use `cell_normal` instead of `normal`,
    # but we could do the same with the latter.
    ν_n = cell_normal(ctype_n, cnodes_n, ξ_n)
    ν_p = cell_normal(ctype_p, cnodes_p, ξ_p)

    _cos = ν_p ⋅ ν_n
    _sin = ν_p[1] * ν_n[2] - ν_p[2] * ν_n[1] # 2D-vector cross-product

    _R = SA[_cos (-_sin); _sin _cos]

    return _R
end

function _coplanar_rotation(
    cnodes_n::AbstractArray{Node{3, T}, N},
    cnodes_p,
    ctype_n::AbstractEntityType{2},
    ctype_p,
    ξ_n,
    ξ_p,
) where {T, N}

    # We choose to use `cell_normal` instead of `normal`,
    # but we could do the same with the latter.
    ν_n = cell_normal(ctype_n, cnodes_n, ξ_n)
    ν_p = cell_normal(ctype_p, cnodes_p, ξ_p)

    _cos = ν_p ⋅ ν_n
    _sin = cross(ν_p, ν_n) # this is not really a 'sinus', it is (sinus x u)

    # u = _sin / ||_sin||
    # So we must ensure that `_sin` is not 0, otherwise we return the null vector
    norm_sin = norm(_sin)
    u = norm_sin > eps(norm_sin) ? _sin ./ norm_sin : _sin

    _R = _cos * I + _cross_product_matrix(_sin) + (1 - _cos) * (u ⊗ u)

    return _R
end

"""
The cross product between two vectors 'a' and 'b', a × b, can be expressed
as a matrix-vector product between the "cross-product-matrix" of 'a' and the
vector 'b'.
"""
function _cross_product_matrix(a)
    @assert size(a) == (3,)
    SA[
        0 (-a[3]) a[2]
        a[3] 0 (-a[1])
        (-a[2]) a[1] 0
    ]
end

"""
Normal of a facet of a hypersurface.

See [`cell_normal`]@ref for the computation method.
"""
struct CellNormal <: AbstractLazy
    function CellNormal(mesh::AbstractMesh)
        @assert topodim(mesh) < spacedim(mesh) "CellNormal has only sense when dealing with hypersurface, maybe you confused it with FaceNormal?"
        return new()
    end
end

LazyOperators.materialize(ν::CellNormal, ::CellInfo) = ν

function LazyOperators.materialize(::CellNormal, cPoint::CellPoint{ReferenceDomain})
    ctype = get_celltype(cPoint)
    cnodes = get_cellnodes(cPoint)
    ξ = get_coords(cPoint)
    return cell_normal(ctype, cnodes, ξ)
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

LazyOperators.materialize(f::Function, ::AbstractLazy) = f
LazyOperators.materialize(a::AbstractArray{<:Number}, ::AbstractLazy) = a
LazyOperators.materialize(a::Number, ::AbstractLazy) = a
LazyOperators.materialize(a::LinearAlgebra.UniformScaling, ::AbstractLazy) = a
