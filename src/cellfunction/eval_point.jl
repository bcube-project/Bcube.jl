abstract type DomainStyle end

LazyOperators.pretty_name(a::DomainStyle) = string(typeof(a))

"""
    ReferenceDomain()

Subtype of [`DomainStyle`](@ref) used to describe function that are
defined on reference shape of the corresponding cell.
"""
struct ReferenceDomain <: DomainStyle end

"""
    PhysicalDomain()

Subtype of [`DomainStyle`](@ref) used to describe function that are
defined on the physical cell.
"""
struct PhysicalDomain <: DomainStyle end

"""
    DomainStyle(a)

Return the domain style of `a` (reference or local)
"""
function DomainStyle(a)
    error("`DomainStyle` is not defined for $(typeof(a))")
end

"""
    change_domain(a, target_domain::DomainStyle)
"""
function change_domain(a, target_domain::DomainStyle)
    change_domain(a, DomainStyle(a), target_domain)
end
change_domain(a, input_domain::T, target_domain::T) where {T <: DomainStyle} = a
function change_domain(a, input_domain::DomainStyle, target_domain::DomainStyle)
    error("`change_domain` is not defined for $(typeof(f))")
end

"""
    same_domain(a, b)

Return `Val(true)` if `a` and `b` have the same `DomainStyle`.
Return `Val(false)` otherwise.
"""
same_domain(a, b) = same_domain(DomainStyle(a), DomainStyle(b))
same_domain(a::DS, b::DS) where {DS <: DomainStyle} = Val(true)
same_domain(a::DomainStyle, b::DomainStyle) = Val(false)

"""
    common_target_domain(a, b)

Return a commom target `DomainStyle` for `a` and `b`.
"""
common_target_domain(a, b) = common_target_domain(DomainStyle(a), DomainStyle(b))
common_target_domain(a::DS, b::DS) where {DS <: DomainStyle} = DS()
common_target_domain(a, b, c...) = common_target_domain(common_target_domain(a, b), c...)
# make `PhysicalDomain` "wins" by default :
common_target_domain(a::ReferenceDomain, b::PhysicalDomain) = PhysicalDomain()
common_target_domain(a::PhysicalDomain, b::ReferenceDomain) = PhysicalDomain()

"""
    AbstractCellPoint{DS}

Abstract type to represent a point defined in a cell.

# Subtypes should implement :
* `get_coord(p::AbstractCellPoint)`
* `change_domain(p::AbstractCellPoint, ds::DomainStyle)`
"""
abstract type AbstractCellPoint{DS} end

DomainStyle(p::AbstractCellPoint{DS}) where {DS} = DS()

function get_coord(p::AbstractCellPoint)
    error("`get_coord` is not defined for $(typeof(p))")
end

change_domain(p::AbstractCellPoint{DS}, ::DS) where {DS <: DomainStyle} = p
function change_domain(p::AbstractCellPoint, ds::DomainStyle)
    error("`change_domain` is not defined for $(typeof(p)) and $(typeof(ds))")
end

struct CellPoint{DS, T, C} <: AbstractCellPoint{DS}
    x::T
    cellinfo::C
end

"""
    CellPoint(x, c::CellInfo, ds::DomainStyle)

Subtype of [`AbstractCellPoint`](@ref) used to defined of point in a cell.
An `AbstractCellFunction` can be easily and efficiently evaluated at a `CellPoint`.

`x` can be a tuple or an array of several coordinates of points.
"""
function CellPoint(x::T, c::C, ds::DS) where {T, C <: CellInfo, DS <: DomainStyle}
    CellPoint{DS, T, C}(x, c)
end

get_coord(p::CellPoint) = p.x
get_cellinfo(p::CellPoint) = p.cellinfo
get_cellnodes(p::CellPoint) = nodes(get_cellinfo(p))
get_celltype(p::CellPoint) = celltype(get_cellinfo(p))

function change_domain(p::CellPoint{ReferenceDomain}, target_domain::PhysicalDomain)
    m(x) = mapping(celltype(p.cellinfo), nodes(p.cellinfo), x)
    x_phy = _apply_mapping(m, get_coord(p))
    CellPoint(x_phy, p.cellinfo, target_domain)
end

function change_domain(p::CellPoint{PhysicalDomain}, target_domain::ReferenceDomain)
    m(x) = mapping_inv(nodes(p.cellinfo), celltype(p.cellinfo), x)
    x_ref = _apply_mapping(m, get_coord(p))
    CellPoint(x_ref, p.cellinfo, target_domain)
end

"""
Apply mapping function `f` on a coordinates of a point or on a tuple/array of
several coordinates `x`.
"""
_apply_mapping(f, x) = f(x)
_apply_mapping(f, x::AbstractArray{<:AbstractArray}) = map(f, x)
_apply_mapping(f, x::Tuple{Vararg{AbstractArray}}) = map(f, x)

evaluate_at_cellpoint(f::Function, x::CellPoint) = _evaluate_at_cellpoint(f, get_coord(x))
_evaluate_at_cellpoint(f, x) = f(x)
_evaluate_at_cellpoint(f, x::AbstractArray{<:AbstractArray}) = map(f, x)
_evaluate_at_cellpoint(f, x::Tuple{Vararg{AbstractArray}}) = map(f, x)

"""
A `FacePoint` represent a point on a face. A face is a interface between two cells (except on the boundary).

A `FacePoint` is a `CellPoint` is the sense that its coordinates can always be expressed in the reference
coordinate of one of its adjacent cells.
"""
struct FacePoint{DS, T, F} <: AbstractCellPoint{DS}
    x::T
    faceInfo::F
end
get_coord(facePoint::FacePoint) = facePoint.x
get_faceinfo(facePoint::FacePoint) = facePoint.faceInfo

""" Constructor """
function FacePoint(x, faceInfo::FaceInfo, ds::DomainStyle)
    FacePoint{typeof(ds), typeof(x), typeof(faceInfo)}(x, faceInfo)
end

"""
Return the `CellPoint` corresponding to the `FacePoint` on negative side
"""
function side_n(facePoint::FacePoint{ReferenceDomain})
    faceInfo = get_faceinfo(facePoint)
    cellinfo_n = get_cellinfo_n(faceInfo)
    f = mapping_face(shape(celltype(cellinfo_n)), get_cell_side_n(faceInfo))
    return CellPoint(f(get_coord(facePoint)), cellinfo_n, ReferenceDomain())
end

"""
Return the `CellPoint` corresponding to the `FacePoint` on positive side
"""
function side_p(facePoint::FacePoint{ReferenceDomain})
    faceInfo = get_faceinfo(facePoint)
    cellinfo_n = get_cellinfo_n(faceInfo)
    cellinfo_p = get_cellinfo_p(faceInfo)

    # get global indices of nodes on the face from cell of side_n
    cshape_n = shape(celltype(cellinfo_n))
    cside_n = get_cell_side_n(faceInfo)
    c2n_n = get_nodes_index(cellinfo_n)
    f2n_n = c2n_n[faces2nodes(cshape_n, cside_n)] # assumption:  shape nodes can be directly indexed in entity nodes

    ## @bmxam
    ## NOTE : f2n_n == get_nodes_index(faceInfo) is not true in general

    # get global indices of nodes on the face from cell of side⁺
    cshape_p = shape(celltype(cellinfo_p))
    cside_p = get_cell_side_p(faceInfo)
    c2n_p = get_nodes_index(cellinfo_p)
    f2n_p = c2n_p[faces2nodes(cshape_p, cside_p)]  # assumption:  shape nodes can be directly indexed in entity nodes

    # types-stable version of `indexin(f2n_p, f2n_n)`
    # which return an array of `Int` instead of `Union{Int,Nothing}``
    permut = map(f2n_p) do j
        for (i, k) in enumerate(f2n_n)
            k === j && return i
        end
        return -1 # this must never happen
    end

    f = mapping_face(cshape_p, cside_p, permut)

    return CellPoint(f(get_coord(facePoint)), cellinfo_p, ReferenceDomain())
end

function side_n(facePoint::FacePoint{PhysicalDomain})
    return CellPoint(
        get_coord(facePoint),
        get_cellinfo_n(get_faceinfo(facePoint)),
        PhysicalDomain(),
    )
end
function side_p(facePoint::FacePoint{PhysicalDomain})
    return CellPoint(
        get_coord(facePoint),
        get_cellinfo_p(get_faceinfo(facePoint)),
        PhysicalDomain(),
    )
end

""" @ghislainb : I feel like you could write only on common function for both CellPoint and FacePoint """
function change_domain(p::FacePoint{ReferenceDomain}, ::PhysicalDomain)
    faceInfo = get_faceinfo(p)
    m(x) = mapping(facetype(faceInfo), nodes(faceInfo), x)
    x_phy = _apply_mapping(m, get_coord(p))
    FacePoint(x_phy, faceInfo, PhysicalDomain())
end
