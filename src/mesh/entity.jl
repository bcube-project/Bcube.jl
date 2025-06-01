#
# NOTE : entity are defined according CGNS conventions. Beware that
# for some elements, this is different from the GMSH convention (see Hexa27
# for instance).
# (see https://cgns.github.io/CGNS_docs_current/sids/conv.html)

abstract type AbstractEntity{dim} end
abstract type AbstractNode <: AbstractEntity{0} end
abstract type AbstractFace{dim} <: AbstractEntity{dim} end
abstract type AbstractCell{dim} <: AbstractEntity{dim} end

abstract type AbstractCellType end
abstract type AbstractFaceType end

abstract type AbstractEntityType{dim} end
struct Node_t <: AbstractEntityType{0} end
struct Bar2_t <: AbstractEntityType{1} end
struct Bar3_t <: AbstractEntityType{1} end
struct Bar4_t <: AbstractEntityType{1} end
struct Bar5_t <: AbstractEntityType{1} end
struct Tri3_t <: AbstractEntityType{2} end
struct Tri6_t <: AbstractEntityType{2} end
struct Tri9_t <: AbstractEntityType{2} end
struct Tri10_t <: AbstractEntityType{2} end
struct Tri12_t <: AbstractEntityType{2} end
struct Quad4_t <: AbstractEntityType{2} end
struct Quad8_t <: AbstractEntityType{2} end
struct Quad9_t <: AbstractEntityType{2} end
struct Quad16_t <: AbstractEntityType{2} end
struct Tetra4_t <: AbstractEntityType{3} end
struct Tetra10_t <: AbstractEntityType{3} end
struct Hexa8_t <: AbstractEntityType{3} end
struct Hexa27_t <: AbstractEntityType{3} end
struct Penta6_t <: AbstractEntityType{3} end
struct Pyra5_t <: AbstractEntityType{3} end
struct Poly2_t{N, M} <: AbstractEntityType{2} end
struct Poly3_t{N, M} <: AbstractEntityType{3} end

abstract type AbstractEntityList end
const EntityVector{T} = Vector{T} where {T <: AbstractEntityType}

#---- Generic functions which MUST BE implemented for each concrete type ----
@inline function nnodes(::Type{<:T}) where {T <: AbstractEntityType}
    error("Function ‘nnodes‘ is not defined")
end
@inline nnodes(a::T) where {T <: AbstractEntityType} = nnodes(typeof(a))
@inline function nodes(::Type{<:T}) where {T <: AbstractEntityType}
    error("Function ‘nodes‘ is not defined")
end
@inline nodes(a::T) where {T <: AbstractEntityType} = nodes(typeof(a))
@inline function nedges(::Type{<:T}) where {T <: AbstractEntityType}
    error("Function nedges is not defined")
end
@inline nedges(a::T) where {T <: AbstractEntityType} = nedges(typeof(a))
@inline function edges2nodes(::Type{<:T}) where {T <: AbstractEntityType}
    error("Function edges2nodes is not defined")
end
@inline edges2nodes(a::T) where {T <: AbstractEntityType} = edges2nodes(typeof(a))
@inline function edgetypes(::Type{<:T}) where {T <: AbstractEntityType}
    error("Function edgetypes is not defined")
end
@inline edgetypes(a::T) where {T <: AbstractEntityType} = edgetypes(typeof(a))
@inline edgetypes(a::T, i) where {T <: AbstractEntityType} = edgetypes(a)[i]
@inline function nfaces(::Type{<:T}) where {T <: AbstractEntityType}
    error("Function nfaces is not defined")
end
@inline nfaces(a::T) where {T <: AbstractEntityType} = nfaces(typeof(a))
@inline function faces2nodes(::Type{<:T}) where {T <: AbstractEntityType}
    error("Function faces2nodes is not defined")
end
@inline faces2nodes(a::T) where {T <: AbstractEntityType} = faces2nodes(typeof(a))
@inline topodim(::Type{<:AbstractEntityType{N}}) where {N} = N
@inline topodim(a::T) where {T <: AbstractEntityType} = topodim(typeof(a))
@inline function facetypes(::Type{<:T}) where {T <: AbstractEntityType}
    error("Function facetypes is not defined")
end
@inline facetypes(a::T) where {T <: AbstractEntityType} = facetypes(typeof(a))
@inline facetypes(a::T, i) where {T <: AbstractEntityType} = facetypes(a)[i]

#---- Valid generic functions for each type ----
@inline function edges2nodes(a::T, i) where {T <: AbstractEntityType}
    nedges(a) === 0 ? nothing : edges2nodes(a)[i]
end
@inline function edges2nodes(a::T, i...) where {T <: AbstractEntityType}
    nedges(a) === 0 ? nothing : edges2nodes(a)[i...]
end
@inline function faces2nodes(a::T, i) where {T <: AbstractEntityType}
    nfaces(a) === 0 ? nothing : faces2nodes(a)[i]
end
@inline function faces2nodes(a::T, i...) where {T <: AbstractEntityType}
    nfaces(a) === 0 ? nothing : faces2nodes(a)[i]
end

#---- Type specific functions ----

#- Node
@inline nnodes(::Type{Node_t}) = 1
@inline nodes(::Type{Node_t}) = (1,)
@inline nedges(::Type{Node_t}) = 0
@inline edges2nodes(::Type{Node_t}) = nothing
@inline edgetypes(::Type{Node_t}) = nothing
@inline nfaces(::Type{Node_t}) = 0
@inline faces2nodes(::Type{Node_t}) = nothing
@inline facetypes(::Type{Node_t}) = nothing

#- Bar2
@inline nnodes(::Type{Bar2_t}) = 2
@inline nodes(::Type{Bar2_t}) = (1, 2)
@inline nedges(::Type{Bar2_t}) = 2
@inline edges2nodes(::Type{Bar2_t}) = ((1,), (2,))
@inline edgetypes(::Type{Bar2_t}) = (Node_t(), Node_t())
@inline nfaces(t::Type{Bar2_t}) = nedges(t)
@inline faces2nodes(t::Type{Bar2_t}) = edges2nodes(t)
@inline facetypes(t::Type{Bar2_t}) = edgetypes(t)

#- Bar3
#  N1        N3       N2
#  x---------x---------x
@inline nnodes(::Type{Bar3_t}) = 3
@inline nodes(::Type{Bar3_t}) = (1, 2, 3)
@inline nedges(::Type{Bar3_t}) = 2
@inline edges2nodes(::Type{Bar3_t}) = ((1,), (2,))
@inline edgetypes(::Type{Bar3_t}) = (Node_t(), Node_t())
@inline nfaces(t::Type{Bar3_t}) = nedges(t)
@inline faces2nodes(t::Type{Bar3_t}) = edges2nodes(t)
@inline facetypes(t::Type{Bar3_t}) = edgetypes(t)

#- Bar4
#  N1    N3      N4     N2
#  x------x------x------x
@inline nnodes(::Type{Bar4_t}) = 4
@inline nodes(::Type{Bar4_t}) = (1, 2, 3, 4)
@inline nedges(::Type{Bar4_t}) = 2
@inline edges2nodes(::Type{Bar4_t}) = ((1,), (2,))
@inline edgetypes(::Type{Bar4_t}) = (Node_t(), Node_t())
@inline nfaces(t::Type{Bar4_t}) = nedges(t)
@inline faces2nodes(t::Type{Bar4_t}) = edges2nodes(t)
@inline facetypes(t::Type{Bar4_t}) = edgetypes(t)

#- Bar5
#  N1   N3   N4   N5   N2
#  x----x----x----x----x
@inline nnodes(::Type{Bar5_t}) = 5
@inline nodes(::Type{Bar5_t}) = (1, 2, 3, 4, 5)
@inline nedges(::Type{Bar5_t}) = 2
@inline edges2nodes(::Type{Bar5_t}) = ((1,), (2,))
@inline edgetypes(::Type{Bar5_t}) = (Node_t(), Node_t())
@inline nfaces(t::Type{Bar5_t}) = nedges(t)
@inline faces2nodes(t::Type{Bar5_t}) = edges2nodes(t)
@inline facetypes(t::Type{Bar5_t}) = edgetypes(t)

#-Tri3 ----
@inline nnodes(::Type{Tri3_t}) = 3
@inline nodes(::Type{Tri3_t}) = (1, 2, 3)
@inline nedges(::Type{Tri3_t}) = 3
@inline edges2nodes(::Type{Tri3_t}) = ((1, 2), (2, 3), (3, 1))
@inline edgetypes(::Type{Tri3_t}) = (Bar2_t(), Bar2_t(), Bar2_t())
@inline nfaces(t::Type{Tri3_t}) = nedges(t)
@inline faces2nodes(t::Type{Tri3_t}) = edges2nodes(t)
@inline facetypes(t::Type{Tri3_t}) = edgetypes(t)

#-Tri6 ----
@inline nnodes(::Type{Tri6_t}) = 6
@inline nodes(::Type{Tri6_t}) = (1, 2, 3, 4, 5, 6)
@inline nedges(::Type{Tri6_t}) = 3
@inline edges2nodes(::Type{Tri6_t}) = ((1, 2, 4), (2, 3, 5), (3, 1, 6))
@inline edgetypes(::Type{Tri6_t}) = (Bar3_t(), Bar3_t(), Bar3_t())
@inline nfaces(t::Type{Tri6_t}) = nedges(t)
@inline faces2nodes(t::Type{Tri6_t}) = edges2nodes(t)
@inline facetypes(t::Type{Tri6_t}) = edgetypes(t)

#-Tri9 ----
@inline nnodes(::Type{Tri9_t}) = 9
@inline nodes(::Type{Tri9_t}) = (1, 2, 3, 4, 5, 6, 7, 8, 9)
@inline nedges(::Type{Tri9_t}) = 3
@inline edges2nodes(::Type{Tri9_t}) = ((1, 2, 4, 5), (2, 3, 6, 7), (3, 1, 8, 9))
@inline edgetypes(::Type{Tri9_t}) = (Bar4_t(), Bar4_t(), Bar4_t())
@inline nfaces(t::Type{Tri9_t}) = nedges(t)
@inline faces2nodes(t::Type{Tri9_t}) = edges2nodes(t)
@inline facetypes(t::Type{Tri9_t}) = edgetypes(t)

#-Tri10 ----
@inline nnodes(::Type{Tri10_t}) = 10
@inline nodes(::Type{Tri10_t}) = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
@inline nedges(::Type{Tri10_t}) = 3
@inline edges2nodes(::Type{Tri10_t}) = ((1, 2, 4, 5), (2, 3, 6, 7), (3, 1, 8, 9))
@inline edgetypes(::Type{Tri10_t}) = (Bar4_t(), Bar4_t(), Bar4_t())
@inline nfaces(t::Type{Tri10_t}) = nedges(t)
@inline faces2nodes(t::Type{Tri10_t}) = edges2nodes(t)
@inline facetypes(t::Type{Tri10_t}) = edgetypes(t)

#-Tri12----
@inline nnodes(::Type{Tri12_t}) = 12
@inline nodes(::Type{Tri12_t}) = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
@inline nedges(::Type{Tri12_t}) = 3
@inline function edges2nodes(::Type{Tri12_t})
    ((1, 2, 4, 5, 6), (2, 3, 7, 8, 9), (3, 1, 10, 11, 12))
end
@inline edgetypes(::Type{Tri12_t}) = (Bar5_t(), Bar5_t(), Bar5_t())
@inline nfaces(t::Type{Tri12_t}) = nedges(t)
@inline faces2nodes(t::Type{Tri12_t}) = edges2nodes(t)
@inline facetypes(t::Type{Tri12_t}) = edgetypes(t)

#---- Quad4 ----
@inline nnodes(::Type{Quad4_t}) = 4
@inline nodes(::Type{Quad4_t}) = (1, 2, 3, 4)
@inline nedges(::Type{Quad4_t}) = 4
@inline edges2nodes(::Type{Quad4_t}) = ((1, 2), (2, 3), (3, 4), (4, 1))
@inline edgetypes(::Type{Quad4_t}) = (Bar2_t(), Bar2_t(), Bar2_t(), Bar2_t())
@inline nfaces(t::Type{Quad4_t}) = nedges(t)
@inline faces2nodes(t::Type{Quad4_t}) = edges2nodes(t)
@inline facetypes(t::Type{Quad4_t}) = edgetypes(t)

#---- Quad8 ----
@inline nnodes(::Type{Quad8_t}) = 8
@inline nodes(::Type{Quad8_t}) = (1, 2, 3, 4, 5, 6, 7, 8)
@inline nedges(::Type{Quad8_t}) = 4
@inline edges2nodes(::Type{Quad8_t}) = ((1, 2, 5), (2, 3, 6), (3, 4, 7), (4, 1, 8))
@inline edgetypes(::Type{Quad8_t}) = (Bar3_t(), Bar3_t(), Bar3_t(), Bar3_t())
@inline nfaces(t::Type{Quad8_t}) = nedges(t)
@inline faces2nodes(t::Type{Quad8_t}) = edges2nodes(t)
@inline facetypes(t::Type{Quad8_t}) = edgetypes(t)

#---- Quad9 ----
@inline nnodes(::Type{Quad9_t}) = 9
@inline nodes(::Type{Quad9_t}) = (1, 2, 3, 4, 5, 6, 7, 8, 9)
@inline nedges(::Type{Quad9_t}) = 4
@inline edges2nodes(::Type{Quad9_t}) = ((1, 2, 5), (2, 3, 6), (3, 4, 7), (4, 1, 8))
@inline edgetypes(::Type{Quad9_t}) = (Bar3_t(), Bar3_t(), Bar3_t(), Bar3_t())
@inline nfaces(t::Type{Quad9_t}) = nedges(t)
@inline faces2nodes(t::Type{Quad9_t}) = edges2nodes(t)
@inline facetypes(t::Type{Quad9_t}) = edgetypes(t)

#---- Quad16 ----
@inline nnodes(::Type{Quad16_t}) = 16
@inline nodes(::Type{Quad16_t}) = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
@inline nedges(::Type{Quad16_t}) = 4
@inline function edges2nodes(::Type{Quad16_t})
    ((1, 2, 5, 6), (2, 3, 7, 8), (3, 4, 9, 10), (4, 1, 11, 12))
end
@inline edgetypes(::Type{Quad16_t}) = (Bar4_t(), Bar4_t(), Bar4_t(), Bar4_t())
@inline nfaces(t::Type{Quad16_t}) = nedges(t)
@inline faces2nodes(t::Type{Quad16_t}) = edges2nodes(t)
@inline facetypes(t::Type{Quad16_t}) = edgetypes(t)

#---- Tetra4 ----
@inline nnodes(::Type{Tetra4_t}) = 4
@inline nodes(::Type{Tetra4_t}) = (1, 2, 3, 4)
@inline nedges(::Type{Tetra4_t}) = 6
@inline edges2nodes(::Type{Tetra4_t}) = ((1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4))
@inline function edgetypes(::Type{Tetra4_t})
    (Bar2_t(), Bar2_t(), Bar2_t(), Bar2_t(), Bar2_t(), Bar2_t())
end
@inline nfaces(t::Type{Tetra4_t}) = 4
@inline faces2nodes(::Type{Tetra4_t}) = ((1, 3, 2), (1, 2, 4), (2, 3, 4), (3, 1, 4))
@inline facetypes(::Type{Tetra4_t}) = (Tri3_t(), Tri3_t(), Tri3_t(), Tri3_t())

#---- Tetra10 ----
@inline nnodes(::Type{Tetra10_t}) = 10
@inline nodes(::Type{Tetra10_t}) = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
@inline nedges(::Type{Tetra10_t}) = 6
@inline function edges2nodes(::Type{Tetra10_t})
    ((1, 2, 5), (2, 3, 6), (3, 1, 7), (1, 4, 8), (2, 4, 9), (3, 4, 10))
end
@inline function edgetypes(::Type{Tetra10_t})
    (Bar3_t(), Bar3_t(), Bar3_t(), Bar3_t(), Bar3_t(), Bar3_t())
end
@inline nfaces(t::Type{Tetra10_t}) = 4
@inline function faces2nodes(::Type{Tetra10_t})
    ((1, 3, 2, 7, 6, 5), (1, 2, 4, 5, 9, 8), (2, 3, 4, 6, 10, 9), (3, 1, 4, 7, 8, 10))
end
@inline facetypes(::Type{Tetra10_t}) = (Tri6_t(), Tri6_t(), Tri6_t())

#---- Hexa8 ----
@inline nnodes(::Type{Hexa8_t}) = 8
@inline nodes(::Type{Hexa8_t}) = (1, 2, 3, 4, 5, 6, 7, 8)
@inline nedges(::Type{Hexa8_t}) = 12
@inline function edges2nodes(::Type{Hexa8_t})
    (
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 8),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 5),
    )
end
@inline function edgetypes(::Type{Hexa8_t})
    (
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
    )
end
@inline nfaces(::Type{Hexa8_t}) = 6
@inline function faces2nodes(::Type{Hexa8_t})
    ((1, 4, 3, 2), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (1, 5, 8, 4), (5, 6, 7, 8))
end
@inline facetypes(t::Type{Hexa8_t}) = Tuple([Quad4_t() for _ in 1:nfaces(t)])

#---- Hexa27 ----
# Warning : follow CGNS convention -> different from GMSH convention
@inline nnodes(::Type{Hexa27_t}) = 27
@inline function nodes(::Type{Hexa27_t})
    (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
    )
end
@inline nedges(::Type{Hexa27_t}) = 12
@inline function edges2nodes(::Type{Hexa27_t})
    (
        (1, 2, 9),
        (2, 3, 10),
        (3, 4, 11),
        (4, 1, 12),
        (1, 5, 13),
        (2, 6, 14),
        (3, 7, 15),
        (4, 8, 16),
        (5, 6, 17),
        (6, 7, 18),
        (7, 8, 19),
        (8, 5, 20),
    )
end
@inline function edgetypes(::Type{Hexa27_t})
    (
        Bar3_t(),
        Bar3_t(),
        Bar3_t(),
        Bar3_t(),
        Bar3_t(),
        Bar3_t(),
        Bar3_t(),
        Bar3_t(),
        Bar3_t(),
        Bar3_t(),
        Bar3_t(),
        Bar3_t(),
    )
end
@inline nfaces(::Type{Hexa27_t}) = 6
@inline function faces2nodes(::Type{Hexa27_t})
    (
        (1, 4, 3, 2, 12, 11, 10, 9, 21),
        (1, 2, 6, 5, 9, 14, 17, 13, 22),
        (2, 3, 7, 6, 10, 15, 18, 14, 23),
        (3, 4, 8, 7, 11, 16, 19, 15, 24),
        (1, 5, 8, 4, 13, 20, 16, 12, 25),
        (5, 6, 7, 8, 17, 18, 19, 20, 26),
    )
end
@inline function facetypes(::Type{Hexa27_t})
    (Quad9_t(), Quad9_t(), Quad9_t(), Quad9_t(), Quad9_t(), Quad9_t())
end

#---- Penta6_t (=Prism) ----
@inline nnodes(::Type{Penta6_t}) = 6
@inline nodes(::Type{Penta6_t}) = (1, 2, 3, 4, 5, 6)
@inline nedges(::Type{Penta6_t}) = 9
@inline function edges2nodes(::Type{Penta6_t})
    ((1, 2), (2, 3), (3, 1), (1, 4), (2, 5), (3, 6), (4, 5), (5, 6), (6, 4))
end
@inline function edgetypes(::Type{Penta6_t})
    (
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
        Bar2_t(),
    )
end
@inline nfaces(::Type{Penta6_t}) = 5
@inline function faces2nodes(::Type{Penta6_t})
    ((1, 2, 5, 4), (2, 3, 6, 5), (3, 1, 4, 6), (1, 3, 2), (4, 5, 6))
end
@inline facetypes(::Type{Penta6_t}) = (Quad4_t(), Quad4_t(), Quad4_t(), Tri3_t(), Tri3_t())

#---- Pyra5_t ----
@inline nnodes(::Type{Pyra5_t}) = 5
@inline nodes(::Type{Pyra5_t}) = (1, 2, 3, 4, 5)
@inline nedges(::Type{Pyra5_t}) = 8
@inline function edges2nodes(::Type{Pyra5_t})
    ((1, 2), (2, 3), (3, 4), (4, 1), (1, 5), (2, 5), (3, 5), (4, 5))
end
@inline edgetypes(::Type{Pyra5_t}) = ntuple(Bar2_t(), nedges(Pyra5_t()))
@inline nfaces(::Type{Pyra5_t}) = 5
@inline function faces2nodes(::Type{Pyra5_t})
    ((1, 4, 3, 2), (1, 2, 5), (2, 3, 5), (3, 4, 5), (4, 1, 5))
end
@inline facetypes(::Type{Pyra5_t}) = (Quad4_t(), Tri3_t(), Tri3_t(), Tri3_t(), Tri3_t())

#---- Poly2_t ----
@inline nnodes(::Type{Poly2_t}) = error("Function ‘nnodes‘ is not defined for ‘Poly2_t‘")
@inline nodes(::Type{Poly2_t}) = error("Function ‘nodes‘ is not defined for ‘Poly2_t‘")
@inline nedges(::Type{Poly2_t}) = error("Function ‘nedges‘ is not defined for ‘Poly2_t‘")
@inline function edges2nodes(::Type{Poly2_t})
    error("Function edges2nodes is not defined for ‘Poly2_t‘")
end
@inline nfaces(::Type{Poly2_t}) = error("Function nfaces is not defined for ‘Poly2_t‘")
@inline function faces2nodes(::Type{Poly2_t})
    error("Function ‘faces2nodes‘ is not defined for ‘Poly2_t‘")
end

#---- Poly3_t ----
@inline nnodes(::Type{Poly3_t}) = error("Function ‘nnodes‘ is not defined for ‘Poly3_t‘")
@inline nodes(::Type{Poly3_t}) = error("Function ‘nodes‘ is not defined for ‘Poly3_t‘")
@inline nedges(::Type{Poly3_t}) = error("Function ‘nedges‘ is not defined for Poly3_t")
@inline function edges2nodes(::Type{Poly3_t})
    error("Function edges2nodes is not defined for Poly3_t")
end
@inline nfaces(::Type{Poly3_t}) = error("Function nfaces is not defined for Poly3_t")
@inline function faces2nodes(::Type{Poly3_t})
    error("Function ‘faces2nodes‘ is not defined for Poly3_t")
end

function f2n_from_c2n(t, c2n::NTuple{N, T}) where {N, T}
    map_ref = faces2nodes(t)
    @assert N === nnodes(t) "Error : invalid number of nodes"
    ntuple(i -> ntuple(j -> c2n[map_ref[i][j]], length(map_ref[i])), length(map_ref))
end

function f2n_from_c2n(t, c2n::AbstractVector)
    @assert length(c2n) === nnodes(t) "Error : invalid number of nodes ($(length(c2n)) ≠ $(nnodes(t)))"
    map_ref = faces2nodes(t)
    #SVector{length(map_ref)}(ntuple(i -> SVector{length(map_ref[i])}(ntuple(j -> c2n[map_ref[i][j]] , length(map_ref[i]))) ,length(map_ref)))
    ntuple(
        i -> SVector{length(map_ref[i])}(
            ntuple(j -> c2n[map_ref[i][j]], length(map_ref[i])),
        ),
        length(map_ref),
    )
end

function cell_side(t::AbstractEntityType, c2n::AbstractVector, f2n::AbstractVector)
    all_f2n = f2n_from_c2n(t, c2n)
    side = myfindfirst(x -> x ⊆ f2n, all_f2n)
end

function oriented_cell_side(t::AbstractEntityType, c2n::AbstractVector, f2n::AbstractVector)
    all_f2n = f2n_from_c2n(t, c2n)
    side = findfirst(x -> x ⊆ f2n, all_f2n)
    if side ≠ nothing
        n = findfirst(isequal(f2n[1]), all_f2n[side])
        n === nothing ? n1 = 0 : n1 = n
        if length(f2n) === 2
            n1 == 2 ? (return -side) : (return side)
        else
            n1 + 1 > length(all_f2n[side]) ? n2 = 1 : n2 = n1 + 1
            f2n[2] == all_f2n[side][n2] ? nothing : side = -side
            return side
        end
    end
    return 0
end

"""
A `Node` is a point in space of dimension `dim`.
"""
struct Node{spaceDim, T}
    x::SVector{spaceDim, T}
end
Node(x::SVector{S, T}) where {S, T} = Node{S, T}(x)
Node(x::Vector{T}) where {T} = Node{length(x), T}(SVector{length(x), T}(x))
@inline get_coords(n::Node) = n.x
get_coords(n::Node, i) = get_coords(n)[i]
get_coords(n::Node, i::Tuple{T, Vararg{T}}) where {T} = map(j -> get_coords(n, j), i)
@inline spacedim(::Node{spaceDim, T}) where {spaceDim, T} = spaceDim
function center(nodes::AbstractVector{T}) where {T <: Node}
    Node(sum(n -> get_coords(n), nodes) / length(nodes))
end
function Base.isapprox(a::Node, b::Node, c...; d...)
    isapprox(get_coords(a), get_coords(b), c...; d...)
end
distance(a::Node, b::Node) = norm(get_coords(a) - get_coords(b))

# TopologyStyle helps dealing of curve in 2D/3D space (isCurvilinear),
# surfaces in 3D space (isSurfacic) or R^n entity in a R^n space (isVolumic)
abstract type TopologyStyle end
struct isNodal <: TopologyStyle end
struct isCurvilinear <: TopologyStyle end
struct isSurfacic <: TopologyStyle end
struct isVolumic <: TopologyStyle end

"""
    topology_style(::AbstractEntityType{topoDim}, ::Node{spaceDim, T}) where {topoDim, spaceDim, T}
    topology_style(::AbstractEntityType{topoDim}, ::AbstractArray{Node{spaceDim, T}, N}) where {spaceDim, T, N, topoDim}


Indicate the `TopologyStyle` of an entity of topology `topoDim` living in space of dimension `spaceDim`.
"""
@inline topology_style(
    ::AbstractEntityType{topoDim},
    ::Node{spaceDim, T},
) where {topoDim, spaceDim, T} = isVolumic() # everything is volumic by default

# Any "Node" is... nodal
@inline topology_style(::AbstractEntityType{0}, ::Node{spaceDim, T}) where {spaceDim, T} =
    isNodal()

# Any "line" in R^2 or R^3 is curvilinear
@inline topology_style(::AbstractEntityType{1}, ::Node{2, T}) where {T} = isCurvilinear()
@inline topology_style(::AbstractEntityType{1}, ::Node{3, T}) where {T} = isCurvilinear()

# A surface in R^2 is "volumic"
@inline topology_style(::AbstractEntityType{2}, ::Node{2, T}) where {T} = isVolumic()

# Any other surface is "surfacic"
@inline topology_style(::AbstractEntityType{2}, ::Node{spaceDim, T}) where {spaceDim, T} =
    isSurfacic()

@inline topology_style(
    etype::AbstractEntityType{topoDim},
    nodes::AbstractArray{Node{spaceDim, T}, N},
) where {spaceDim, T, N, topoDim} = topology_style(etype, nodes[1])
