abstract type AbstractShape{dim} end

struct Point <: AbstractShape{0} end
struct Line <: AbstractShape{1} end
struct Triangle <: AbstractShape{2} end
struct Square <: AbstractShape{2} end
struct Tetra <: AbstractShape{3} end
struct Prism <: AbstractShape{3} end
struct Cube <: AbstractShape{3} end

"""
    shape(::AbstractEntityType)

Return the reference `Shape` corresponding to the given `AbstractEntityType`.
"""
function shape(s::AbstractEntityType)
    error("Function 'shape' not implemented for the given AbstractEntityType : ", s)
end
shape(::Node_t) = Point()
shape(::Bar2_t) = Line()
shape(::Bar3_t) = Line()
shape(::Bar4_t) = Line()
shape(::Tri3_t) = Triangle()
shape(::Tri6_t) = Triangle()
shape(::Tri9_t) = Triangle()
shape(::Tri10_t) = Triangle()
shape(::Quad4_t) = Square()
shape(::Quad9_t) = Square()
shape(::Quad16_t) = Square()
shape(::Hexa8_t) = Cube()
shape(::Penta6_t) = Prism()

"""
    entity(s::AbstractShape, ::Val{D}) where D

Return the geometrical `Entity` corresponding to the `AbstractShape` of a given degree `D`.

Remark : Returned `entity` must be consistent with the corresponding `Lagrange` function space.
"""
function entity(s::AbstractShape, ::Val{D}) where {D}
    error(
        "Function 'entity' is not implemented for the given AbstractShape : ",
        s,
        " with degree=",
        D,
    )
end
entity(::Point, ::Val{D}) where {D} = Node_t()
entity(::Line, ::Val{0}) = Bar2_t()
entity(::Line, ::Val{1}) = Bar2_t()
entity(::Line, ::Val{2}) = Bar3_t()
entity(::Line, ::Val{3}) = Bar4_t()
entity(::Triangle, ::Val{0}) = Tri3_t()
entity(::Triangle, ::Val{1}) = Tri3_t()
entity(::Triangle, ::Val{2}) = Tri6_t()
entity(::Triangle, ::Val{3}) = Tri10_t()
entity(::Square, ::Val{0}) = Quad4_t()
entity(::Square, ::Val{1}) = Quad4_t()
entity(::Square, ::Val{2}) = Quad9_t()
entity(::Square, ::Val{3}) = Quad16_t()

"""
    nvertices(::AbstractShape)

Indicate how many vertices a shape has.
"""
nvertices(::AbstractShape) = error("Function 'nvertices' is not defined")

"""
    nedges(::AbstractShape)

Generic function. Indicate how many edges a shape has.
"""
nedges(::AbstractShape) = error("Function 'nedges' is not defined")

"""
    nfaces(::AbstractShape)

Indicate how many faces a shape has.
"""
nfaces(::AbstractShape) = error("Function 'nfaces' is not defined")

"""
    coords(::AbstractShape)

Return node coordinates of the shape in the reference space.
"""
coords(::AbstractShape) = error("Function 'coordinates' is not defined")

"""
    coords(shape::AbstractShape,i)

Return the coordinates of the `i`th shape vertices. `i` can be a tuple of
indices, then the multiples vertices's coordinates are returned.
"""
@inline coords(shape::AbstractShape, i) = coords(shape)[i]

function coords(shape::AbstractShape, i::Tuple{T, Vararg{T}}) where {T}
    map(j -> coords(shape, j), i)
end
coords(shape::AbstractShape, i::AbstractVector) = map(j -> coords(shape, j), i)

"""
    normals(::AbstractShape)

Return the outward normals of all the faces of the shape.
"""
normals(::AbstractShape) = error("Function 'normals' is not defined")

"""
    normal(shape::AbstractShape, i)

Return the outward normal of the `i`th face of the shape.
"""
normal(shape::AbstractShape, i) = normals(shape)[i]

"""
    face_area(::AbstractShape)

Return the length/area of the faces of a shape.
"""
function face_area(::AbstractShape)
    error("Function 'face_area' not implemented for the given Shape.")
end

"""
    faces2nodes(::AbstractShape)

Return the index of the vertices on the faces of a shape.
"""
function faces2nodes(::AbstractShape)
    error("Function 'faces2nodes' not implemented for the given Shape.")
end

"""
    faces2nodes(shape::AbstractShape, side)

Return the index of the vertices on the `iside`-th face of a shape. If `side` is positive, the face is oriented
preserving the cell normal. If `side` is negative, the face is returned with the opposite direction (i.e reverse
node order).
"""
function faces2nodes(shape::AbstractShape, side)
    side > 0 ? faces2nodes(shape)[side] : reverse(faces2nodes(shape)[-side])
end

"""
    face_shapes(::AbstractShape)

Return a tuple of the Shape of each face of the given (cell) Shape. For instance, a `Triangle` has
three faces, all of them are `Line`.
"""
function face_shapes(::AbstractShape)
    error("Function 'face_shapes' not implemented for the given Shape.")
end

"""
    face_shapes(shape::AbstractShape, i)

Shape of `i`-th shape of the input shape.
"""
face_shapes(shape::AbstractShape, i) = face_shapes(shape)[i]

"""
    center(::AbstractShape)

Center of the `AbstractShape`.

# Implementation
Specialize for better performances
"""
function center(s::AbstractShape)
    return sum(coords(s)) / nvertices(s)
end

nvertices(::Line) = nnodes(Bar2_t())
nvertices(::Triangle) = nnodes(Tri3_t())
nvertices(::Square) = nnodes(Quad4_t())
nvertices(::Tetra) = nnodes(Tetra4_t())
nvertices(::Cube) = nnodes(Hexa8_t())
nvertices(::Prism) = nnodes(Penta6_t())

nedges(::Line) = nedges(Bar2_t())
nedges(::Triangle) = nedges(Tri3_t())
nedges(::Square) = nedges(Quad4_t())
nedges(::Tetra) = nedges(Tetra4_t())
nedges(::Cube) = nedges(Hexa8_t())
nedges(::Prism) = nedges(Penta6_t())

nfaces(::Line) = nfaces(Bar2_t())
nfaces(::Triangle) = nfaces(Tri3_t())
nfaces(::Square) = nfaces(Quad4_t())
nfaces(::Tetra) = nfaces(Tetra4_t())
nfaces(::Cube) = nfaces(Hexa8_t())
nfaces(::Prism) = nfaces(Penta6_t())

coords(::Line) = (SA[-1.0], SA[1.0])
coords(::Triangle) = (SA[0.0, 0.0], SA[1.0, 0.0], SA[0.0, 1.0])
coords(::Square) = (SA[-1.0, -1.0], SA[1.0, -1.0], SA[1.0, 1.0], SA[-1.0, 1.0])
function coords(::Cube)
    (
        SA[-1.0, -1.0, -1.0],
        SA[1.0, -1.0, -1.0],
        SA[1.0, 1.0, -1.0],
        SA[-1.0, 1.0, -1.0],
        SA[-1.0, -1.0, 1.0],
        SA[1.0, -1.0, 1.0],
        SA[1.0, 1.0, 1.0],
        SA[-1.0, 1.0, 1.0],
    )
end
function coords(::Prism)
    (
        SA[0.0, 0.0, -1.0],
        SA[1.0, 0.0, -1.0],
        SA[0.0, 1.0, -1.0],
        SA[0.0, 0.0, 1.0],
        SA[1.0, 0.0, 1.0],
        SA[0.0, 1.0, 1.0],
    )
end

face_area(::Line) = SA[1.0, 1.0]
face_area(::Triangle) = SA[1.0, √(2.0), 1.0]
face_area(::Square) = SA[2.0, 2.0, 2.0, 2.0]
face_area(::Cube) = SA[4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
face_area(::Prism) = SA[2.0, 2 * √(2.0), 2.0, 0.5, 0.5]

faces2nodes(::Line) = (SA[1], SA[2])
faces2nodes(::Triangle) = (SA[1, 2], SA[2, 3], SA[3, 1])
faces2nodes(::Square) = (SA[1, 2], SA[2, 3], SA[3, 4], SA[4, 1])
function faces2nodes(::Cube)
    (
        SA[1, 4, 3, 2],
        SA[1, 2, 6, 5],
        SA[2, 3, 7, 6],
        SA[3, 4, 8, 7],
        SA[1, 5, 8, 4],
        SA[5, 6, 7, 8],
    )
end
function faces2nodes(::Prism)
    (SA[1, 2, 5, 4], SA[2, 3, 6, 5], SA[3, 1, 4, 6], SA[1, 3, 2], SA[4, 5, 6])
end

# Normals
normals(::Line) = (SA[-1.0], SA[1.0])
normals(::Triangle) = (SA[0.0, -1.0], SA[1.0, 1.0] ./ √(2), SA[-1.0, 0.0])
normals(::Square) = (SA[0.0, -1.0], SA[1.0, 0.0], SA[0.0, 1.0], SA[-1.0, 0.0])
function normals(::Cube)
    (
        SA[0.0, 0.0, -1.0],
        SA[0.0, -1.0, 0.0],
        SA[1.0, 0.0, 0.0],
        SA[0.0, 1.0, 0.0],
        SA[-1.0, 0.0, 0.0],
        SA[0.0, 0.0, 1.0],
    )
end
function normals(::Prism)
    (
        SA[0.0, -1.0, 0.0],
        SA[√(2.0) / 2.0, √(2.0) / 2.0, 0.0],
        SA[-1.0, 0.0, 0.0],
        SA[0.0, 0.0, -1.0],
        SA[0.0, 0.0, 1.0],
    )
end

# Centers
center(::Line) = SA[0.0]
center(::Triangle) = SA[1.0 / 3.0, 1.0 / 3.0]
center(::Square) = SA[0.0, 0.0]
center(::Cube) = SA[0.0, 0.0, 0.0]
center(::Prism) = SA[1.0 / 3.0, 1.0 / 3.0, 0.0]

# Face shapes : see notes in generic function documentation
face_shapes(::Line) = (Point(), Point())
face_shapes(shape::Union{Triangle, Square}) = ntuple(i -> Line(), nfaces(shape))
face_shapes(shape::Cube) = ntuple(i -> Square(), nfaces(shape))
face_shapes(::Prism) = (Square(), Square(), Square(), Triangle(), Triangle())

# Measure
measure(::Line) = 2.0
measure(::Triangle) = 0.5
measure(::Square) = 4.0
measure(::Cube) = 8.0
measure(::Prism) = 1.0
