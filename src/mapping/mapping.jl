# COMMON
"""
    mapping(ctype::AbstractEntityType, cnodes, ξ)
    mapping(cshape::AbstractShape, cnodes, ξ)

Map the reference shape on the local shape.

# Implementation
This function must be implemented for all shape.

# `::Bar2_t`
Map the reference 2-nodes bar [-1,1] on the local bar:
``F(\\xi) = \\dfrac{x_r - x_l}{2} \\xi + \\dfrac{x_r + x_l}{2}``

# `::Tri3_t`
Map the reference 3-nodes Triangle [0,1] x [0,1] on the local triangle.
``F(\\xi \\\\ \\eta) = (1 - \\xi - \\eta) M_1 + x M_2 + y M_3``

# `::Quad4_t`
Map the reference 4-nodes square [-1,1] x [-1,1] on the 4-quadrilateral.

# `::Tri6_t`
Map the reference 6-nodes triangle [0,1] x [0,1] on the P2 curved-triangle.
`` F(\\xi) = \\sum \\lambda_i(\\xi) x_i ``
where ``\\lambda_i`` are the Lagrange P2 shape functions and ``x_i`` are the local
curved-triangle vertices' coordinates.

# `::Quad9_t`
Map the reference 4-nodes square [-1,1] x [-1,1] on the P2 curved-quadrilateral.
`` F(\\xi) = \\sum \\lambda_i(\\xi) x_i ``
where ``\\lambda_i`` are the Lagrange P2 shape functions and ``x_i`` are the local
curved-quadrilateral vertices' coordinates.

# `::Hexa8_t`
Map the reference 8-nodes cube [-1,1] x [-1,1] x [-1,1] on the 8-hexa.

# `::Hexa27_t`
Map the reference 8-nodes cube [-1,1] x [-1,1] x [-1,1] on the 27-hexa.

# `::Penta6_t`
Map the reference 6-nodes prism [0,1] x [0,1] x [-1,1] on the 6-penta (prism).

"""
function mapping(::AbstractEntityType, cnodes, ξ)
    error("Function 'mapping' is not defined for this AbstractEntityType")
end
mapping(ctype::AbstractEntityType, cnodes) = ξ -> mapping(ctype, cnodes, ξ)

"""
    mapping_inv(::AbstractEntityType, cnodes, x)

Map the local shape on the reference shape.

# Implementation
This function does not have to be implemented for all shape.

# `::Bar2_t`
Map the local bar on the reference 2-nodes bar [-1,1]:
``F^{-1}(x) = \\dfrac{2x - x_r - x_l}{x_r - x_l}``

# `::Tri3_t`
Map the local triangle on the reference 3-nodes Triangle [0,1] x [0,1].

TODO: check this formulae with SYMPY

```math
F^{-1} \\begin{pmatrix} x \\\\ y \\end{pmatrix} =
\\frac{1}{x_1(y_2-y_3) + x_2(y_3-y_1) + x_3(y_1-y_2)}
\\begin{pmatrix}
    (y_3-y_1)x + (x_1 - x_3)y + (x_3 y_1 - x_1 y_3) \\\\
    (y_1-y_2)x + (x_2 - x_1)x + (x_1 y_2 - x_2 y_1)
\\end{pmatrix}
```

# ::`Quad4_t`
Map the  PARALLELOGRAM quadrilateral on the reference 4-nodes square [-1,1] x [-1,1].
Warning : this mapping is only corrects for parallelogram quadrilateral, not for any quadrilateral.

-----
TODO: check this formulae with SYMPY
-----

```math
F^{-1} \\begin{pmatrix} x \\\\ y \\end{pmatrix} =
\\begin{pmatrix}
    a_1 x + b_1 y + c_1 \\\\
    a_2 x + b_2 y + c_2
\\end{pmatrix}
```
with
```math
\\begin{aligned}
    a_1 & = \\dfrac{-2 (y_3-y_2)}{\\Delta} \\\\
    b_1 & = \\dfrac{2 (x_3-x_2)}{\\Delta} \\\\
    c_1 & = -1 - a_1 x_1 - b_1 y_1 \\\\
    a_2 & = \\dfrac{-2 (y_1-y_2)}{\\Delta} \\\\
    b_2 & = \\dfrac{2 (x_1 - x_2)}{\\Delta} \\\\
    c_2 & = -1 - a_2 x_1 - b_2 y_1
\\end{aligned}
```
where
`` \\Delta = (x_1 - x_2)(y_3 - y_2) - (x_3 - x_2)(y_1 - y_2)``
"""
mapping_inv(::AbstractEntityType, cnodes, x) =
    error("Function 'mapping_inv' is not defined")
mapping_inv(ctype::AbstractEntityType, cnodes) = x -> mapping_inv(ctype, cnodes, x)

"""
    mapping_jacobian(nodes, ctype::AbstractEntityType, ξ)

Jacobian matrix of the mapping : ``\\dfrac{\\partial F_i}{\\partial \\xi_j}``.

# Implementation
Default version using ForwardDiff, but can be specified for each shape.
"""
function mapping_jacobian(cnodes, ctype::AbstractEntityType, ξ)
    ForwardDiff.jacobian(η -> mapping(ctype, cnodes, η), ξ)
end

"""
    mapping_jacobian_inv(nodes, etype::AbstractEntityType, ξ)

Inverse of the mapping jacobian matrix. This is not exactly equivalent to the `mapping_inv_jacobian` since this function is
evaluated in the reference element.

# Implementation
Default version using ForwardDiff, but can be specified for each shape.
"""
function mapping_jacobian_inv(cnodes, ctype::AbstractEntityType, ξ)
    inv(ForwardDiff.jacobian(mapping(ctype, cnodes), ξ))
end

"""
    mapping_inv_jacobian(nodes, etype::AbstractEntityType, x)

Jacobian matrix of the inverse mapping : ``\\dfrac{\\partial F_i^{-1}}{\\partial x_j}``

Contrary to `mapping_jacobian_inv`, this function is not always defined because the
inverse mapping, F^-1, is not always defined.

# Implementation
Default version using LinearAlgebra to inverse the matrix, but can be specified for each shape (if it exists).
"""
function mapping_inv_jacobian(cnodes, ctype::AbstractEntityType, x)
    inv(mapping_jacobian(cnodes, ctype, mapping_inv(ctype, cnodes, x)))
end

"""
    mapping_det_jacobian(nodes, etype::AbstractEntityType, ξ)

Absolute value of the determinant of the mapping Jacobian matrix, expressed in the reference element.

# Implementation
Default version using `mapping_jacobian`, but can be specified for each shape.
"""
function mapping_det_jacobian(nodes, etype::AbstractEntityType, ξ)
    abs(det(mapping_jacobian(nodes, etype, ξ)))
end

"""
    mapping(cshape::AbstractShape, cnodes, ξ)

Returns the mapping of the an abstract shape (=ref element) to a target element defined by its `nodes`.

For instance, if `cshape == Line`, then the mapping is the same wether the input is the Shape or a `Bar2_t`.
However if the cell is of type `Bar3_t`, it is still the `Bar2_t` mapping that is returned.
"""
function mapping(cshape::AbstractShape, cnodes, ξ)
    error("Function 'mapping' is not defined for this shape")
end
mapping(cshape::AbstractShape, cnodes) = ξ -> mapping(cshape, cnodes, ξ)

"""
    mapping_face(cshape::AbstractShape, side)

Build a mapping from the face reference element (corresponding to the `side`-th face of `cshape`)
to the cell reference element (i.e the `cshape`).

# Implementation
We could define this function as an alias to `mapping_face(cshape, side, 1:nnodes(face_shapes(cshape, side))`
but for performance issue, I prefer to keep two independant functions for now.
"""
function mapping_face(cshape::AbstractShape, side)
    f2n = faces2nodes(cshape, side)
    _coords = coords(cshape, f2n)
    fnodes = map(Node, _coords)
    return MappingFace(mapping(face_shapes(cshape, side), fnodes), nothing)
end

"""
    mapping_face(cshape::AbstractShape, side, permutation)

Build a mapping from the face reference element (corresponding to the `side`-th face of `cshape`)
to the cell reference element (i.e the `cshape`), using a permutation of the face nodes.
"""
function mapping_face(cshape::AbstractShape, side, permutation)
    f2n = faces2nodes(cshape, side)[permutation]
    _coords = coords(cshape, f2n)
    fnodes = Node.(_coords)
    return MappingFace(mapping(face_shapes(cshape, side), fnodes), nothing)
end

struct MappingFace{F1, F2}
    f1::F1
    f2::F2
end
(m::MappingFace)(x) = m.f1(x)
CallableStyle(::Type{<:MappingFace}) = IsCallableStyle()

# POINT : this may seem stupid, but it is usefull for coherence
mapping(::Node_t, cnodes, ξ) = cnodes[1].x
mapping(::Point, cnodes, ξ) = mapping(Node_t(), cnodes, ξ)

# LINE
mapping(::Line, cnodes, ξ) = mapping(Bar2_t(), cnodes, ξ)

function mapping(::Bar2_t, cnodes, ξ)
    (cnodes[2].x - cnodes[1].x) / 2.0 .* ξ + (cnodes[2].x + cnodes[1].x) / 2.0
end

function mapping_inv(::Bar2_t, cnodes, x)
    SA[(2 * x[1] - cnodes[2].x[1] - cnodes[1].x[1]) / (cnodes[2].x[1] - cnodes[1].x[1])]
end

"""
    mapping_jacobian(nodes, ::Bar2_t, ξ)

Mapping's jacobian matrix for the reference 2-nodes bar [-1, 1] to the local bar.

``\\dfrac{\\partial F}{\\partial \\xi} = \\dfrac{x_r - x_l}{2}``
"""
function mapping_jacobian(nodes, ::Bar2_t, ξ)
    axes(nodes) == (Base.OneTo(2),) || error("Invalid number of nodes")
    @inbounds (nodes[2].x .- nodes[1].x) ./ 2.0
end

"""
    mapping_jacobian_inv(nodes, ::Bar2_t, ξ)

Inverse of mapping's jacobian matrix for the reference 2-nodes bar [-1, 1] to the local bar.

``\\dfrac{\\partial F}{\\partial \\xi}^{-1} = \\dfrac{2}{x_r - x_l}``
"""
mapping_jacobian_inv(nodes, ::Bar2_t, ξ) = @SMatrix[2.0 / (nodes[2].x[1] .- nodes[1].x[1])]

"""
    mapping_inv_jacobian(nodes, ::Bar2_t, x)

Mapping's jacobian matrix for the local bar to the reference 2-nodes bar [-1, 1].

``\\dfrac{\\partial F^{-1}}{\\partial x} = \\dfrac{2}{x_r - x_l}``
"""
mapping_inv_jacobian(nodes, ::Bar2_t, x) = 2.0 / (nodes[2].x[1] - nodes[1].x[1])

"""
    mapping_det_jacobian(nodes, ::Bar2_t, ξ)

Absolute value of the determinant of the mapping Jacobian matrix for the
reference 2-nodes bar [-1,1] to the local bar mapping.

``|det(J(\\xi))| = \\dfrac{|x_r - x_l|}{2}``
"""
mapping_det_jacobian(nodes, ::Bar2_t, ξ) = norm(nodes[2].x - nodes[1].x) / 2.0

function mapping(::Bar3_t, cnodes, ξ)
    ξ .* (ξ .- 1) / 2 .* cnodes[1].x .+ ξ .* (ξ .+ 1) / 2 .* cnodes[2].x .+
    (1 .- ξ) .* (1 .+ ξ) .* cnodes[3].x
end

"""
    mapping_jacobian(nodes, ::Bar3_t, ξ)

Mapping's jacobian matrix for the reference 2-nodes bar [-1, 1] to the local bar.

``\\dfrac{\\partial F}{\\partial \\xi} = \\frac{1}{2} \\left( (2\\xi - 1) M_1 + (2\\xi + 1)M_2 - 4 \\xi M_3\\right)
"""
function mapping_jacobian(nodes, ::Bar3_t, ξ)
    (nodes[1].x .* (2 .* ξ .- 1) + nodes[2].x .* (2 .* ξ .+ 1) - nodes[3].x .* 4 .* ξ) ./ 2
end

"""
    mapping_jacobian_inv(nodes, ::Bar3_t, ξ)

Inverse of mapping's jacobian matrix for the reference 2-nodes bar [-1, 1] to the local bar.

``\\dfrac{\\partial F}{\\partial \\xi}^{-1} = \\frac{2}{(2\\xi - 1) M_1 + (2\\xi + 1)M_2 - 4 \\xi M_3}``
"""
function mapping_jacobian_inv(nodes, ::Bar3_t, ξ)
    2.0 / (nodes[1].x .* (2 .* ξ .- 1) + nodes[2].x .* (2 .* ξ .+ 1) - nodes[3].x .* 4 .* ξ)
end

# TRIANGLE P1
mapping(::Triangle, cnodes, ξ) = mapping(Tri3_t(), cnodes, ξ)

function mapping(::Tri3_t, cnodes, ξ)
    return (1 - ξ[1] - ξ[2]) .* cnodes[1].x + ξ[1] .* cnodes[2].x + ξ[2] .* cnodes[3].x
end

function mapping_inv(::Tri3_t, cnodes, x)
    # Alias (should be inlined, but waiting for Ghislain's modification of Node)
    x1 = cnodes[1].x[1]
    x2 = cnodes[2].x[1]
    x3 = cnodes[3].x[1]
    y1 = cnodes[1].x[2]
    y2 = cnodes[2].x[2]
    y3 = cnodes[3].x[2]

    return SA[
        (y3 - y1) * x[1] + (x1 - x3) * x[2] + (x3 * y1 - x1 * y3),
        (y1 - y2) * x[1] + (x2 - x1) * x[2] + (x1 * y2 - x2 * y1),
    ] / (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
end

"""
    mapping_jacobian(nodes, ::Tri3_t, ξ)

Mapping's jacobian matrix for the reference 3-nodes Triangle [0,1] x [0,1] to the local
triangle mapping.

```math
\\dfrac{\\partial F_i}{\\partial \\xi_j} =
\\begin{pmatrix}
    M_2 - M_1 & M_3 - M_1
\\end{pmatrix}
```
"""
function mapping_jacobian(nodes, ::Tri3_t, ξ)
    return hcat(nodes[2].x - nodes[1].x, nodes[3].x - nodes[1].x)
end

"""
    mapping_jacobian_inv(nodes, ::Tri3_t, ξ)

Inverse of mapping's jacobian matrix for the reference 3-nodes Triangle [0,1] x [0,1] to the local
triangle mapping.

```math
\\dfrac{\\partial F_i}{\\partial \\xi_j}^{-1} =
\\frac{1}{(x_1 - x_2)(y_1 - y_3) - (x_1 - x_3)(y_1 - y_2)}
\\begin{pmatrix}
    -y_1 + y_3 &  x_1 - x_3 \\\\
     y_1 - y_2 & -x_1 + x_2
\\end{pmatrix}
```
"""
function mapping_jacobian_inv(nodes, ::Tri3_t, ξ)
    # Alias (should be inlined, but waiting for Ghislain's modification of Node)
    x1 = nodes[1].x[1]
    x2 = nodes[2].x[1]
    x3 = nodes[3].x[1]
    y1 = nodes[1].x[2]
    y2 = nodes[2].x[2]
    y3 = nodes[3].x[2]

    return SA[
        -y1+y3 x1-x3
        y1-y2 -x1+x2
    ] ./ ((x1 - x2) * (y1 - y3) - (x1 - x3) * (y1 - y2))
end

"""
    mapping_inv_jacobian(nodes, ::Tri3_t, x)

Mapping's jacobian matrix for the local triangle to the reference 3-nodes
Triangle [0,1] x [0,1] mapping.

-----
TODO: check this formulae with SYMPY
-----


```math
\\frac{\\partial F_i^{-1}}{\\partial x_j} =
\\frac{1}{x_1 (y_2 - y_3) + x_2 (y_3 - y_1) + x_3 (y_1 - y_2)}
\\begin{pmatrix}
    y_3 - y_1 & x_1 - x_3 \\\\
    y_1 - y_2 & x_2 - x_1
\\end{pmatrix}
```
"""
function mapping_inv_jacobian(nodes, ::Tri3_t, x)
    # Alias (should be inlined, but waiting for Ghislain's modification of Node)
    x1 = nodes[1].x[1]
    x2 = nodes[2].x[1]
    x3 = nodes[3].x[1]
    y1 = nodes[1].x[2]
    y2 = nodes[2].x[2]
    y3 = nodes[3].x[2]

    return SA[
        (y3-y1) (x1-x3)
        (y1-y2) (x2-x1)
    ] / (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
end

"""
    mapping_det_jacobian(nodes, ::Tri3_t, ξ)

Absolute value of the determinant of the mapping Jacobian matrix for the
the reference 3-nodes Triangle [0,1] x [0,1] to the local triangle mapping.

`` |J| = |(x_2 - x_1) (y_3 - y_1) - (x_3 - x_1) (y_2 - y_1)|``
"""
function mapping_det_jacobian(nodes, ::Tri3_t, ξ)
    # Alias (should be inlined, but waiting for Ghislain's modification of Node)
    x1 = nodes[1].x[1]
    x2 = nodes[2].x[1]
    x3 = nodes[3].x[1]
    y1 = nodes[1].x[2]
    y2 = nodes[2].x[2]
    y3 = nodes[3].x[2]

    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
end

# Quad P1
mapping(::Square, cnodes, ξ) = mapping(Quad4_t(), cnodes, ξ)

function mapping(::Quad4_t, cnodes, ξ)
    return (
        (ξ[1] - 1) * (ξ[2] - 1) .* cnodes[1].x - (ξ[1] + 1) * (ξ[2] - 1) .* cnodes[2].x +
        (ξ[1] + 1) * (ξ[2] + 1) .* cnodes[3].x - (ξ[1] - 1) * (ξ[2] + 1) .* cnodes[4].x
    ) ./ 4
end

function mapping_inv(::Quad4_t, cnodes, x)
    # Alias (should be inlined, but waiting for Ghislain's modification of Node)
    x1 = cnodes[1].x[1]
    x2 = cnodes[2].x[1]
    x3 = cnodes[3].x[1]
    y1 = cnodes[1].x[2]
    y2 = cnodes[2].x[2]
    y3 = cnodes[3].x[2]

    # Copied from fortran
    delta = (x1 - x2) * (y3 - y2) - (x3 - x2) * (y1 - y2)
    a = SA[-2.0 * (y3 - y2), -2.0 * (y1 - y2)] ./ delta
    b = SA[2.0 * (x3 - x2), 2.0 * (x1 - x2)] ./ delta
    c = SA[-1.0 - a[1] * x1 - b[1] * y1, -1.0 - a[2] * x1 - b[2] * y1]

    return a .* x[1] + b .* x[2] + c
end

"""
    mapping_jacobian(nodes, ::Quad4_t, ξ)

Mapping's jacobian matrix for the reference square [-1,1] x [-1,1]
to the 4-quadrilateral

```math
\\frac{\\partial F}{\\partial \\xi} = -M_1 + M_2 + M_3 - M_4 + \\eta (M_1 - M_2 + M_3 - M_4)
\\frac{\\partial F}{\\partial \\eta} = -M_1 - M_2 + M_3 + M_4 + \\xi (M_1 - M_2 + M_3 - M_4)
```
"""
function mapping_jacobian(nodes, ::Quad4_t, ξ)
    return hcat(
        -nodes[1].x + nodes[2].x + nodes[3].x - nodes[4].x +
        ξ[2] .* (nodes[1].x - nodes[2].x + nodes[3].x - nodes[4].x),
        -nodes[1].x - nodes[2].x +
        nodes[3].x +
        nodes[4].x +
        ξ[1] .* (nodes[1].x - nodes[2].x + nodes[3].x - nodes[4].x),
    ) ./ 4
end

"""
    mapping_jacobian(nodes, ::Quad4_t, ξ)

Inverse of mapping's jacobian matrix for the reference square [-1,1] x [-1,1]
to the 4-quadrilateral
"""
function mapping_jacobian_inv(nodes::AbstractArray{<:Node{2, T}}, ::Quad4_t, ξη) where {T}
    # Alias
    axes(nodes) == (Base.OneTo(4),) || error("Invalid number of nodes")
    axes(ξη) == (Base.OneTo(2),) || error("Invalid number of coordinates")
    @inbounds begin
        x1, y1 = nodes[1].x
        x2, y2 = nodes[2].x
        x3, y3 = nodes[3].x
        x4, y4 = nodes[4].x
        ξ = ξη[1]
        η = ξη[2]
    end
    return 4 .* SA[
        (1 - ξ) * (y1 - y4)+(1 + ξ) * (y2 - y3) (1 - ξ) * (x4 - x1)+(1 + ξ) * (x3 - x2)
        (1 - η) * (y2 - y1)+(1 + η) * (y3 - y4) (1 - η) * (x1 - x2)+(1 + η) * (x4 - x3)
    ] ./ (
        ((1 - ξ) * (x4 - x1) + (1 + ξ) * (x3 - x2)) *
        ((1 - η) * (y2 - y1) + (1 + η) * (y3 - y4)) -
        ((1 - η) * (x2 - x1) + (1 + η) * (x3 - x4)) *
        ((1 - ξ) * (y4 - y1) + (1 + ξ) * (y3 - y2))
    )
end

"""
    mapping_inv_jacobian(nodes, ::Quad4_t, x)

Mapping's jacobian matrix for the PARALLELOGRAM quadrilateral to the
reference 4-nodes square [-1,1] x [-1,1].

-----
TODO: check this formulae with SYMPY + need formulae for general quad, not paraquad
-----

```math
F^{-1} \\begin{pmatrix} x \\\\ y \\end{pmatrix} =
\\frac{2}{(x_1-x_2) (y_3-y_2) - (x_3-x_2) (y_1-y_2)}
\\begin{pmatrix}
    (y_2 - y_3) & (x_3 - x_2) \\\\
    (y_2 - y_1) & (x_1 - x_2)
\\end{pmatrix}
```
"""
# function mapping_inv_jacobian(nodes, ::Quad4_t, x)
#     # Alias (should be inlined, but waiting for Ghislain's modification of Node)
#     x1 = nodes[1].x[1]; x2 = nodes[2].x[1]; x3 = nodes[3].x[1]
#     y1 = nodes[1].x[2]; y2 = nodes[2].x[2]; y3 = nodes[3].x[2]

#     return SA[
#             (y2 - y3)    (x3 - x2);
#             (y2 - y1)    (x1 - x2)
#     ] .* 2. ./ ((x1-x2) * (y3-y2) - (x3-x2)*(y1-y2))
# end

"""
    mapping_det_jacobian(nodes, ::Quad4_t, ξ)

Absolute value of the determinant of the mapping Jacobian matrix for the
the reference square [-1,1] x [-1,1] to the 4-quadrilateral mapping.
"""
function mapping_det_jacobian(nodes::AbstractArray{<:Node{2, T}}, ::Quad4_t, ξη) where {T}
    axes(nodes) == (Base.OneTo(4),) || error("Invalid number of nodes")
    axes(ξη) == (Base.OneTo(2),) || error("Invalid number of coordinates")
    @inbounds begin
        x1, y1 = nodes[1].x
        x2, y2 = nodes[2].x
        x3, y3 = nodes[3].x
        x4, y4 = nodes[4].x
        ξ = ξη[1]
        η = ξη[2]
    end
    return abs(
        -((1 - ξ) * (x4 - x1) + (1 + ξ) * (x3 - x2)) *
        ((1 - η) * (y2 - y1) + (1 + η) * (y3 - y4)) +
        ((1 - η) * (x2 - x1) + (1 + η) * (x3 - x4)) *
        ((1 - ξ) * (y4 - y1) + (1 + ξ) * (y3 - y2)),
    ) / 16.0
end

# TRIANGLE P2
function mapping(::Tri6_t, cnodes, ξ)
    # Shape functions
    λ₁  = (1 - ξ[1] - ξ[2]) * (1 - 2 * ξ[1] - 2 * ξ[2])  # = (1 - x - y)(1 - 2x - 2y)
    λ₂  = ξ[1] * (2 * ξ[1] - 1) # = x (2x - 1)
    λ₃  = ξ[2] * (2 * ξ[2] - 1) # = y (2y - 1)
    λ₁₂ = 4 * ξ[1] * (1 - ξ[1] - ξ[2]) # = 4x (1 - x - y)
    λ₂₃ = 4 * ξ[1] * ξ[2]
    λ₃₁ = 4 * ξ[2] * (1 - ξ[1] - ξ[2]) # = 4y (1 - x - y)

    # Is there a way to write this with a more consise way?
    # something like sum( (lambda, n) -> lambda * n, zip(lambda, nodes))
    # Note that the use of parenthesis is mandatory here otherwise only the first line is returned
    return (
        λ₁ .* cnodes[1].x +
        λ₂ .* cnodes[2].x +
        λ₃ .* cnodes[3].x +
        λ₁₂ .* cnodes[4].x +
        λ₂₃ .* cnodes[5].x +
        λ₃₁ .* cnodes[6].x
    )
end

# PARAQUAD P2
function mapping(::Quad9_t, cnodes, ξ)
    # Shape functions
    λ₁ = ξ[1] * ξ[2] * (1 - ξ[1]) * (1 - ξ[2]) / 4 # =   xy (1 - x)(1 - y) / 4
    λ₂ = -ξ[1] * ξ[2] * (1 + ξ[1]) * (1 - ξ[2]) / 4 # = - xy (1 + x)(1 - y) / 4
    λ₃ = ξ[1] * ξ[2] * (1 + ξ[1]) * (1 + ξ[2]) / 4 # =   xy (1 + x)(1 + y) / 4
    λ₄ = -ξ[1] * ξ[2] * (1 - ξ[1]) * (1 + ξ[2]) / 4 # = - xy (1 - x)(1 + y) / 4
    λ₁₂ = -(1 + ξ[1]) * (1 - ξ[1]) * ξ[2] * (1 - ξ[2]) / 2 # = - (1 + x)(1 -x)y(1 - y) / 2
    λ₂₃ = ξ[1] * (1 + ξ[1]) * (1 - ξ[2]) * (1 + ξ[2]) / 2 # = x(1 + x)(1 - y)(1 + y) / 2
    λ₃₄ = (1 - ξ[1]) * (1 + ξ[1]) * ξ[2] * (1 + ξ[2]) / 2 # = (1 - x)(1 + x)y(1 + y) /2
    λ₄₁ = -ξ[1] * (1 - ξ[1]) * (1 - ξ[2]) * (1 + ξ[2]) / 2 # = - x(1 - x)(1 - y)(1 + y) / 2
    λ₁₂₃₄ = (1 - ξ[1]) * (1 + ξ[1]) * (1 - ξ[2]) * (1 + ξ[2]) # = (1 - x)(1 + x)(1 - y)(1 + y)

    # Is there a way to write this with a more consise way?
    # something like sum( (lambda, n) -> lambda * n, zip(lambda, nodes))
    # Note that the use of parenthesis is mandatory here otherwise only the first line is returned
    return (
        λ₁ .* cnodes[1].x +
        λ₂ .* cnodes[2].x +
        λ₃ .* cnodes[3].x +
        λ₄ .* cnodes[4].x +
        λ₁₂ .* cnodes[5].x +
        λ₂₃ .* cnodes[6].x +
        λ₃₄ .* cnodes[7].x +
        λ₄₁ .* cnodes[8].x +
        λ₁₂₃₄ .* cnodes[9].x
    )
end

function __ordered_lagrange_shape_fns(::Type{Bcube.Quad16_t})
    refCoords = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]

    # Lagrange nodes ordering (index in `refCoords`)
    I_to_ij = [
        1 1 # 1
        4 1 # 2
        4 4 # 3
        1 4 # 4
        2 1 # 5
        3 1 # 6
        4 2 # 7
        4 3 # 8
        3 4 # 9
        2 4 # 10
        1 3 # 11
        1 2 # 12
        2 2 # 13
        3 2 # 14
        3 3 # 15
        2 3 # 16
    ]
    @variables ξ[1:2]
    _ξ = ξ[1]
    _η = ξ[2]
    _λs = [
        _lagrange_poly(I_to_ij[k, 1], _ξ, refCoords) *
        _lagrange_poly(I_to_ij[k, 2], _η, refCoords) for k in 1:16
    ]
    expr_λs = Symbolics.toexpr.((_λs))
    return :(SA[$(expr_λs...)])
end

@generated function _ordered_lagrange_shape_fns(entity::AbstractEntityType, ξ)
    __ordered_lagrange_shape_fns(entity)
end

# Warning : only valid for "tensor" "Lagrange" entities
function mapping(ctype::Union{Quad16_t}, cnodes, ξ)
    λs = _ordered_lagrange_shape_fns(ctype, ξ)
    return sum(((λ, node),) -> λ .* coords(node), zip(λs, cnodes))
end

# Hexa8
mapping(::Cube, cnodes, ξ) = mapping(Hexa8_t(), cnodes, ξ)

function mapping(::Hexa8_t, cnodes, ξηζ)
    ξ = ξηζ[1]
    η = ξηζ[2]
    ζ = ξηζ[3]
    return (
        (1 - ξ) * (1 - η) * (1 - ζ) .* cnodes[1].x  # = (1 - x) * (1 - y) * (1 - z) / 8
        +
        (1 + ξ) * (1 - η) * (1 - ζ) .* cnodes[2].x  # = (1 + x) * (1 - y) * (1 - z) / 8
        +
        (1 + ξ) * (1 + η) * (1 - ζ) .* cnodes[3].x  # = (1 + x) * (1 + y) * (1 - z) / 8
        +
        (1 - ξ) * (1 + η) * (1 - ζ) .* cnodes[4].x  # = (1 - x) * (1 + y) * (1 - z) / 8
        +
        (1 - ξ) * (1 - η) * (1 + ζ) .* cnodes[5].x  # = (1 - x) * (1 - y) * (1 + z) / 8
        +
        (1 + ξ) * (1 - η) * (1 + ζ) .* cnodes[6].x  # = (1 + x) * (1 - y) * (1 + z) / 8
        +
        (1 + ξ) * (1 + η) * (1 + ζ) .* cnodes[7].x  # = (1 + x) * (1 + y) * (1 + z) / 8
        +
        (1 - ξ) * (1 + η) * (1 + ζ) .* cnodes[8].x  # = (1 - x) * (1 + y) * (1 + z) / 8
    ) ./ 8
end

function mapping_jacobian(nodes, ::Hexa8_t, ξηζ)
    ξ = ξηζ[1]
    η = ξηζ[2]
    ζ = ξηζ[3]
    M1 = nodes[1].x
    M2 = nodes[2].x
    M3 = nodes[3].x
    M4 = nodes[4].x
    M5 = nodes[5].x
    M6 = nodes[6].x
    M7 = nodes[7].x
    M8 = nodes[8].x
    return hcat(
        ((M2 - M1) * (1 - η) + (M3 - M4) * (1 + η)) * (1 - ζ) +
        ((M6 - M5) * (1 - η) + (M7 - M8) * (1 + η)) * (1 + ζ),
        ((M4 - M1) * (1 - ξ) + (M3 - M2) * (1 + ξ)) * (1 - ζ) +
        ((M8 - M5) * (1 - ξ) + (M7 - M6) * (1 + ξ)) * (1 + ζ),
        ((M5 - M1) * (1 - ξ) + (M6 - M2) * (1 + ξ)) * (1 - η) +
        ((M8 - M4) * (1 - ξ) + (M7 - M3) * (1 + ξ)) * (1 + η),
    ) ./ 8.0
end

# Remark : the determinant, obtained with sympy, is big (approx. 3000 caracters)
# function mapping_det_jacobian(nodes, ::Hexa8_t, ξη)
#     # Alias (should be inlined, but waiting for Ghislain's modification of Node)
#     x1 = nodes[1].x[1]; x2 = nodes[2].x[1]; x3 = nodes[3].x[1]; x4 = nodes[4].x[1]
#     x5 = nodes[5].x[1]; x6 = nodes[6].x[1]; x7 = nodes[7].x[1]; x8 = nodes[8].x[1]

#     y1 = nodes[1].x[2]; y2 = nodes[2].x[2]; y3 = nodes[3].x[2]; y4 = nodes[4].x[2]
#     y5 = nodes[5].x[2]; y6 = nodes[6].x[2]; y7 = nodes[7].x[2]; y8 = nodes[8].x[2]

#     z1 = nodes[1].x[3]; z2 = nodes[2].x[3]; z3 = nodes[3].x[3]; z4 = nodes[4].x[3]
#     z5 = nodes[5].x[3]; z6 = nodes[6].x[3]; z7 = nodes[7].x[3]; z8 = nodes[8].x[3]

# end

# Hexa27

function mapping(::Hexa27_t, cnodes, ξηζ)
    ξ = ξηζ[1]
    η = ξηζ[2]
    ζ = ξηζ[3]
    return (
        ξ * η * ζ * (ξ - 1) * (η - 1) * (ζ - 1) / 8.0 .* cnodes[1].x +
        ξ * η * ζ * (ξ + 1) * (η - 1) * (ζ - 1) / 8.0 .* cnodes[2].x +
        ξ * η * ζ * (ξ + 1) * (η + 1) * (ζ - 1) / 8.0 .* cnodes[3].x +
        ξ * η * ζ * (ξ - 1) * (η + 1) * (ζ - 1) / 8.0 .* cnodes[4].x +
        ξ * η * ζ * (ξ - 1) * (η - 1) * (ζ + 1) / 8.0 .* cnodes[5].x +
        ξ * η * ζ * (ξ + 1) * (η - 1) * (ζ + 1) / 8.0 .* cnodes[6].x +
        ξ * η * ζ * (ξ + 1) * (η + 1) * (ζ + 1) / 8.0 .* cnodes[7].x +
        ξ * η * ζ * (ξ - 1) * (η + 1) * (ζ + 1) / 8.0 .* cnodes[8].x +
        -η * ζ * (ξ^2 - 1) * (η - 1) * (ζ - 1) / 4.0 .* cnodes[9].x +
        -ξ * ζ * (ξ + 1) * (η^2 - 1) * (ζ - 1) / 4.0 .* cnodes[10].x +
        -η * ζ * (ξ^2 - 1) * (η + 1) * (ζ - 1) / 4.0 .* cnodes[11].x +
        -ξ * ζ * (ξ - 1) * (η^2 - 1) * (ζ - 1) / 4.0 .* cnodes[12].x +
        -ξ * η * (ξ - 1) * (η - 1) * (ζ^2 - 1) / 4.0 .* cnodes[13].x +
        -ξ * η * (ξ + 1) * (η - 1) * (ζ^2 - 1) / 4.0 .* cnodes[14].x +
        -ξ * η * (ξ + 1) * (η + 1) * (ζ^2 - 1) / 4.0 .* cnodes[15].x +
        -ξ * η * (ξ - 1) * (η + 1) * (ζ^2 - 1) / 4.0 .* cnodes[16].x +
        -η * ζ * (ξ^2 - 1) * (η - 1) * (ζ + 1) / 4.0 .* cnodes[17].x +
        -ξ * ζ * (ξ + 1) * (η^2 - 1) * (ζ + 1) / 4.0 .* cnodes[18].x +
        -η * ζ * (ξ^2 - 1) * (η + 1) * (ζ + 1) / 4.0 .* cnodes[19].x +
        -ξ * ζ * (ξ - 1) * (η^2 - 1) * (ζ + 1) / 4.0 .* cnodes[20].x +
        ζ * (ξ^2 - 1) * (η^2 - 1) * (ζ - 1) / 2.0 .* cnodes[21].x +
        η * (ξ^2 - 1) * (η - 1) * (ζ^2 - 1) / 2.0 .* cnodes[22].x +
        ξ * (ξ + 1) * (η^2 - 1) * (ζ^2 - 1) / 2.0 .* cnodes[23].x +
        η * (ξ^2 - 1) * (η + 1) * (ζ^2 - 1) / 2.0 .* cnodes[24].x +
        ξ * (ξ - 1) * (η^2 - 1) * (ζ^2 - 1) / 2.0 .* cnodes[25].x +
        ζ * (ξ^2 - 1) * (η^2 - 1) * (ζ + 1) / 2.0 .* cnodes[26].x +
        -(ξ^2 - 1) * (η^2 - 1) * (ζ^2 - 1) .* cnodes[27].x
    )
end

# Penta6
mapping(::Prism, cnodes, ξ) = mapping(Penta6_t(), cnodes, ξ)

function mapping(::Penta6_t, cnodes, ξηζ)
    ξ = ξηζ[1]
    η = ξηζ[2]
    ζ = ξηζ[3]
    return (
        (1 - ξ - η) * (1 - ζ) .* cnodes[1].x +
        ξ * (1 - ζ) .* cnodes[2].x +
        η * (1 - ζ) .* cnodes[3].x +
        (1 - ξ - η) * (1 + ζ) .* cnodes[4].x +
        ξ * (1 + ζ) .* cnodes[5].x +
        η * (1 + ζ) .* cnodes[6].x
    ) ./ 2.0
end

function mapping_jacobian(nodes, ::Penta6_t, ξηζ)
    ξ = ξηζ[1]
    η = ξηζ[2]
    ζ = ξηζ[3]
    M1 = nodes[1].x
    M2 = nodes[2].x
    M3 = nodes[3].x
    M4 = nodes[4].x
    M5 = nodes[5].x
    M6 = nodes[6].x
    return hcat(
        (1 - ζ) * (M2 - M1) + (1 + ζ) * (M5 - M4),
        (1 - ζ) * (M3 - M1) + (1 + ζ) * (M6 - M4),
        (1 - ξ - η) * (M4 - M1) + ξ * (M5 - M2) + η * (M6 - M3),
    ) ./ 2.0
end
