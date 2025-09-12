# COMMON
"""
    mapping(ctype::AbstractEntityType, cnodes, ξ)
    mapping(cshape::AbstractShape, cnodes, ξ)

Map the reference shape on the local shape.

# Implementation
This function must be implemented for all shape.

# `::Bar2_t`
Map the reference 2-nodes bar [-1,1] on the local bar:
```math
F(\\xi) = \\dfrac{x_r - x_l}{2} \\xi + \\dfrac{x_r + x_l}{2}
```

# `::Tri3_t`
Map the reference 3-nodes Triangle [0,1] x [0,1] on the local triangle.
```math
F(\\xi, \\eta) = (1 - \\xi - \\eta) M_1 + \\xi M_2 + \\eta M_3
```

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

# `::Pyra5_t`
Map the reference 5-nodes pyramid [-1,1] x [-1,1] x [0,1] on the 5-pyra.
See https://www.math.u-bordeaux.fr/~durufle/montjoie/pyramid.php

"""
function mapping(type_or_shape, cnodes, ξ)
    error("Function 'mapping' is not defined for $(typeof(type_or_shape))")
end
mapping(type_or_shape, cnodes) = ξ -> mapping(type_or_shape, cnodes, ξ)

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
function mapping_inv(::AbstractEntityType, cnodes, x)
    error("Function 'mapping_inv' is not defined")
end
mapping_inv(ctype::AbstractEntityType, cnodes) = x -> mapping_inv(ctype, cnodes, x)

"""
    mapping_jacobian(ctype::AbstractEntityType, cnodes, ξ)

Jacobian matrix of the mapping : ``\\dfrac{\\partial F_i}{\\partial \\xi_j}``.

# Implementation
Default version using ForwardDiff, but can be specified for each shape.

# `::Bar2_t`
Mapping's jacobian matrix for the reference 2-nodes bar [-1, 1] to the local bar.
``\\dfrac{\\partial F}{\\partial \\xi} = \\dfrac{x_r - x_l}{2}``

# `::Bar3_t`
Mapping's jacobian matrix for the reference 2-nodes bar [-1, 1] to the local bar.
``\\dfrac{\\partial F}{\\partial \\xi} = \\frac{1}{2} \\left( (2\\xi - 1) M_1 + (2\\xi + 1)M_2 - 4 \\xi M_3\\right)

# `::Tri3_t`
Mapping's jacobian matrix for the reference 3-nodes Triangle [0,1] x [0,1] to the local
triangle mapping.
```math
\\dfrac{\\partial F_i}{\\partial \\xi_j} =
\\begin{pmatrix}
    M_2 - M_1 & M_3 - M_1
\\end{pmatrix}
```

# `::Quad4_t`
Mapping's jacobian matrix for the reference square [-1,1] x [-1,1]
to the 4-quadrilateral
```math
\\frac{\\partial F}{\\partial \\xi} = -M_1 + M_2 + M_3 - M_4 + \\eta (M_1 - M_2 + M_3 - M_4)
```
```math
\\frac{\\partial F}{\\partial \\eta} = -M_1 - M_2 + M_3 + M_4 + \\xi (M_1 - M_2 + M_3 - M_4)
```
"""
function mapping_jacobian(ctype::AbstractEntityType, cnodes, ξ)
    ForwardDiff.jacobian(η -> mapping(ctype, cnodes, η), ξ)
end

"""
    mapping_jacobian_inv(ctype::AbstractEntityType, cnodes, ξ)

Inverse of the mapping jacobian matrix. This is not exactly equivalent to the `mapping_inv_jacobian` since this function is
evaluated in the reference element.

# Implementation
Default version using ForwardDiff, but can be specified for each shape.

# `::Bar2_t`
Inverse of mapping's jacobian matrix for the reference 2-nodes bar [-1, 1] to the local bar.
```math
\\dfrac{\\partial F}{\\partial \\xi}^{-1} = \\dfrac{2}{x_r - x_l}
```

# `::Bar3_t`
Inverse of mapping's jacobian matrix for the reference 2-nodes bar [-1, 1] to the local bar.
```math
\\dfrac{\\partial F}{\\partial \\xi}^{-1} = \\frac{2}{(2\\xi - 1) M_1 + (2\\xi + 1)M_2 - 4 \\xi M_3}
```

# `::Tri3_t`
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

# `::Quad4_t`
Inverse of mapping's jacobian matrix for the reference square [-1,1] x [-1,1]
to the 4-quadrilateral
"""
function mapping_jacobian_inv(ctype::AbstractEntityType, cnodes, ξ)
    inv(mapping_jacobian(ctype, cnodes, ξ))
end

"""
    mapping_inv_jacobian(ctype::AbstractEntityType, cnodes, x)

Jacobian matrix of the inverse mapping : ``\\dfrac{\\partial F_i^{-1}}{\\partial x_j}``

Contrary to `mapping_jacobian_inv`, this function is not always defined because the
inverse mapping, F^-1, is not always defined.

# Implementation
Default version using LinearAlgebra to inverse the matrix, but can be specified for each shape (if it exists).

# `::Bar2_t`
Mapping's jacobian matrix for the local bar to the reference 2-nodes bar [-1, 1].
```math
\\dfrac{\\partial F^{-1}}{\\partial x} = \\dfrac{2}{x_r - x_l}
```

# `::Tri3_t`
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
function mapping_inv_jacobian(ctype::AbstractEntityType, cnodes, x)
    inv(mapping_jacobian(ctype, cnodes, mapping_inv(ctype, cnodes, x)))
end

"""
    mapping_det_jacobian(ctype::AbstractEntityType, cnodes, ξ)
    mapping_det_jacobian(::TopologyStyle, ctype::AbstractEntityType, cnodes, ξ)

Absolute value of the determinant of the mapping Jacobian matrix, expressed in the reference element.

# Implementation
For a volumic cell (line in 1D, quad in 2D, cube in 3D), the default version uses `mapping_jacobian`,
but the function can be specified for each shape.

For a curvilinear cell (line in 2D, or in 3D), the formula is always J = ||F'|| where F is the mapping.
Hence we always fallback to the "standard" version, like for the volumic case.

Finally, the surfacic cell (quad in 3D) needs a special treatment, see [`mapping_jacobian_hypersurface`](@ref).

# `::Bar2_t`
Absolute value of the determinant of the mapping Jacobian matrix for the
reference 2-nodes bar [-1,1] to the local bar mapping.
``|det(J(\\xi))| = \\dfrac{|x_r - x_l|}{2}``

# `::Tri3_t`
Absolute value of the determinant of the mapping Jacobian matrix for the
the reference 3-nodes Triangle [0,1] x [0,1] to the local triangle mapping.

``|J| = |(x_2 - x_1) (y_3 - y_1) - (x_3 - x_1) (y_2 - y_1)|``
"""
function mapping_det_jacobian(ctype::AbstractEntityType, cnodes, ξ)
    abs(det(mapping_jacobian(ctype, cnodes, ξ)))
end

function mapping_det_jacobian(::isSurfacic, ctype, cnodes, ξ)
    jac = mapping_jacobian_hypersurface(ctype, cnodes, ξ)
    return abs(det(jac))
end

function mapping_det_jacobian(::Union{isCurvilinear, isVolumic}, ctype, cnodes, ξ)
    mapping_det_jacobian(ctype, cnodes, ξ)
end

"""
    mapping_jacobian_hypersurface(ctype, cnodes, ξ)

"Augmented" jacobian matrix of the mapping.

Let's consider a ``\\mathbb{R}^2`` surface in ``\\mathbb{R}^3``. The mapping
``F_\\Gamma(\\xi, \\eta)`` maps the reference coordinate system to the physical coordinate
system. It's jacobian ``J_\\Gamma`` is not squared. We can 'extend' this mapping to reach any point in
``\\mathbb{R}^3`` (and not only the surface) using
```math
F(\\xi, \\eta, \\zeta) = F_\\Gamma(\\xi, \\eta) + \\zeta \\nu
```
where ``\\nu`` is the conormal. Then the restriction of the squared jacobian of ``F``
to the surface is simply
```math
J|_\\Gamma = (J_\\Gamma~~\\nu)
```
"""
function mapping_jacobian_hypersurface(ctype, cnodes, ξ)
    Jref = mapping_jacobian(ctype, cnodes, ξ)
    ν = cell_normal(ctype, cnodes, ξ)
    J = hcat(Jref, ν)
    return J
end

"""
    mapping_face(cshape::AbstractShape, side)
    mapping_face(cshape::AbstractShape, side, permutation)

Build a mapping from the face reference element (corresponding to the `side`-th face of `cshape`)
to the cell reference element (i.e the `cshape`).

Build a mapping from the face reference element (corresponding to the `side`-th face of `cshape`)
to the cell reference element (i.e the `cshape`). If `permutation` is present, the mapping is built
using this permutation.
"""
function mapping_face(cshape::AbstractShape, side)
    f2n = faces2nodes(cshape, side)
    _coords = get_coords(cshape, f2n)
    fnodes = map(Node, _coords)
    return MappingFace(mapping(face_shapes(cshape, side), fnodes), nothing)
end

function mapping_face(cshape::AbstractShape, side, permutation)
    f2n = faces2nodes(cshape, side)[permutation]
    _coords = get_coords(cshape, f2n)
    fnodes = Node.(_coords)
    return MappingFace(mapping(face_shapes(cshape, side), fnodes), nothing)
end

struct MappingFace{F1, F2}
    f1::F1
    f2::F2
end
(m::MappingFace)(x) = m.f1(x)
CallableStyle(::Type{<:MappingFace}) = IsCallableStyle()

#---------------- POINT : this may seem stupid, but it is usefull for coherence
mapping(::Node_t, cnodes, ξ) = cnodes[1].x
mapping(::Point, cnodes, ξ) = mapping(Node_t(), cnodes, ξ)

#---------------- LINE
mapping(::Line, cnodes, ξ) = mapping(Bar2_t(), cnodes, ξ)

function mapping_det_jacobian(ctype::AbstractEntityType{1}, cnodes, ξ)
    norm(mapping_jacobian(ctype, cnodes, ξ))
end

function mapping(::Bar2_t, cnodes, ξ)
    (cnodes[2].x - cnodes[1].x) / 2.0 .* ξ + (cnodes[2].x + cnodes[1].x) / 2.0
end

function mapping_inv(::Bar2_t, cnodes, x)
    SA[(2 * x[1] - cnodes[2].x[1] - cnodes[1].x[1]) / (cnodes[2].x[1] - cnodes[1].x[1])]
end

function mapping_jacobian(::Bar2_t, cnodes, ξ)
    axes(cnodes) == (Base.OneTo(2),) || error("Invalid number of nodes")
    @inbounds (cnodes[2].x .- cnodes[1].x) ./ 2.0
end

function mapping_jacobian_inv(::Bar2_t, cnodes, ξ)
    @SMatrix[2.0 / (cnodes[2].x[1] .- cnodes[1].x[1])]
end

mapping_inv_jacobian(::Bar2_t, cnodes, x) = 2.0 / (cnodes[2].x[1] - cnodes[1].x[1])

mapping_det_jacobian(::Bar2_t, cnodes, ξ) = norm(cnodes[2].x - cnodes[1].x) / 2.0

function mapping(::Bar3_t, cnodes, ξ)
    ξ .* (ξ .- 1) / 2 .* cnodes[1].x .+ ξ .* (ξ .+ 1) / 2 .* cnodes[2].x .+
    (1 .- ξ) .* (1 .+ ξ) .* cnodes[3].x
end

function mapping_jacobian(::Bar3_t, cnodes, ξ)
    (cnodes[1].x .* (2 .* ξ .- 1) + cnodes[2].x .* (2 .* ξ .+ 1) - cnodes[3].x .* 4 .* ξ) ./
    2
end

function mapping_jacobian_inv(::Bar3_t, cnodes, ξ)
    2.0 /
    (cnodes[1].x .* (2 .* ξ .- 1) + cnodes[2].x .* (2 .* ξ .+ 1) - cnodes[3].x .* 4 .* ξ)
end

#---------------- TRIANGLE P1
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

function mapping_jacobian(::Tri3_t, cnodes, ξ)
    return hcat(cnodes[2].x - cnodes[1].x, cnodes[3].x - cnodes[1].x)
end

function mapping_jacobian_inv(::Tri3_t, cnodes, ξ)
    x1 = cnodes[1].x[1]
    x2 = cnodes[2].x[1]
    x3 = cnodes[3].x[1]
    y1 = cnodes[1].x[2]
    y2 = cnodes[2].x[2]
    y3 = cnodes[3].x[2]

    return SA[
        -y1+y3 x1-x3
        y1-y2 -x1+x2
    ] ./ ((x1 - x2) * (y1 - y3) - (x1 - x3) * (y1 - y2))
end

function mapping_inv_jacobian(::Tri3_t, cnodes, x)
    x1 = cnodes[1].x[1]
    x2 = cnodes[2].x[1]
    x3 = cnodes[3].x[1]
    y1 = cnodes[1].x[2]
    y2 = cnodes[2].x[2]
    y3 = cnodes[3].x[2]

    return SA[
        (y3-y1) (x1-x3)
        (y1-y2) (x2-x1)
    ] / (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
end

function mapping_det_jacobian(::Tri3_t, cnodes, ξ)
    x1 = cnodes[1].x[1]
    x2 = cnodes[2].x[1]
    x3 = cnodes[3].x[1]
    y1 = cnodes[1].x[2]
    y2 = cnodes[2].x[2]
    y3 = cnodes[3].x[2]

    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
end

#---------------- Quad P1
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

function mapping_jacobian(::Quad4_t, cnodes, ξ)
    return hcat(
        -cnodes[1].x + cnodes[2].x + cnodes[3].x - cnodes[4].x +
        ξ[2] .* (cnodes[1].x - cnodes[2].x + cnodes[3].x - cnodes[4].x),
        -cnodes[1].x - cnodes[2].x +
        cnodes[3].x +
        cnodes[4].x +
        ξ[1] .* (cnodes[1].x - cnodes[2].x + cnodes[3].x - cnodes[4].x),
    ) ./ 4
end

function mapping_jacobian_inv(::Quad4_t, cnodes::AbstractArray{<:Node{2, T}}, ξη) where {T}
    # Alias
    axes(cnodes) == (Base.OneTo(4),) || error("Invalid number of nodes")
    axes(ξη) == (Base.OneTo(2),) || error("Invalid number of coordinates")
    @inbounds begin
        x1, y1 = cnodes[1].x
        x2, y2 = cnodes[2].x
        x3, y3 = cnodes[3].x
        x4, y4 = cnodes[4].x
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

function mapping_det_jacobian(::Quad4_t, cnodes::AbstractArray{<:Node{2, T}}, ξη) where {T}
    axes(cnodes) == (Base.OneTo(4),) || error("Invalid number of nodes")
    axes(ξη) == (Base.OneTo(2),) || error("Invalid number of coordinates")
    @inbounds begin
        x1, y1 = cnodes[1].x
        x2, y2 = cnodes[2].x
        x3, y3 = cnodes[3].x
        x4, y4 = cnodes[4].x
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

#---------------- TRIANGLE P2
function mapping(::Tri6_t, cnodes, ξ)
    # Shape functions
    λ₁ = (1 - ξ[1] - ξ[2]) * (1 - 2 * ξ[1] - 2 * ξ[2])  # = (1 - x - y)(1 - 2x - 2y)
    λ₂ = ξ[1] * (2 * ξ[1] - 1) # = x (2x - 1)
    λ₃ = ξ[2] * (2 * ξ[2] - 1) # = y (2y - 1)
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

#---------------- PARAQUAD P2
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
    return sum(((λ, node),) -> λ .* get_coords(node), zip(λs, cnodes))
end

#---------------- Hexa8
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

function mapping_jacobian(::Hexa8_t, cnodes, ξηζ)
    ξ = ξηζ[1]
    η = ξηζ[2]
    ζ = ξηζ[3]
    M1 = cnodes[1].x
    M2 = cnodes[2].x
    M3 = cnodes[3].x
    M4 = cnodes[4].x
    M5 = cnodes[5].x
    M6 = cnodes[6].x
    M7 = cnodes[7].x
    M8 = cnodes[8].x
    return hcat(
        ((M2 - M1) * (1 - η) + (M3 - M4) * (1 + η)) * (1 - ζ) +
        ((M6 - M5) * (1 - η) + (M7 - M8) * (1 + η)) * (1 + ζ),
        ((M4 - M1) * (1 - ξ) + (M3 - M2) * (1 + ξ)) * (1 - ζ) +
        ((M8 - M5) * (1 - ξ) + (M7 - M6) * (1 + ξ)) * (1 + ζ),
        ((M5 - M1) * (1 - ξ) + (M6 - M2) * (1 + ξ)) * (1 - η) +
        ((M8 - M4) * (1 - ξ) + (M7 - M3) * (1 + ξ)) * (1 + η),
    ) ./ 8.0
end

#---------------- Hexa27
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

"""
    mapping(nodes, ::Tetra4_t, ξ)

Map the reference 4-nodes Tetraahedron [0,1] x [0,1] x [0,1] on the local triangle.

```
"""
function mapping(::Tetra4_t, cnodes, ξ)
    return (1 - ξ[1] - ξ[2] - ξ[3]) .* cnodes[1].x +
           ξ[1] .* cnodes[2].x +
           ξ[2] .* cnodes[3].x +
           ξ[3] .* cnodes[4].x
end

#---------------- Penta6
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

function mapping_jacobian(::Penta6_t, cnodes, ξηζ)
    ξ = ξηζ[1]
    η = ξηζ[2]
    ζ = ξηζ[3]
    M1 = cnodes[1].x
    M2 = cnodes[2].x
    M3 = cnodes[3].x
    M4 = cnodes[4].x
    M5 = cnodes[5].x
    M6 = cnodes[6].x
    return hcat(
        (1 - ζ) * (M2 - M1) + (1 + ζ) * (M5 - M4),
        (1 - ζ) * (M3 - M1) + (1 + ζ) * (M6 - M4),
        (1 - ξ - η) * (M4 - M1) + ξ * (M5 - M2) + η * (M6 - M3),
    ) ./ 2.0
end

#---------------- Pyra5
mapping(::Pyramid, cnodes, ξ) = mapping(Pyra5_t(), cnodes, ξ)

function mapping(::Pyra5_t, cnodes, ξηζ)
    ξ = ξηζ[1]
    η = ξηζ[2]
    ζ = ξηζ[3]

    # to avoid a singularity in z = 1, we replace (1-ζ) (which is always a
    # positive quantity), by (1 + ε - ζ).
    ε = eps()
    return (
        (1 - ξ - ζ) * (1 - η - ζ) / (4 * (1 + ε - ζ)) * cnodes[1].x +
        (1 + ξ - ζ) * (1 - η - ζ) / (4 * (1 + ε - ζ)) * cnodes[2].x +
        (1 + ξ - ζ) * (1 + η - ζ) / (4 * (1 + ε - ζ)) * cnodes[3].x +
        (1 - ξ - ζ) * (1 + η - ζ) / (4 * (1 + ε - ζ)) * cnodes[4].x +
        ζ * cnodes[5].x
    )
end

"""
    normal(ctype::AbstractEntityType, cnodes, iside, ξ)
    normal(::TopologyStyle, ctype::AbstractEntityType, cnodes, iside, ξ)

Normal vector of the `iside`th face of a cell, evaluated at position `ξ` in the face reference element.
So for the normal vector of the face of triangle living in a 3D space, `ξ` will be 1D (because the face
is a line, which 1D).

Beware this function needs the nodes `cnodes` and the type `ctype` of the cell (and not of the face).

TODO: If `iside` is positive, then the outward normal (with respect to the cell) is returned, otherwise
the inward normal is returned.

# `::isCurvilinear`
Note that the "face" normal vector of a curve is the "direction" vector at the given extremity.

# `::isVolumic`
``n^{loc} = J^{-\\intercal} n^{ref}``

"""
function normal(ctype::AbstractEntityType, cnodes, iside, ξ)
    normal(topology_style(ctype, cnodes), ctype, cnodes, iside, ξ)
end

function normal(::isCurvilinear, ctype::AbstractEntityType, cnodes, iside, ξ)
    # mapping face-reference-element (here, a node) to cell-reference-element (here, a Line)
    # Since a Line has always only two nodes, the node is necessary the `iside`-th
    ξ_cell = get_coords(Line())[iside]

    return normalize(mapping_jacobian(ctype, cnodes, ξ_cell) .* normal(shape(ctype), iside))
end

function normal(::isSurfacic, ctype::AbstractEntityType, cnodes, iside, ξ)
    # Get cell shape and face type and shape
    cshape = shape(ctype)
    ftype = facetypes(ctype)[iside]

    # Get face nodes
    fnodes = map(i -> cnodes[i], faces2nodes(ctype)[iside])

    # Get face direction vector (face Jacobian)
    u = mapping_jacobian(ftype, fnodes, ξ)

    # Get face parametrization function
    fp = mapping_face(cshape, iside) # mapping face-ref -> cell-ref

    # Compute surface jacobian
    J = mapping_jacobian(ctype, cnodes, fp(ξ))

    # Compute vector that will help orient outward normal
    orient = mapping(ctype, cnodes, fp(ξ)) - center(ctype, cnodes)

    # Normal direction
    n = J[:, 1] × J[:, 2] × u

    # Orient normal outward and normalize
    return normalize(orient ⋅ n .* n)
end

function normal(::isVolumic, ctype::AbstractEntityType, cnodes, iside, ξ)
    # Cell shape
    cshape = shape(ctype)

    # Face parametrization to send ξ from ref-face-element to the ref-cell-element
    fp = mapping_face(cshape, iside) # mapping face-ref -> cell-ref

    # Inverse of the Jacobian matrix (but here `y` is in the cell-reference element)
    # Warning : do not use `mapping_inv_jacobian` which requires the knowledge of `mapping_inv` (useless here)
    Jinv(y) = mapping_jacobian_inv(ctype, cnodes, y)

    return normalize(transpose(Jinv(fp(ξ))) * normal(cshape, iside))
end

"""
    cell_normal(ctype::AbstractEntityType, cnodes, ξ) where {T, N}

Compute the cell normal vector of an entity of topology dimension equals to (n-1) in a n-D space,
for instance a curve in a 2D space. This vector is expressed in the cell-reference coordinate system.

Do not confuse the cell normal vector with the cell-side (i.e face) normal vector.

# Topology dimension 1
the curve direction vector, u, is J/||J||. Then n = [-u.y, u.x].

"""
function cell_normal(
    ctype::AbstractEntityType{1},
    cnodes::AbstractArray{Node{2, T}, N},
    ξ,
) where {T, N}
    Jref = mapping_jacobian(ctype, cnodes, ξ)
    return normalize(SA[-Jref[2], Jref[1]])
end

function cell_normal(
    ctype::AbstractEntityType{2},
    cnodes::AbstractArray{Node{3, T}, N},
    ξ,
) where {T, N}
    J = mapping_jacobian(ctype, cnodes, ξ)
    return normalize(J[:, 1] × J[:, 2])
end

"""
    center(ctype::AbstractEntityType, cnodes)

Return the center of the `AbstractEntityType` by mapping the center of the corresponding `Shape`.

# Warning
Do not use this function on a face of a cell : since the face is of dimension "n-1", the mapping
won't be appropriate.
"""
center(ctype::AbstractEntityType, cnodes) = mapping(ctype, cnodes, center(shape(ctype)))

"""
    get_cell_centers(mesh::Mesh)

Get mesh cell centers coordinates (assuming perfectly flat cells)
"""
function get_cell_centers(mesh::Mesh)
    c2n = connectivities_indices(mesh, :c2n)
    celltypes = cells(mesh)
    centers = map(1:ncells(mesh)) do icell
        ctype = celltypes[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        center(ctype, cnodes)
    end
    return centers
end
