# This file gathers all Taylor-related interpolations

struct Taylor <: AbstractFunctionSpaceType end

FunctionSpace(::Val{:Taylor}, degree::Integer) = FunctionSpace(Taylor(), degree)

basis_functions_style(::FunctionSpace{<:Taylor}) = ModalBasisFunctionsStyle()

function shape_functions(
    fs::FunctionSpace{<:Taylor},
    ::Val{N},
    shape::AbstractShape,
    ξ,
) where {N} #::Union{T,AbstractVector{T}} ,T<:Number
    if N == 1
        return _scalar_shape_functions(fs, shape, ξ)
    elseif N < MAX_LENGTH_STATICARRAY
        return kron(SMatrix{N, N}(1I), _scalar_shape_functions(fs, shape, ξ))
    else
        return kron(Diagonal([1.0 for i in 1:N]), _scalar_shape_functions(fs, shape, ξ))
    end
end
function shape_functions(
    fs::FunctionSpace{<:Taylor, D},
    n::Val{N},
    shape::AbstractShape,
) where {D, N}
    ξ -> shape_functions(fs, n, shape, ξ)
end

"""
    shape_functions(::FunctionSpace{<:Taylor}, ::AbstractShape, ξ)

# Implementation
For N > 1, the default version consists in "replicating" the shape functions.
If `shape_functions` returns the vector `[λ₁; λ₂; λ₃]`, and if the `FESpace` is of size `2`,
then this default behaviour consists in returning the matrix `[λ₁ 0; λ₂ 0; λ₃ 0; 0 λ₁; 0 λ₂; 0 λ₃]`.

# Any shape, order 0
``\\hat{\\lambda}(\\xi) = 1``

# Line
## Order 1
```math
\\hat{\\lambda}_1(\\xi) = 1 \\hspace{1cm} \\hat{\\lambda}_1(\\xi) = \\frac{\\xi}{2}
```

# Square
## Order 1
```math
\\begin{aligned}
    & \\hat{\\lambda}_1(\\xi, \\eta) = 0 \\\\
    & \\hat{\\lambda}_2(\\xi, \\eta) = \\frac{\\xi}{2} \\\\
    & \\hat{\\lambda}_3(\\xi, \\eta) = \\frac{\\eta}{2}
\\end{aligned}
```
"""
function _doc_shape_functions_taylor end

"""
    ∂λξ_∂ξ(::FunctionSpace{<:Taylor}, ::Val{1}, ::AbstractShape, ξ)

# Line
## Order 0
``\\nabla \\hat{\\lambda}(\\xi) = 0``

## Order 1
```math
\\nabla \\hat{\\lambda}_1(\\xi) = 0 \\hspace{1cm} \\nabla \\hat{\\lambda}_1(\\xi) = \\frac{1}{2}
```

# Square
## Order 0
```math
\\hat{\\lambda}_1(\\xi, \\eta) = \\begin{pmatrix} 0 \\\\ 0 \\end{pmatrix}
```

## Order 1
```math
\\begin{aligned}
    & \\nabla \\hat{\\lambda}_1(\\xi, \\eta) = \\begin{pmatrix} 0 \\\\ 0 \\end{pmatrix} \\\\
    & \\nabla \\hat{\\lambda}_2(\\xi, \\eta) = \\begin{pmatrix} \\frac{1}{2} \\\\ 0 \\end{pmatrix} \\\\
    & \\nabla \\hat{\\lambda}_3(\\xi, \\eta) = \\begin{pmatrix} 0 \\\\ \\frac{1}{2} \\end{pmatrix}
\\end{aligned}
```
"""
function _doc_∂λξ_∂ξ_taylor end

# Shared functions for all Taylor elements of some kind
function _scalar_shape_functions(::FunctionSpace{<:Taylor, 0}, ::AbstractShape, ξ)
    return SA[1.0]
end

# Functions for Line shape
function ∂λξ_∂ξ(::FunctionSpace{<:Taylor, 0}, ::Val{1}, ::Line, ξ)
    return SA[0.0]
end

function _scalar_shape_functions(::FunctionSpace{<:Taylor, 1}, ::Line, ξ)
    return SA[
        1.0
        ξ[1] / 2
    ]
end

function ∂λξ_∂ξ(::FunctionSpace{<:Taylor, 1}, ::Val{1}, ::Line, ξ)
    return SA[
        0.0
        1.0 / 2.0
    ]
end

# Functions for Square shape
function ∂λξ_∂ξ(::FunctionSpace{<:Taylor, 0}, ::Val{1}, ::Union{Square, Triangle}, ξ)
    return SA[0.0 0.0]
end

function _scalar_shape_functions(::FunctionSpace{<:Taylor, 1}, ::Square, ξ)
    return SA[
        1.0
        ξ[1] / 2
        ξ[2] / 2
    ]
end

function ∂λξ_∂ξ(::FunctionSpace{<:Taylor, 1}, ::Val{1}, ::Square, ξ)
    return SA[
        0.0 0.0
        1.0/2 0.0
        0.0 1.0/2
    ]
end

# Number of dofs
ndofs(::FunctionSpace{<:Taylor, N}, ::Line) where {N} = N + 1
ndofs(::FunctionSpace{<:Taylor, 0}, ::Union{Square, Triangle}) = 1
ndofs(::FunctionSpace{<:Taylor, 1}, ::Union{Square, Triangle}) = 3

# For Taylor base there are never any dof on vertex, edge or face
function idof_by_vertex(::FunctionSpace{<:Taylor, N}, shape::AbstractShape) where {N}
    fill(Int[], nvertices(shape))
end

function idof_by_edge(::FunctionSpace{<:Taylor, N}, shape::AbstractShape) where {N}
    ntuple(i -> SA[], nedges(shape))
end
function idof_by_edge_with_bounds(
    ::FunctionSpace{<:Taylor, N},
    shape::AbstractShape,
) where {N}
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_face(::FunctionSpace{<:Taylor, N}, shape::AbstractShape) where {N}
    ntuple(i -> SA[], nfaces(shape))
end
function idof_by_face_with_bounds(
    ::FunctionSpace{<:Taylor, N},
    shape::AbstractShape,
) where {N}
    ntuple(i -> SA[], nfaces(shape))
end
