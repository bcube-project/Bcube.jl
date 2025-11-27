# This file gathers all Taylor-related interpolations

struct Taylor <: AbstractFunctionSpaceType end

FunctionSpace(::Val{:Taylor}, degree::Integer) = FunctionSpace(Taylor(), degree)

basis_functions_style(::FunctionSpace{<:Taylor}) = ModalBasisFunctionsStyle()

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

# Shared functions for all Taylor elements of some kind
function _scalar_shape_functions(::FunctionSpace{<:Taylor, 0}, ::AbstractShape, ξ)
    return SA[one(eltype(ξ))]
end

# Functions for Line shape
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
function ∂λξ_∂ξ(::FunctionSpace{<:Taylor, 0}, ::Val{1}, ::Line, ξ)
    return SA[zero(eltype(ξ))]
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
    _zero = zero(eltype(ξ))
    return SA[_zero _zero]
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
get_ndofs(::FunctionSpace{<:Taylor, N}, ::Line) where {N} = N + 1
get_ndofs(::FunctionSpace{<:Taylor, 0}, ::Union{Square, Triangle}) = 1
get_ndofs(::FunctionSpace{<:Taylor, 1}, ::Union{Square, Triangle}) = 3

# For Taylor base there are never any dof on vertex, edge or face
# Rq: proceeding with "@eval" rather than using `AbstractShape` helps solving ambiguities
for S in (:Line, :Triangle, :Square, :Cube, :Tetra, :Prism, :Pyramid)
    @eval idof_by_vertex(::FunctionSpace{<:Taylor}, shape::$S) =
        ntuple(i -> SA[], nvertices(shape))

    @eval idof_by_edge(::FunctionSpace{<:Taylor}, shape::$S) =
        ntuple(i -> SA[], nedges(shape))

    @eval idof_by_edge_with_bounds(::FunctionSpace{<:Taylor}, shape::$S) =
        ntuple(i -> SA[], nedges(shape))

    @eval idof_by_face(::FunctionSpace{<:Taylor}, shape::$S) =
        ntuple(i -> SA[], nfaces(shape))

    @eval idof_by_face_with_bounds(::FunctionSpace{<:Taylor}, shape::$S) =
        ntuple(i -> SA[], nfaces(shape))
end
