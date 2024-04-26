
"""
Abstract structure for the different types of function space, for
instance the Lagrange function space, the Taylor function space etc.
"""
abstract type AbstractFunctionSpaceType end

abstract type AbstractFunctionSpace{type, degree} end

"""
    get_type(::AbstractFunctionSpace{type})

Getter for the `type` of the `AbstractFunctionSpace`
"""
get_type(::AbstractFunctionSpace{type}) where {type} = type

"""
    get_degree(::AbstractFunctionSpace{type, degree}) where{type, degree}

Return the `degree` associated to the `AbstractFunctionSpace`.
"""
get_degree(::AbstractFunctionSpace{type, degree}) where {type, degree} = degree

struct FunctionSpace{type, degree} <: AbstractFunctionSpace{type, degree} end

"""
    FunctionSpace(fstype::Symbol, degree::Integer)
    FunctionSpace(fstype::AbstractFunctionSpaceType, degree::Integer)

Build a `FunctionSpace` of the designated `FunctionSpaceType` and `degree`.

# Examples
```jldoctest
julia> FunctionSpace(:Lagrange, 2)
FunctionSpace{Bcube.Lagrange{:Uniform}, 2}()
```
"""
function FunctionSpace(fstype::AbstractFunctionSpaceType, degree::Integer)
    FunctionSpace{typeof(fstype), degree}()
end

FunctionSpace(fstype::Symbol, degree::Integer) = FunctionSpace(Val(fstype), degree)

"""
    shape_functions(::AbstractFunctionSpace, ::Val{N}, shape::AbstractShape, ξ) where N
    shape_functions(::AbstractFunctionSpace, shape::AbstractShape, ξ)

Return the list of shape functions corresponding to a `FunctionSpace` and a `Shape`. `N` is the size
of the finite element space (default: `N=1` if the argument is not provided).

The result is a vector of all the shape functions evaluated at position ξ, and not a
tuple of the different shape functions. This choice is optimal for performance.

Note : `λ = ξ -> shape_functions(fs, shape, ξ); λ(ξ)[i]` is faster than `λ =shape_functions(fs, shape); λ[i](ξ)`

# Implementation
Default version, should be overriden for each concrete FunctionSpace.
"""
function shape_functions(::AbstractFunctionSpace, ::Val{N}, ::AbstractShape, ξ) where {N}
    error("Function 'shape_functions' not implemented")
end
function shape_functions(
    fs::AbstractFunctionSpace,
    n::Val{N},
    shape::AbstractShape,
) where {N}
    ξ -> shape_functions(fs, n, shape, ξ)
end

function shape_functions(fs::AbstractFunctionSpace, shape::AbstractShape, ξ)
    shape_functions(fs, Val(1), shape, ξ)
end
function shape_functions(fs::AbstractFunctionSpace, shape::AbstractShape)
    ξ -> shape_functions(fs, Val(1), shape, ξ)
end

"""
    shape_functions_vec(fs::AbstractFunctionSpace, ::Val{N}, shape::AbstractShape, ξ) where {N}

Return all the shape functions of FunctionSpace on a Shape evaluated in ξ as a vector.

`N` is the the size (number of components) of the finite element space.

---

    shape_functions_vec(fs::AbstractFunctionSpace{T,D}, n::Val{N}, shape::AbstractShape) where {T,D, N}

The shape functions are returned as a vector of functions.

# Implementation
This is implementation is not always valid, but it is for Lagrange and Taylor spaces (the only
two spaces available up to 20/01/23).
"""
function shape_functions_vec(
    fs::AbstractFunctionSpace,
    n::Val{N},
    shape::AbstractShape,
    ξ,
) where {N}
    _ndofs = ndofs(fs, shape)
    if N == 1
        return SVector{_ndofs}(
            _scalar_shape_functions(fs, shape, ξ)[idof] for idof in 1:_ndofs
        )
    else
        λs = shape_functions(fs, n, shape, ξ)
        ndofs_tot = N * _ndofs
        return SVector{ndofs_tot}(SVector{N}(λs[i, j] for j in 1:N) for i in 1:ndofs_tot)
    end
end

function shape_functions_vec(
    fs::AbstractFunctionSpace,
    n::Val{N},
    shape::AbstractShape,
) where {N}
    _ndofs = ndofs(fs, shape)
    if N == 1
        return SVector{_ndofs}(
            ξ -> _scalar_shape_functions(fs, shape, ξ)[idof] for idof in 1:_ndofs
        )
    else
        ndofs_tot = N * _ndofs
        return SVector{ndofs_tot}(
            ξ -> begin
                a = shape_functions(fs, n, shape, ξ)
                SVector{N}(a[i, j] for j in 1:N)
            end for i in 1:ndofs_tot
        )
    end
end

"""
    ∂λξ_∂ξ(::AbstractFunctionSpace, ::Val{N}, shape::AbstractShape) where N

Gradient, with respect to the reference coordinate system, of shape functions for any function space.
The result is an array whose elements are the gradient of each shape functions.
`N` is the size of the finite element space.

# Implementation

Default version using automatic differentiation. Specialize to increase performance.
"""
function ∂λξ_∂ξ end

function ∂λξ_∂ξ(fs::AbstractFunctionSpace, n::Val{N}, shape::AbstractShape) where {N}
    ξ -> ∂λξ_∂ξ(fs, n, shape, ξ)
end

# default : rely on forwarddiff
_diff(f, x::AbstractVector{<:Number}) = ForwardDiff.jacobian(f, x)
_diff(f, x::Number) = ForwardDiff.derivative(f, x)
function ∂λξ_∂ξ(fs::AbstractFunctionSpace, n::Val{1}, shape::AbstractShape, ξ)
    _diff(shape_functions(fs, n, shape), ξ)
end

# alias for scalar case
function ∂λξ_∂ξ(fs::AbstractFunctionSpace, shape::AbstractShape, ξ)
    ∂λξ_∂ξ(fs, Val(1), shape, ξ)
end
function ∂λξ_∂ξ(fs::AbstractFunctionSpace, shape::AbstractShape, ξ::AbstractVector)
    ∂λξ_∂ξ(fs, Val(1), shape, ξ)
end
function ∂λξ_∂ξ(fs::AbstractFunctionSpace, shape::AbstractShape)
    ξ -> ∂λξ_∂ξ(fs, Val(1), shape, ξ)
end

"""
    idof_by_face(::AbstractFunctionSpace, ::AbstractShape)

Return the local indices of the dofs lying on each face of the `Shape`.

Dofs lying on the face edges are excluded, only "face-interior" dofs are considered.

The result is a Tuple of arrays of integers. Arrays maybe be empty. See `Lagrange`
interpolation for simple examples.
"""
function idof_by_face(::AbstractFunctionSpace, ::AbstractShape)
    error("Function 'idof_by_face' is not defined for this FunctionSpace and Shape")
end

"""
    idof_by_face_with_bounds(::AbstractFunctionSpace, ::AbstractShape)

Return the local indices of the dofs lying on each face of the `Shape`.

Dofs lying on the face edges are included

The result is a Tuple of arrays of integers. Arrays maybe be empty. See `Lagrange`
interpolation for simple examples.
"""
function idof_by_face_with_bounds(::AbstractFunctionSpace, ::AbstractShape)
    error(
        "Function 'idof_by_face_with_bounds' is not defined for this FunctionSpace and Shape",
    )
end

"""
    idof_by_edge(::AbstractFunctionSpace, ::AbstractShape)

Return the local indices of the dofs lying on each edge of the `Shape`.

Dofs lying on the edge vertices are excluded.

The result is a Tuple of arrays of integers. Arrays maybe be empty.
See `Lagrange` interpolation for simple examples.
"""
function idof_by_edge(::AbstractFunctionSpace, ::AbstractShape)
    error("Function 'idof_by_edge' is not defined for this FunctionSpace and Shape")
end

"""
    idof_by_edge_with_bounds(::AbstractFunctionSpace, ::AbstractShape)

Return the local indices of the dofs lying on each edge of the `Shape`.

Dofs lying on the edge vertices are included.

The result is a Tuple of arrays of integers. Arrays maybe be empty.
See `Lagrange` interpolation for simple examples.
"""
function idof_by_edge_with_bounds(::AbstractFunctionSpace, ::AbstractShape)
    error(
        "Function 'idof_by_edge_with_bounds' is not defined for this FunctionSpace and Shape",
    )
end

"""
    idof_by_vertex(::AbstractFunctionSpace, ::AbstractShape)

Return the local indices of the dofs lying on each vertex of the `Shape`.

Beware that we are talking about the `Shape`, not the `EntityType`. So 'interior' vertices
of the `EntityType` are not taken into account for instance. See `Lagrange` interpolation
for simple examples.
"""
function idof_by_vertex(::AbstractFunctionSpace, ::AbstractShape)
    error("Function 'idof_by_vertex' is not defined")
end

"""
    ndofs(fs::AbstractFunctionSpace, shape::AbstractShape)

Number of dofs associated to the given interpolation.
"""
function ndofs(fs::AbstractFunctionSpace, shape::AbstractShape)
    error(
        "Function 'ndofs' is not defined for the given FunctionSpace $fs and shape $shape",
    )
end

"""
    get_coords(fs::AbstractFunctionSpace,::AbstractShape)

Return node coordinates in the reference space for associated function space and shape.
"""
function get_coords(fs::AbstractFunctionSpace, ::AbstractShape)
    error("Function 'coords' is not defined")
end

abstract type AbtractBasisFunctionsStyle end
struct NodalBasisFunctionsStyle <: AbtractBasisFunctionsStyle end
struct ModalBasisFunctionsStyle <: AbtractBasisFunctionsStyle end

"""
    basis_functions_style(fs::AbstractFunctionSpace)

Return the style (modal or nodal) corresponding to the basis functions of the 'fs'.
"""
function basis_functions_style(fs::AbstractFunctionSpace)
    error("'basis_functions_style' is not defined for type : $(typeof(fs))")
end

get_quadrature(fs::AbstractFunctionSpace) = get_quadrature(basis_functions_style(fs), fs)
function get_quadrature(::ModalBasisFunctionsStyle, fs::AbstractFunctionSpace)
    error("'get_quadrature' is not invalid for modal basis functions")
end
function get_quadrature(::NodalBasisFunctionsStyle, fs::AbstractFunctionSpace)
    error("'get_quadrature' is not defined for type : $(typeof(fs))")
end

# `collocation` does not apply to modal basis functions.
function is_collocated(::ModalBasisFunctionsStyle, fs::AbstractFunctionSpace, quad)
    IsNotCollocatedStyle()
end

function is_collocated(
    ::NodalBasisFunctionsStyle,
    fs::AbstractFunctionSpace,
    quad::AbstractQuadratureRule,
)
    return is_collocated(get_quadrature(fs), get_quadrature(quad)())
end

function is_collocated(fs::AbstractFunctionSpace, quad)
    is_collocated(basis_functions_style(fs), fs, quad)
end
