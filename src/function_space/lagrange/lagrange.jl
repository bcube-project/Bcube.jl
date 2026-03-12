# This file gathers all Lagrange-related interpolations

struct Lagrange{T} <: AbstractFunctionSpaceType end

function Lagrange(type)
    @assert type ∈ (:Uniform, :Legendre, :Lobatto) "Lagrange type=$type is not supported"
    Lagrange{type}()
end
Lagrange() = Lagrange(:Uniform) # default Lagrange constructor when `type` is not prescribed

FunctionSpace(::Val{:Lagrange}, degree::Integer) = FunctionSpace(Lagrange(), degree)

lagrange_quadrature_type(::Lagrange{T}) where {T} = error("No quadrature type is associated with Lagrange type $T")
lagrange_quadrature_type(::Lagrange{:Uniform})    = QuadratureUniform()
lagrange_quadrature_type(::Lagrange{:Legendre})   = QuadratureLegendre()
lagrange_quadrature_type(::Lagrange{:Lobatto})    = QuadratureLobatto()

function lagrange_quadrature_type(fs::FunctionSpace{<:Lagrange})
    lagrange_quadrature_type(get_type(fs)())
end

function lagrange_quadrature(fs::FunctionSpace{<:Lagrange})
    return Quadrature(lagrange_quadrature_type(fs), get_degree(fs))
end

get_quadrature(fs::FunctionSpace{<:Lagrange}) = lagrange_quadrature(fs)
function get_quadrature(::NodalBasisFunctionsStyle, fs::FunctionSpace{<:Lagrange})
    return lagrange_quadrature(fs)
end

#default:
function is_collocated(
    ::ModalBasisFunctionsStyle,
    ::FunctionSpace{<:Lagrange},
    ::Quadrature,
)
    IsNotCollocatedStyle()
end

function is_collocated(
    ::NodalBasisFunctionsStyle,
    fs::FunctionSpace{<:Lagrange},
    quad::Quadrature,
)
    return is_collocated(lagrange_quadrature(fs), quad)
end

basis_functions_style(::FunctionSpace{<:Lagrange}) = NodalBasisFunctionsStyle()

"""
    shape_functions(::FunctionSpace{<:Lagrange}, :: Val{N}, ::AbstractShape, ξ) where {N}

# Implementation
For N > 1, the default version consists in "replicating" the shape functions.
If `shape_functions` returns the vector `[λ₁; λ₂; λ₃]`, and if the `FESpace` is of size `2`,
then this default behaviour consists in returning the matrix `[λ₁ 0; λ₂ 0; λ₃ 0; 0 λ₁; 0 λ₂; 0 λ₃]`.

# Triangle
## Order 1
```math
\\hat{\\lambda}_1(\\xi, \\eta) = 1 - \\xi - \\eta \\hspace{1cm}
\\hat{\\lambda}_2(\\xi, \\eta) = \\xi                \\hspace{1cm}
\\hat{\\lambda}_3(\\xi, \\eta) = \\eta
```

## Order 2
```math
\\begin{aligned}
    & \\hat{\\lambda}_1(\\xi, \\eta) = (1 - \\xi - \\eta)(1 - 2 \\xi - 2 \\eta) \\\\
    & \\hat{\\lambda}_2(\\xi, \\eta) = \\xi (2\\xi - 1) \\\\
    & \\hat{\\lambda}_3(\\xi, \\eta) = \\eta (2\\eta - 1) \\\\
    & \\hat{\\lambda}_{12}(\\xi, \\eta) = 4 \\xi (1 - \\xi - \\eta) \\\\
    & \\hat{\\lambda}_{23}(\\xi, \\eta) = 4 \\xi \\eta \\\\
    & \\hat{\\lambda}_{31}(\\xi, \\eta) = 4 \\eta (1 - \\xi - \\eta)
\\end{aligned}
```

# Tetra

## Order 1
```math
\\hat{\\lambda}_1(\\xi, \\eta, \\zeta) = (1 - \\xi - \\eta - \\zeta) \\hspace{1cm}
\\hat{\\lambda}_2(\\xi, \\eta, \\zeta) = \\xi                        \\hspace{1cm}
\\hat{\\lambda}_3(\\xi, \\eta, \\zeta) = \\eta                       \\hspace{1cm}
\\hat{\\lambda}_5(\\xi, \\eta, \\zeta) = \\zeta                      \\hspace{1cm}
```

# Prism
## Order 1
```math
\\begin{aligned}
    \\hat{\\lambda}_1(\\xi, \\eta, \\zeta) = (1 - \\xi - \\eta)(1 - \\zeta)/2 \\hspace{1cm}
    \\hat{\\lambda}_2(\\xi, \\eta, \\zeta) = \\xi (1 - \\zeta)/2          \\hspace{1cm}
    \\hat{\\lambda}_3(\\xi, \\eta, \\zeta) = \\eta (1 - \\zeta)/2  \\hspace{1cm}
    \\hat{\\lambda}_5(\\xi, \\eta, \\zeta) = (1 - \\xi - \\eta)(1 + \\zeta)/2 \\hspace{1cm}
    \\hat{\\lambda}_6(\\xi, \\eta, \\zeta) = \\xi (1 + \\zeta)/2          \\hspace{1cm}
    \\hat{\\lambda}_7(\\xi, \\eta, \\zeta) = \\eta (1 + \\zeta)/2  \\hspace{1cm}
\\end{aligned}
```
"""
function shape_functions(
    fs::FunctionSpace{<:Lagrange, D},
    ::Val{N},
    shape::AbstractShape,
    ξ,
) where {D, N}
    if N == 1
        return _scalar_shape_functions(fs, shape, ξ)
    elseif N < MAX_LENGTH_STATICARRAY
        return kron(SMatrix{N, N}(1I), _scalar_shape_functions(fs, shape, ξ))
    else
        return kron(Diagonal([1.0 for i in 1:N]), _scalar_shape_functions(fs, shape, ξ))
    end
end
function shape_functions(
    fs::FunctionSpace{<:Lagrange, D},
    n::Val{N},
    shape::AbstractShape,
) where {D, N}
    ξ -> shape_functions(fs, n, shape, ξ)
end

"""
    ∂λξ_∂ξ(::FunctionSpace{<:Lagrange}, ::Val{1}, ::AbstractShape, ξ)

# Triangle
## Order 0
```math
\\nabla \\hat{\\lambda}(\\xi, \\eta) =
\\begin{pmatrix}
    0 \\\\ 0
\\end{pmatrix}
```

## Order 1
```math
\\begin{aligned}
    & \\nabla \\hat{\\lambda}_1(\\xi, \\eta) =
        \\begin{pmatrix}
            -1 \\\\ -1
        \\end{pmatrix} \\\\
    & \\nabla \\hat{\\lambda}_2(\\xi, \\eta) =
        \\begin{pmatrix}
            1 \\\\ 0
        \\end{pmatrix} \\\\
    & \\nabla \\hat{\\lambda}_3(\\xi, \\eta) =
        \\begin{pmatrix}
            0 \\\\ 1
        \\end{pmatrix} \\\\
\\end{aligned}
```

## Order 2
```math
\\begin{aligned}
    & \\nabla \\hat{\\lambda}_1(\\xi, \\eta) =
        \\begin{pmatrix}
            -3 + 4 (\\xi + \\eta) \\\\ -3 + 4 (\\xi + \\eta)
        \\end{pmatrix} \\\\
    & \\nabla \\hat{\\lambda}_2(\\xi, \\eta) =
        \\begin{pmatrix}
            -1 + 4 \\xi \\\\ 0
        \\end{pmatrix} \\\\
    & \\nabla \\hat{\\lambda}_3(\\xi, \\eta) =
        \\begin{pmatrix}
            0 \\\\ -1 + 4 \\eta
        \\end{pmatrix} \\\\
    & \\nabla \\hat{\\lambda}_{12}(\\xi, \\eta) =
        4 \\begin{pmatrix}
            1 - 2 \\xi - \\eta \\\\ - \\xi
        \\end{pmatrix} \\\\
    & \\nabla \\hat{\\lambda}_{23}(\\xi, \\eta) =
        4 \\begin{pmatrix}
            \\eta \\\\ \\xi
        \\end{pmatrix} \\\\
    & \\nabla \\hat{\\lambda}_{31}(\\xi, \\eta) =
        4 \\begin{pmatrix}
            - \\eta \\\\ 1 - 2 \\eta - \\xi
        \\end{pmatrix} \\\\
\\end{aligned}
```

# Square
## Order 0
```math
\\nabla \\hat{\\lambda}(\\xi, \\eta) =
\\begin{pmatrix}
    0 \\\\ 0
\\end{pmatrix}
```

"""
# function ∂λξ_∂ξ(fs::FunctionSpace{<:Lagrange}, ::Val{1}, ::AbstractShape, ξ::Number)
#     error("∂λξ_∂ξ not implemented for fs = $fs")
# end