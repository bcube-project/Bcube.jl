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

function _lagrange_poly(j, ξ, nodes)
    coef = 1.0 / prod((nodes[j] - nodes[k]) for k in eachindex(nodes) if k ≠ j)
    return coef * prod((ξ - nodes[k]) for k in eachindex(nodes) if k ≠ j)
end

function __shape_functions_symbolic(quadrule::AbstractQuadratureRule, ::Type{T}) where {T}
    nodes = get_nodes(quadrule)
    @variables ξ::eltype(T)
    l = [_lagrange_poly(j, ξ, nodes) for j in eachindex(nodes)]
    expr_l = Symbolics.toexpr.((l))
    return :(SA[$(expr_l...)])
end

@generated function _shape_functions_symbolic(
    ::Line,
    ::Q,
    ξ::T,
) where {Q <: AbstractQuadrature, T}
    quadrule = QuadratureRule(Line(), Q())
    __shape_functions_symbolic(quadrule, T)
end

"""
    shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, ::Shape, ξ) where {D, Shape<:Line}
    ∂λξ_∂ξ_symbolic(fs::FunctionSpace{<:Lagrange, D}, ::Shape, ξ) where {D, Shape<:Line}

# Implementation
Based on `Symbolic.jl`. First tests show that this version is slower than the implementation based on `meta`
when `D` is greater. Further investigations are needed to understand this behavior.

`shape_functions_symbolic` uses a "generated" function named `_shape_functions_symbolic`. The core of the
generated function is an `Expression` that is created by `__shape_functions_symbolic`. This latter function
uses the `Symbolics` package and the lagrange polynomials (defined in `_lagrange_poly`).
"""
function shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, ::Line, ξ) where {D}
    quadtype = lagrange_quadrature_type(fs)
    quad = Quadrature(quadtype, Val(D))
    _shape_functions_symbolic(Line(), quad, ξ[1])
end

function _shape_functions_symbolic_square(
    l1::SVector{N1, T},
    l2::SVector{N2, T},
) where {N1, N2, T}
    N = N1 * N2
    if N < MAX_LENGTH_STATICARRAY
        return SVector{N, T}(l1[i] * l2[j] for i in 1:N1, j in 1:N2)
    else
        _l1 = Array(l1)
        _l2 = Array(l2)
        return vec([i * j for i in _l1, j in _l2])
    end
end

function shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, ::Square, ξ) where {D}
    l1 = shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, Line(), ξ[1])
    l2 = shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, Line(), ξ[2])
    return _shape_functions_symbolic_square(l1, l2)
end

function shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, ::Cube, ξ) where {D}
    l1 = shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, Line(), ξ[1])
    l2 = shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, Line(), ξ[2])
    l3 = shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, Line(), ξ[3])
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(Cube(), Quadrature(quadtype, Val(D)))
    N = length(quadrule)
    if N < MAX_LENGTH_STATICARRAY
        return SVector{N}(i * j * k for i in l1, j in l2, k in l3)
    else
        _l1 = Array(l1)
        _l2 = Array(l2)
        _l3 = Array(l3)
        return vec([i * j * k for i in _l1, j in _l2, k in _l3])
    end
end

# ######### NOT USED ##########
function __∂λξ_∂ξ_symbolic(
    ::Line,
    ::Val{D},
    qt::AbstractQuadratureType,
    ::Type{T},
) where {D, T}
    quadrule = QuadratureRule(Line(), Quadrature(qt, Val(D)))
    nodes = get_nodes(quadrule)
    @variables ξ::eltype(T)
    l = [_lagrange_poly(j, ξ, nodes) for j in eachindex(nodes)]
    Diff = Differential(ξ)
    dl = expand_derivatives.(Diff.(l))
    expr_l = Symbolics.toexpr.(simplify.(dl))
    return :(SA[$(expr_l...)])
end

@generated function _∂λξ_∂ξ_symbolic(
    ::Line,
    ::Val{D},
    qt::AbstractQuadratureType,
    ξ::T,
) where {D, T}
    __∂λξ_∂ξ_symbolic(Line(), Val(D), qt, T)
end

function ∂λξ_∂ξ_symbolic(fs::FunctionSpace{<:Lagrange, D}, ::Line, ξ) where {D}
    quadtype = lagrange_quadrature_type(fs)
    _∂λξ_∂ξ_symbolic(Line(), Val(D), quadtype, ξ)
end
# #############################

function get_ndofs(fs::FunctionSpace{<:Lagrange, D}, shape::AbstractShape) where {D}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(D)))
    return length(quadrule)
end

function _scalar_shape_functions(
    fs::FunctionSpace{<:Lagrange, D},
    shape::AbstractShape,
    ξ,
) where {D}
    shape_functions_symbolic(fs, shape, ξ)
end

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

# Lagrange function for degree = 0
_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Line, ξ) = SA[one(eltype(ξ))]
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Line) = 1

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
function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Line, ξ::Number)
    SA[zero(ξ)]
end

# Functions for Triangle shape
_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Triangle, ξ) = SA[one(eltype(ξ))]
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Triangle) = 1

function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Triangle, ξ)
    _zero = zero(eltype(ξ))
    return SA[_zero _zero]
end

function _scalar_shape_functions(::FunctionSpace{<:Lagrange, 1}, ::Triangle, ξ)
    return SA[
        1 - ξ[1] - ξ[2]
        ξ[1]
        ξ[2]
    ]
end

function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 1}, ::Val{1}, ::Triangle, ξ)
    return SA[
        -1.0 -1.0
        1.0 0.0
        0.0 1.0
    ]
end

get_ndofs(::FunctionSpace{<:Lagrange, 1}, ::Triangle) = 3

function _scalar_shape_functions(::FunctionSpace{<:Lagrange, 2}, ::Triangle, ξ)
    return SA[
        (1 - ξ[1] - ξ[2]) * (1 - 2ξ[1] - 2ξ[2])  # = (1 - x - y)(1 - 2x - 2y)
        ξ[1] * (2ξ[1] - 1)                 # = x (2x - 1)
        ξ[2] * (2ξ[2] - 1)                 # = y (2y - 1)
        4ξ[1] * (1 - ξ[1] - ξ[2])            # = 4x (1 - x - y)
        4ξ[1] * ξ[2]
        4ξ[2] * (1 - ξ[1] - ξ[2])             # = 4y (1 - x - y)
    ]
end

function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 2}, ::Val{1}, ::Triangle, ξ)
    return SA[
        -3+4(ξ[1] + ξ[2]) -3+4(ξ[1] + ξ[2])
        -1+4ξ[1] 0.0
        0.0 -1+4ξ[2]
        4(1 - 2ξ[1] - ξ[2]) -4ξ[1]
        4ξ[2] 4ξ[1]
        -4ξ[2] 4(1 - 2ξ[2] - ξ[1])
    ]
end

get_ndofs(::FunctionSpace{<:Lagrange, 2}, ::Triangle) = 6

function _scalar_shape_functions(::FunctionSpace{<:Lagrange, 3}, ::Triangle, ξ)
    λ1 = 1 - ξ[1] - ξ[2]
    λ2 = ξ[1]
    λ3 = ξ[2]
    return SA[
        0.5 * (3 * λ1 - 1) * (3 * λ1 - 2) * λ1
        0.5 * (3 * λ2 - 1) * (3 * λ2 - 2) * λ2
        0.5 * (3 * λ3 - 1) * (3 * λ3 - 2) * λ3
        4.5 * λ1 * λ2 * (3 * λ1 - 1)
        4.5 * λ1 * λ2 * (3 * λ2 - 1)
        4.5 * λ2 * λ3 * (3 * λ2 - 1)
        4.5 * λ2 * λ3 * (3 * λ3 - 1)
        4.5 * λ3 * λ1 * (3 * λ3 - 1)
        4.5 * λ3 * λ1 * (3 * λ1 - 1)
        27 * λ1 * λ2 * λ3
    ]
end

get_ndofs(::FunctionSpace{<:Lagrange, 3}, ::Triangle) = 10

# Functions for Square shape
_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Square, ξ) = SA[one(eltype(ξ))]
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Square) = 1

function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Square, ξ)
    _zero = zero(eltype(ξ))
    return SA[_zero _zero]
end

# Tetra
_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Tetra, ξ) = SA[one(eltype(ξ))]
function grad_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Tetra, ξ)
    _zero = zero(eltype(ξ))
    return SA[_zero _zero _zero]
end
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Tetra) = 1

"""
    shape_functions(::FunctionSpace{<:Lagrange, 1}, ::Tetra, ξ)

Shape functions for Tetra Lagrange element of degree 1 in a 3D space.

```math
\\hat{\\lambda}_1(\\xi, \\eta, \\zeta) = (1 - \\xi - \\eta - \\zeta) \\hspace{1cm}
\\hat{\\lambda}_2(\\xi, \\eta, \\zeta) = \\xi                        \\hspace{1cm}
\\hat{\\lambda}_3(\\xi, \\eta, \\zeta) = \\eta                       \\hspace{1cm}
\\hat{\\lambda}_5(\\xi, \\eta, \\zeta) = \\zeta                      \\hspace{1cm}
```
"""
function _scalar_shape_functions(::FunctionSpace{<:Lagrange, 1}, ::Tetra, ξηζ)
    ξ, η, ζ = ξηζ
    return SA[
        1 - ξ - η - ζ
        ξ
        η
        ζ
    ]
end
function grad_shape_functions(::FunctionSpace{<:Lagrange, 1}, ::Val{1}, ::Tetra, ξηζ)
    return SA[
        -1 -1 -1
        1 0 0
        0 1 0
        0 0 1
    ]
end
get_ndofs(::FunctionSpace{<:Lagrange, 1}, ::Tetra) = 4

# Functions for Cube shape
_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Cube, ξ) = SA[one(eltype(ξ))]
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Cube) = 1
function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Cube, ξ)
    _zero = zero(eltype(ξ))
    SA[_zero _zero _zero]
end

# Prism : Lagrange prismatic elements are built from the cartesian product of the Triangle and
# the Line
function _scalar_shape_functions(fs::FunctionSpace{<:Lagrange, N}, ::Prism, ξηζ) where {N}
    ξ, η, ζ = ξηζ
    λ_tri = _scalar_shape_functions(fs, Triangle(), SA[ξ, η])
    λ_line = _scalar_shape_functions(fs, Line(), SA[ζ])
    return SVector{length(λ_tri) * length(λ_line)}(λt * λl for λt in λ_tri, λl in λ_line)
end

function get_ndofs(fs::FunctionSpace{<:Lagrange, N}, ::Prism) where {N}
    get_ndofs(fs, Triangle()) * get_ndofs(fs, Line())
end

# Pyramid
_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Pyramid, ξ) = SA[one(eltype(ξ))]
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Pyramid) = 1

function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Pyramid, ξ)
    _zero = zero(eltype(ξ))
    SA[_zero _zero _zero]
end

function _scalar_shape_functions(::FunctionSpace{<:Lagrange, 1}, ::Pyramid, ξηζ)
    ξ = ξηζ[1]
    η = ξηζ[2]
    ζ = ξηζ[3]

    # to avoid a singularity in z = 1, we replace (1-ζ) (which is always a
    # positive quantity), by (1 + ε - ζ).
    ε = eps()
    return SA[
        (1 - ξ - ζ) * (1 - η - ζ) / (4 * (1 + ε - ζ))
        (1 + ξ - ζ) * (1 - η - ζ) / (4 * (1 + ε - ζ))
        (1 + ξ - ζ) * (1 + η - ζ) / (4 * (1 + ε - ζ))
        (1 - ξ - ζ) * (1 + η - ζ) / (4 * (1 + ε - ζ))
        ζ
    ]
end

get_ndofs(::FunctionSpace{<:Lagrange, 1}, ::Pyramid) = 5

# bmxam/remark : I first tried to write "generic" functions covering multiple case. But it's easy to
# forget some case and think they are already covered. In the end, it's easier to write them all explicitely,
# except for the one I wrote that are very intuitive (at least for me).
# Nevertheless, it is indeed possible to write functions working for every Lagrange element:
# for degree > 1, one dof per vertex
# for degree > 2, (degree - 1) dof per edge
# (...)
# Moreover, it's easier to gather them all in one place instead of mixing them with `shape_functions` definitions

# Generic rule 1 : no dof per vertex, edge or face for any Lagrange element of degree 0
for S in (:Line, :Triangle, :Square, :Cube, :Tetra, :Prism, :Pyramid)
    @eval idof_by_vertex(::FunctionSpace{<:Lagrange, 0}, shape::$S) =
        ntuple(i -> SA[], nvertices(shape))
    @eval idof_by_edge(::FunctionSpace{<:Lagrange, 0}, shape::$S) =
        ntuple(i -> SA[], nedges(shape))
    @eval idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 0}, shape::$S) =
        ntuple(i -> SA[], nedges(shape))
    @eval idof_by_face(::FunctionSpace{<:Lagrange, 0}, shape::$S) =
        ntuple(i -> SA[], nfaces(shape))
    @eval idof_by_face_with_bounds(::FunctionSpace{<:Lagrange, 0}, shape::$S) =
        ntuple(i -> SA[], nfaces(shape))
end

# Generic rule 2 (for non tensorial elements): one dof per vertex for any Lagrange element of degree > 0
function idof_by_vertex(
    ::FunctionSpace{<:Lagrange, degree},
    shape::AbstractShape,
) where {degree}
    ntuple(i -> SA[i], nvertices(shape))
end

function _idof_by_vertex(fs::FunctionSpace{<:Lagrange, degree}, shape::Line) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n = _get_num_nodes_per_dim(quadrule)
    return :(SA[1], SA[$n])
end

function _idof_by_vertex(
    fs::FunctionSpace{<:Lagrange, degree},
    shape::Square,
) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n1, n2 = _get_num_nodes_per_dim(quadrule)
    I = LinearIndices((1:n1, 1:n2))
    p1, p2, p3, p4 = I[1, 1], I[n1, 1], I[n1, n2], I[1, n2]
    return :(SA[$p1], SA[$p2], SA[$p3], SA[$p4])
end

function _idof_by_vertex(fs::FunctionSpace{<:Lagrange, degree}, shape::Cube) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n1, n2, n3 = _get_num_nodes_per_dim(quadrule)
    I = LinearIndices((1:n1, 1:n2, 1:n3))
    p1, p2, p3, p4 = I[1, 1, 1], I[n1, 1, 1], I[n1, n2, 1], I[1, n2, 1]
    p5, p6, p7, p8 = I[1, 1, n3], I[n1, 1, n3], I[n1, n2, n3], I[1, n2, n3]
    return :(SA[$p1], SA[$p2], SA[$p3], SA[$p4], SA[$p5], SA[$p6], SA[$p7], SA[$p8])
end

function idof_by_vertex(::FunctionSpace{<:Lagrange, 0}, shape::Union{Line, Square, Cube})
    ntuple(i -> SA[], nvertices(shape))
end
@generated function idof_by_vertex(
    fs::FunctionSpace{<:Lagrange},
    shape::Union{Line, Square, Cube},
)
    return _idof_by_vertex(fs(), shape())
end

# Generic rule 3 : for `AbstractShape` of topology dimension less than `3`, `idof_by_face` alias `idof_by_edge`
const AbstractShape_1_2 = Union{AbstractShape{1}, AbstractShape{2}}
function idof_by_face(
    fs::FunctionSpace{type, degree},
    shape::AbstractShape_1_2,
) where {type, degree}
    idof_by_edge(fs, shape)
end
function idof_by_face_with_bounds(
    fs::FunctionSpace{type, degree},
    shape::AbstractShape_1_2,
) where {type, degree}
    idof_by_edge_with_bounds(fs, shape)
end

# Line
function idof_by_edge(::FunctionSpace{<:Lagrange, degree}, shape::Line) where {degree}
    ntuple(i -> SA[], nedges(shape))
end
function idof_by_edge_with_bounds(
    fs::FunctionSpace{<:Lagrange, degree},
    shape::Line,
) where {degree}
    idof_by_vertex(fs, shape)
end

# Triangle
function idof_by_edge(::FunctionSpace{<:Lagrange, 1}, shape::Triangle)
    ntuple(i -> SA[], nedges(shape))
end
function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 1}, shape::Triangle)
    (SA[1, 2], SA[2, 3], SA[3, 1])
end

idof_by_edge(::FunctionSpace{<:Lagrange, 2}, ::Triangle) = (SA[4], SA[5], SA[6])
function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 2}, ::Triangle)
    (SA[1, 2, 4], SA[2, 3, 5], SA[3, 1, 6])
end

idof_by_edge(::FunctionSpace{<:Lagrange, 3}, ::Triangle) = (SA[4, 5], SA[6, 7], SA[8, 9])
function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 3}, ::Triangle)
    (SA[1, 2, 4, 5], SA[2, 3, 6, 7], SA[3, 1, 8, 9])
end

#for Quad
function __idof_by_edge(n1, n2, range1, range2)
    I = LinearIndices((1:n1, 1:n2))
    e1 = I[range1, 1]            # edge x⁻
    e2 = I[n1, range2]           # edge y⁺
    e3 = I[reverse(range1), n2]  # edge x⁺
    e4 = I[1, reverse(range2)]   # edge y⁻
    return e1, e2, e3, e4
end

# for Cube
function __idof_by_edge(n1, n2, n3, range1, range2, range3)
    I   = LinearIndices((1:n1, 1:n2, 1:n3))
    e1  = I[range1, 1, 1]
    e2  = I[n1, range2, 1]
    e3  = I[reverse(range1), n2, 1]
    e4  = I[1, reverse(range2), 1]
    e5  = I[1, 1, range3]
    e6  = I[n1, 1, range3]
    e7  = I[n1, n2, range3]
    e8  = I[1, n2, range3]
    e9  = I[range1, 1, n3]
    e10 = I[n1, range2, n3]
    e11 = I[reverse(range1), n2, n3]
    e12 = I[1, reverse(range2), n3]
    return e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12
end

function _idof_by_edge(
    fs::FunctionSpace{<:Lagrange, degree},
    shape::Union{Square, Cube},
) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n = _get_num_nodes_per_dim(quadrule)
    range = ntuple(i -> 2:(n[i] - 1), length(n))
    e = __idof_by_edge(n..., range...)
    expr = [:(SA[$(x...)]) for x in e]
    return :(tuple($(expr...)))
end

function idof_by_edge(::FunctionSpace{<:Bcube.Lagrange, 0}, shape::Union{Cube, Square})
    ntuple(i -> SA[], nedges(shape))
end
@generated function idof_by_edge(
    fs::FS,
    shape::Shape,
) where {FS <: FunctionSpace{<:Lagrange}, Shape <: Union{Square, Cube}}
    return _idof_by_edge(fs(), shape())
end

function _idof_by_edge_with_bounds(
    fs::FunctionSpace{<:Lagrange, degree},
    shape::Union{Square, Cube},
) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n = _get_num_nodes_per_dim(quadrule)
    range = ntuple(i -> 1:n[i], length(n))
    e = __idof_by_edge(n..., range...)
    # put extrema nodes at first and second position :
    # e = map(e) do x
    #     length(x) == 2  && return x
    #     return [x[1],x[end],x[2:end-1]...]
    # end
    expr = [:(SA[$(x...)]) for x in e]
    return :(tuple($(expr...)))
end

@generated function idof_by_edge_with_bounds(
    fs::FS,
    shape::Shape,
) where {FS <: FunctionSpace{<:Lagrange}, Shape <: Union{Square, Cube}}
    return _idof_by_edge_with_bounds(fs(), shape())
end

# for Cube
function __idof_by_face(n1, n2, n3, range1, range2, range3)
    I  = LinearIndices((1:n1, 1:n2, 1:n3))
    f5 = I[1, range2, range3]  # face x⁻
    f3 = I[n1, range2, range3]  # face x⁺
    f2 = I[range1, 1, range3]  # face y⁻
    f4 = I[range1, n2, range3]  # face y⁺
    f1 = I[range1, range2, 1]  # face z⁻
    f6 = I[range1, range2, n3]  # face z⁺
    return f1, f2, f3, f4, f5, f6
end

function _idof_by_face(fs::FunctionSpace{<:Lagrange, degree}, shape::Cube) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n = _get_num_nodes_per_dim(quadrule)
    range = ntuple(i -> 2:(n[i] - 1), length(n))
    f = vec.(__idof_by_face(n..., range...))
    expr = [:(SA[$(x...)]) for x in f]
    return :(tuple($(expr...)))
end

@generated function idof_by_face(fs::FunctionSpace{<:Lagrange}, shape::Union{Cube})
    return _idof_by_face(fs(), shape())
end

function _idof_by_face_with_bounds(
    fs::FunctionSpace{<:Lagrange, degree},
    shape::Cube,
) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n = _get_num_nodes_per_dim(quadrule)
    range = ntuple(i -> 1:n[i], length(n))
    f = vec.(__idof_by_face(n..., range...))
    expr = [:(SA[$(x...)]) for x in f]
    return :(tuple($(expr...)))
end

@generated function idof_by_face_with_bounds(fs::FunctionSpace{<:Lagrange}, shape::Cube)
    return _idof_by_face_with_bounds(fs(), shape())
end

# Tetra
function idof_by_edge(::FunctionSpace{<:Lagrange, 1}, shape::Tetra)
    ntuple(i -> SA[], nedges(shape))
end
function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 1}, shape::Tetra)
    (SA[1, 2], SA[2, 3], SA[3, 1], SA[1, 4], SA[2, 4], SA[3, 4])
end

function idof_by_face(::FunctionSpace{<:Lagrange, 1}, shape::Tetra)
    ntuple(i -> SA[], nfaces(shape))
end
function idof_by_face_with_bounds(::FunctionSpace{<:Lagrange, 1}, shape::Tetra)
    (SA[1, 3, 2], SA[1, 2, 4], SA[2, 3, 4], SA[3, 1, 4])
end

# Prism
function idof_by_edge(::FunctionSpace{<:Lagrange, 1}, shape::Prism)
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_edge(fs::FunctionSpace{<:Lagrange, N}, ::Prism) where {N}
    idof_edge_tri = idof_by_edge(fs, Triangle())
    ndofs_tri = get_ndofs(fs, Triangle())
    ndofs_line = get_ndofs(fs, Line())

    return (
        idof_edge_tri[1],
        idof_edge_tri[2],
        idof_edge_tri[3],
        SVector{ndofs_line - 2}(1 + (i - 1) * ndofs_tri for i in 2:(ndofs_line - 1)),
        SVector{ndofs_line - 2}(2 + (i - 1) * ndofs_tri for i in 2:(ndofs_line - 1)),
        SVector{ndofs_line - 2}(3 + (i - 1) * ndofs_tri for i in 2:(ndofs_line - 1)),
        idof_edge_tri[1] .+ (ndofs_line - 1) * ndofs_tri,
        idof_edge_tri[2] .+ (ndofs_line - 1) * ndofs_tri,
        idof_edge_tri[3] .+ (ndofs_line - 1) * ndofs_tri,
    )
end

function idof_by_edge_with_bounds(fs::FunctionSpace{<:Lagrange, N}, ::Prism) where {N}
    idof_edge_tri = idof_by_edge_with_bounds(fs, Triangle())
    ndofs_tri = get_ndofs(fs, Triangle())
    ndofs_line = get_ndofs(fs, Line())

    return (
        idof_edge_tri[1],
        idof_edge_tri[2],
        idof_edge_tri[3],
        SVector{ndofs_line}(1 + (i - 1) * ndofs_tri for i in 1:ndofs_line),
        SVector{ndofs_line}(2 + (i - 1) * ndofs_tri for i in 1:ndofs_line),
        SVector{ndofs_line}(3 + (i - 1) * ndofs_tri for i in 1:ndofs_line),
        idof_edge_tri[1] .+ (ndofs_line - 1) * ndofs_tri,
        idof_edge_tri[2] .+ (ndofs_line - 1) * ndofs_tri,
        idof_edge_tri[3] .+ (ndofs_line - 1) * ndofs_tri,
    )
end

function idof_by_face(::FunctionSpace{<:Lagrange, 1}, shape::Prism)
    ntuple(i -> SA[], nfaces(shape))
end

# Note : recall that the Prism is obtained by cartesian product between Triangle and Line.
# Every inner dof on the "edge" of a Triangle becomes a dof of a side-face of the Prism, excluding the
# one lying on the bottom and top edges.
# WARNING : only Triangle WITHOUT INSIDE NODE are supported
function idof_by_face(fs::FunctionSpace{<:Lagrange, N}, ::Prism) where {N}
    idof_edge_tri = idof_by_edge(fs, Triangle())
    ndofs_tri = get_ndofs(fs, Triangle())
    ndofs_line = get_ndofs(fs, Line()) # >=3 (ensured by multi-dispatch on N)
    ndofs_line_inner = ndofs_line - 2 # exclude bottom and top edges

    #  Re "(i+1)" because "i" starts at "1" whereas the first element is "2"
    # Ideally, we would write vcat(ntuple(i -> idof_edge_tri[1] .+ (i - 1) * ndofs_tri, 2:ndofs_line_inner-1)...)
    return (
        vcat(ntuple(i -> idof_edge_tri[1] .+ (i + 1 - 1) * ndofs_tri, ndofs_line_inner)...),
        vcat(ntuple(i -> idof_edge_tri[2] .+ (i + 1 - 1) * ndofs_tri, ndofs_line_inner)...),
        vcat(ntuple(i -> idof_edge_tri[3] .+ (i + 1 - 1) * ndofs_tri, ndofs_line_inner)...),
        SA[], # bottom face (z=zmin)
        SA[], # top face (z=zmax)
    )
end

# Note : recall that the Prism is obtained by cartesian product between Triangle and Line.
# Every dof on the "edge" of a Triangle becomes a dof of a side-face of the Prism. Additionnaly,
# every dof of the Triangle becomes a dof of the bottom and top faces of the Prism.
function idof_by_face_with_bounds(fs::FunctionSpace{<:Lagrange, N}, ::Prism) where {N}
    idof_edge_tri = idof_by_edge_with_bounds(fs, Triangle())
    ndofs_tri = get_ndofs(fs, Triangle())
    ndofs_line = get_ndofs(fs, Line())

    return (
        vcat(ntuple(i -> idof_edge_tri[1] .+ (i - 1) * ndofs_tri, ndofs_line)...),
        vcat(ntuple(i -> idof_edge_tri[2] .+ (i - 1) * ndofs_tri, ndofs_line)...),
        vcat(ntuple(i -> idof_edge_tri[3] .+ (i - 1) * ndofs_tri, ndofs_line)...),
        SVector{ndofs_tri}(i for i in 1:ndofs_tri), # bottom face (z=zmin)
        SVector{ndofs_tri}(
            i for i in ((ndofs_tri * (ndofs_line - 1)) + 1):(ndofs_tri * ndofs_line)
        ), # top face (z=zmax)
    )
end

# Pyramid
function idof_by_edge(::FunctionSpace{<:Lagrange, 1}, shape::Pyramid)
    ntuple(i -> SA[], nedges(shape))
end
function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 1}, shape::Pyramid)
    (SA[1, 2], SA[2, 3], SA[3, 4], SA[4, 1], SA[1, 5], SA[2, 5], SA[3, 5], SA[4, 5])
end

function idof_by_face(::FunctionSpace{<:Lagrange, 1}, shape::Pyramid)
    ntuple(i -> SA[], nfaces(shape))
end
function idof_by_face_with_bounds(::FunctionSpace{<:Lagrange, 1}, shape::Pyramid)
    (SA[1, 4, 3, 2], SA[1, 2, 5], SA[2, 3, 5], SA[3, 4, 5], SA[4, 1, 5])
end

# Generic versions for Lagrange 0 and 1 (any shape)
# Rq: proceeding with "@eval" rather than using `AbstractShape` helps solving ambiguities
for S in (:Line, :Triangle, :Square, :Cube, :Tetra, :Prism, :Pyramid)
    @eval get_coords(::FunctionSpace{<:Lagrange, 0}, shape::$S) = (center(shape),)
end
for S in (:Triangle, :Tetra, :Prism, :Pyramid)
    @eval get_coords(::FunctionSpace{<:Lagrange, 1}, shape::$S) = get_coords(shape)
end

# Line, Square, Cube
function _make_staticvector(x)
    if isa(x, SVector) || !(length(x) < MAX_LENGTH_STATICARRAY)
        return x
    else
        return SA[x]
    end
end

function _coords_gen_impl(shape::AbstractShape, quad::AbstractQuadrature)
    quadrule = QuadratureRule(shape, quad)
    nodes = _make_staticvector.(get_nodes(quadrule))
    return :(tuple($(nodes...)))
end
@generated function _coords_gen(
    ::Shape,
    ::Quad,
) where {Shape <: AbstractShape, Quad <: AbstractQuadrature}
    _coords_gen_impl(Shape(), Quad())
end
function _get_coords(
    fs::FunctionSpace{<:Lagrange, D},
    shape::Union{Line, Square, Cube},
) where {D}
    quadtype = lagrange_quadrature_type(fs)
    quad = Quadrature(quadtype, Val(D))
    _coords_gen(shape, quad)
end
get_coords(fs::FunctionSpace{<:Lagrange, D}, shape::Line) where {D} = _get_coords(fs, shape)
function get_coords(fs::FunctionSpace{<:Lagrange, D}, shape::Square) where {D}
    _get_coords(fs, shape)
end
get_coords(fs::FunctionSpace{<:Lagrange, D}, shape::Cube) where {D} = _get_coords(fs, shape)

# Triangle
function get_coords(::FunctionSpace{<:Lagrange, 2}, shape::Triangle)
    (
        get_coords(shape)...,
        sum(get_coords(shape, (1, 2))) / 2,
        sum(get_coords(shape, (2, 3))) / 2,
        sum(get_coords(shape, (3, 1))) / 2,
    )
end

function get_coords(::FunctionSpace{<:Lagrange, 3}, shape::Triangle)
    (
        get_coords(shape)...,
        (2 / 3) * get_coords(shape, 1) + (1 / 3) * get_coords(shape, 2),
        (1 / 3) * get_coords(shape, 1) + (2 / 3) * get_coords(shape, 2),
        (2 / 3) * get_coords(shape, 2) + (1 / 3) * get_coords(shape, 3),
        (1 / 3) * get_coords(shape, 2) + (2 / 3) * get_coords(shape, 3),
        (2 / 3) * get_coords(shape, 3) + (1 / 3) * get_coords(shape, 1),
        (1 / 3) * get_coords(shape, 3) + (2 / 3) * get_coords(shape, 1),
        sum(get_coords(shape)) / 3,
    )
end

# Prism : cartesian product between the Triangle and the Line

# Rq: this implementation triggers a small allocation due to the call to
# `get_coords(fs, Bcube.Triangle())`. The rest of the function does not
# allocate
function get_coords(fs::FunctionSpace{<:Bcube.Lagrange}, ::Bcube.Prism)
    x_tri = get_coords(fs, Bcube.Triangle())
    x_line = get_coords(fs, Bcube.Line())

    return mapreduce(
        xl -> map(xt -> SA[xt..., xl], x_tri),
        (a, b) -> (a..., b...),
        first.(x_line),
    )
end
