# This file gathers shape function generation utils

using Symbolics
using LinearAlgebra
using ForwardDiff

@variables x[1:3]

abstract type GeneratedFiniteElement <: AbstractFunctionSpaceType end
struct LagrangeGenerated <: GeneratedFiniteElement end
struct HermiteGenerated <: GeneratedFiniteElement end

function FunctionSpace(::Val{:LagrangeGenerated}, degree::Integer)
    FunctionSpace(LagrangeGenerated(), degree)
end
function FunctionSpace(::Val{:HermiteGenerated}, degree::Integer)
    FunctionSpace(HermiteGenerated(), degree)
end

basis_functions_style(::FunctionSpace{<:LagrangeGenerated}) = NodalBasisFunctionsStyle()

abstract type LinearForm end

"""
  Defines a point evaluation linear functional l(f) = f(x) (x is fixed)
"""
struct PointEvaluationForm <: LinearForm
    x::Vector{Float64}
end

function (l::PointEvaluationForm)(f::Function)
    return f(l.x)
end

"""
  Defines a derivative point evaluation linear functional l(f) = f'(x) (x is fixed)
"""
struct PointDerivativeForm <: LinearForm
    x::Vector{Float64}
end

function (l::PointDerivativeForm)(f::Function)
    dfdx = x -> ForwardDiff.gradient(f, x)[1]
    return dfdx(l.x)
end

"""
  Returns the linear forms for continous Lagrange shape functions on a line => evaluation of f at nodes
"""
function Forms(::LagrangeGenerated, ::Line, deg)
    forms = Array{LinearForm}(undef, deg + 1)
    for i in 1:(deg + 1)
        x = [-1.0 + (i - 1) * 2.0 / (deg)]
        forms[i] = PointEvaluationForm(x)
    end

    return forms
end

"""
  Returns the linear forms for continous Lagrange shape functions on a triangle => evaluation of f at nodes
"""
function Forms(::LagrangeGenerated, ::Triangle, deg)
    forms = Array{LinearForm}(undef, Int((deg + 1) * (deg + 2) / 2))
    k = 1
    for i in 0:deg
        for j in 0:(deg - i)
            x = [i / deg, j / deg]
            forms[k] = PointEvaluationForm(x)
            k = k + 1
        end
    end

    return forms
end

"""
  Returns the linear forms for continous Lagrange shape functions on a square => evaluation of f at nodes
"""
function Forms(::LagrangeGenerated, ::Square, deg)
    forms = Array{LinearForm}(undef, (deg + 1)^2)
    k = 1
    for i in 0:deg
        for j in 0:deg
            x = [-1.0 + 2.0 * i / deg, -1.0 + 2.0 * j / deg]
            forms[k] = PointEvaluationForm(x)
            k = k + 1
        end
    end

    return forms
end

"""
  Returns the linear forms for cubic Hermite shape functions on a line => evaluation of f and dfdx at nodes (x=-1 and x=1)
"""
function Forms(::HermiteGenerated, ::Line, deg)
    forms = Array{LinearForm}(undef, deg + 1)

    if (deg != 3)
        println("Warning: 1D Hermite basis is only defined for deg=3. Using deg=3")
        deg = 3
    end

    forms[1] = PointEvaluationForm([-1.0])
    forms[2] = PointDerivativeForm([-1.0])

    forms[3] = PointEvaluationForm([1.0])
    forms[4] = PointDerivativeForm([1.0])

    return forms
end

"""
  Primary 1D polynomial basis. The shape functions are to be expressed in this basis (temporarily, just for their construction)
"""
function primary_basis(::Line, deg::Int)
    return LegendrePolynomials(deg + 1)
end

"""
  Primary 2D polynomial basis on a square. The shape functions are to be expressed in this basis (temporarily, just for their construction)
"""
function primary_basis(::Square, deg::Int)
    basis = Array{Num}(undef, (deg + 1)^2)

    P = LegendrePolynomials(deg + 1)

    n = 1
    for i in 1:(deg + 1)
        for j in 1:(deg + 1)
            basis[n] = P[i] * substitute(P[j], Dict(x[1] => x[2]))
            n = n + 1
        end
    end

    return basis
end

"""
  Functions that define Legendre polynomials
"""
LegendrePolynomials(::Val{1}) = 1.0

function LegendrePolynomials(::Val{2})
    return [1.0, x[1]]
end

function LegendrePolynomials(n::Int)
    poly = Array{Num}(undef, n)

    poly[1] = 1.0
    poly[2] = x[1]

    for i in 2:(n - 1)
        poly[i + 1] = ((2 * i - 1) / i) * x[1] * poly[i] - ((i - 1) / i) * poly[i - 1]
    end

    return poly
end

"""
  Primary 2D polynomial basis on a triangle. The shape functions are to be expressed in this basis (temporarily, just for their construction)
"""
function primary_basis(::Triangle, deg::Int)

    #b1(x) = 1.0 - x[1] - x[2]
    #b2(x) = x[1]
    #b3(x) = x[2]

    basis = Array{Num}(undef, Int((deg + 1) * (deg + 2) / 2))
    n = 1
    for i in 0:deg
        for j in 0:deg
            for k in 0:deg
                if i + j + k == deg
                    #@show i,j,k,n, i+j+k
                    coef = factorial(deg) / (factorial(i) * factorial(j) * factorial(k))
                    basis[n] = coef * (1.0 - x[1] - x[2])^i * x[1]^j * x[2]^k
                    n = n + 1
                end
            end
        end
    end

    return basis
end

"""
  Computes α
"""
function Compute_α(
    fe::K,
    shape::S,
    ::Val{D},
) where {K <: GeneratedFiniteElement, S <: AbstractShape, D}
    forms = Forms(fe, shape, D)

    primaryBasis = primary_basis(shape, D)

    nforms = size(forms)[1]
    B = zeros(nforms, nforms)
    B = [
        forms[i](build_function(primaryBasis[j], x; expression = Val{false})) for
        i in 1:nforms, j in 1:nforms
    ]

    return transpose(inv(B))
end

"""
  Generates the shape functions (linear combination of Legendre polynomials) associated with a given set of linear forms
"""
function ShapeFunctions(
    fe::K,
    ::Val{D},
    shape::S,
) where {K <: GeneratedFiniteElement, D, S <: AbstractShape}
    forms = Forms(fe, shape, D)

    primaryBasis = primary_basis(shape, D)

    nforms = size(forms)[1]

    alpha = Compute_α(fe, shape, Val(D))

    shapeFunctions = Array{Num}(undef, nforms)

    for i in 1:nforms
        let
            shapeFunctions[i] = 0.0
            for j in 1:nforms
                g = shapeFunctions[i]
                shapeFunctions[i] += alpha[i, j] * primaryBasis[j]
            end
        end
    end

    return shapeFunctions
end

"""
  Compute the shape functions and convert the symbolic expressions to generated functions
"""
function build_expr(shapeFunctions)
    f_expr = build_function(shapeFunctions, [x])
    Base.remove_linenums!.(f_expr)
    return f_expr
end

function shape_functions_impl(
    ::S,
    ::Val{D},
    ::K,
) where {S <: AbstractShape, D, K <: GeneratedFiniteElement}
    shapeFunctions = ShapeFunctions(K(), Val(D), S())
    _expr = Symbolics.toexpr.(shapeFunctions)
    expr = :([$(_expr...)])
    return expr
end

@generated function _scalar_shape_functions(
    ::K,
    ::Val{D},
    ::S,
    x::AbstractVector,
) where {K <: GeneratedFiniteElement, D, S <: AbstractShape}
    shape_functions_impl(S(), Val(D), K())
end

function _scalar_shape_functions(
    fs::FunctionSpace{<:GeneratedFiniteElement, D},
    shape::AbstractShape,
    ξ,
) where {D}
    SF = _scalar_shape_functions(get_type(fs)(), Val(get_degree(fs)), shape, ξ)
    return SVector{size(SF)[1]}(SF)
end

function shape_functions(
    fs::FunctionSpace{<:GeneratedFiniteElement, D},
    ::Val{N},
    shape::AbstractShape,
    ξ,
) where {D, N}
    if N == 1
        return _scalar_shape_functions(fs, shape, ξ)
        #elseif N < MAX_LENGTH_STATICARRAY
        #    return kron(SMatrix{N, N}(1I), _scalar_shape_functions(fs, shape, ξ))
        #else
        #    return kron(Diagonal([1.0 for i in 1:N]), _scalar_shape_functions(fs, shape, ξ))
    else
        error("generated shape functions not implemented for N>1")
    end
end
function shape_functions(
    fs::FunctionSpace{<:GeneratedFiniteElement, D},
    n::Val{N},
    shape::AbstractShape,
) where {D, N}
    ξ -> shape_functions(fs, n, shape, ξ)
end

# Number of dofs and idof for Hermite generated on a line
get_ndofs(::FunctionSpace{<:HermiteGenerated, 3}, ::Line) = 4
function idof_by_vertex(::FunctionSpace{<:HermiteGenerated, 3}, shape::Line)
    (SA[1, 3], SA[2, 4])
end
function idof_by_edge(::FunctionSpace{<:HermiteGenerated, 3}, shape::Line)
    ntuple(i -> SA[], nedges(shape))
end
function idof_by_edge_with_bounds(::FunctionSpace{<:HermiteGenerated, 3}, shape::Line)
    (SA[1, 3], SA[2, 4])
end

# Number of dofs and idof for Lagrange generated on a line
get_ndofs(::FunctionSpace{<:LagrangeGenerated, N}, ::Line) where {N} = N + 1
function idof_by_vertex(::FunctionSpace{<:LagrangeGenerated, N}, shape::Line) where {N}
    (SA[1], SA[2])
end
function idof_by_edge(::FunctionSpace{<:LagrangeGenerated, N}, shape::Line) where {N}
    ntuple(i -> SA[], nedges(shape))
end
function idof_by_edge_with_bounds(
    ::FunctionSpace{<:LagrangeGenerated, N},
    shape::Line,
) where {N}
    (SA[1], SA[2])
end

# Number of dofs and idof for Lagrange generated on a triangle
function get_ndofs(::FunctionSpace{<:LagrangeGenerated, N}, ::Triangle) where {N}
    Int((N + 1) * (N + 2) / 2)
end
function idof_by_vertex(::FunctionSpace{<:LagrangeGenerated, N}, shape::Triangle) where {N}
    (SA[1], SA[Int((N + 1) * (N + 2) / 2)], SA[N + 1])
end
function idof_by_edge(::FunctionSpace{<:LagrangeGenerated, N}, shape::Triangle) where {N}
    idof_1 = [1 + Int(i * (N + 1 - (i - 1) / 2)) for i in 1:(N - 1)]
    idof_2 = [N + 1 + Int(i * (N + 1 - (i + 1) / 2)) for i in 1:(N - 1)]
    idof_3 = [i + 1 for i in 1:(N - 1)]
    return (
        SVector{N - 1, Int64}(idof_1),
        SVector{N - 1, Int64}(idof_2),
        SVector{N - 1, Int64}(idof_3),
    )
end
function idof_by_edge_with_bounds(
    ::FunctionSpace{<:LagrangeGenerated, N},
    shape::Triangle,
) where {N}
    idof_1 = [1 + Int(i * (N + 1 - (i - 1) / 2)) for i in 0:N]
    idof_2 = [N + 1 + Int(i * (N + 1 - (i + 1) / 2)) for i in 0:N]
    idof_3 = [i + 1 for i in 0:N]
    return (
        SVector{N + 1, Int64}(idof_1),
        SVector{N + 1, Int64}(idof_2),
        SVector{N + 1, Int64}(idof_3),
    )
end

# Number of dofs and idof for Lagrange generated on a square
get_ndofs(::FunctionSpace{<:LagrangeGenerated, N}, ::Square) where {N} = (N + 1)^2
function idof_by_vertex(::FunctionSpace{<:LagrangeGenerated, N}, shape::Square) where {N}
    (SA[1], SA[N + 1], SA[(N + 1)^2 - N], SA[(N + 1)^2])
end
function idof_by_edge(::FunctionSpace{<:LagrangeGenerated, N}, shape::Square) where {N}
    idof_1 = [i for i in 2:N]
    idof_2 = [(i - 1) * (N + 1) + 1 for i in 2:N]
    idof_3 = [i for i in ((N + 1)^2 - N + 1):((N + 1)^2 - 1)]
    idof_4 = [i * (N + 1) for i in 2:N]
    return (
        SVector{N - 1, Int64}(idof_1),
        SVector{N - 1, Int64}(idof_2),
        SVector{N - 1, Int64}(idof_3),
        SVector{N - 1, Int64}(idof_4),
    )
end
function idof_by_edge_with_bounds(
    ::FunctionSpace{<:LagrangeGenerated, N},
    shape::Square,
) where {N}
    idof_1 = [i for i in 1:(N + 1)]
    idof_2 = [(i - 1) * (N + 1) + 1 for i in 1:(N + 1)]
    idof_3 = [i for i in ((N + 1)^2 - N):((N + 1)^2)]
    idof_4 = [i * (N + 1) for i in 1:(N + 1)]
    return (
        SVector{N + 1, Int64}(idof_1),
        SVector{N + 1, Int64}(idof_2),
        SVector{N + 1, Int64}(idof_3),
        SVector{N + 1, Int64}(idof_4),
    )
end

get_coords(::FunctionSpace{<:LagrangeGenerated, 0}, shape::AbstractShape) = (center(shape),)
function get_coords(::FunctionSpace{<:LagrangeGenerated, 1}, ::Line)
    (SA[-1.0], SA[1.0])
end
function get_coords(::FunctionSpace{<:LagrangeGenerated, 1}, ::Triangle)
    (SA[0.0, 0.0], SA[1.0, 0.0], SA[0.0, 1.0])
end
# function get_coords(::FunctionSpace{<:LagrangeGenerated, D}, ::Triangle) where {D}
#     coords = ()
#     k = 1
#     for i in 0:D
#         for j in 0:(D - i)
#             x = [i / D, j / D]
#             k = k + 1
#         end
#     end
# end
