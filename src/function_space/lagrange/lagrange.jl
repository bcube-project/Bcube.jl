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
