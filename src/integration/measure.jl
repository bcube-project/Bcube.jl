
abstract type AbstractMeasure{D <: AbstractDomain, Q <: AbstractQuadrature} end

get_domain_type(::AbstractMeasure{D}) where {D} = D
get_quadrature_type(::AbstractMeasure{D, Q}) where {D, Q} = Q
get_domain(m::AbstractMeasure) = m.domain
get_quadrature(m::AbstractMeasure) = m.quadrature

function LazyOperators.pretty_name(m::AbstractMeasure)
    "Measure(domain = " *
    pretty_name(get_domain(m)) *
    ", quadrature type = " *
    string(get_quadrature_type(m)) *
    ")"
end

"""
A `Measure` is geometrical domain of integration associated to a way to integrate
on it (i.e a quadrature rule).

`Q` is the quadrature type used to integrate expressions using this measure.
"""
struct Measure{D <: AbstractDomain, Q <: AbstractQuadrature} <: AbstractMeasure{D, Q}
    domain::D
    quadrature::Q
end

"""
    Measure(domain::AbstractDomain, degree::Integer)
    Measure(domain::AbstractDomain, ::Val{degree}) where {degree}

Build a `Measure` on the designated `AbstractDomain` with a default quadrature of degree `degree`.

# Arguments
- `domain::AbstractDomain` : the domain to integrate over
- `degree` : the degree of the quadrature rule (`Legendre` quadrature type by default)

# Examples
```julia-repl
julia> mesh = line_mesh(10)
julia> Ω = CellDomain(mesh)
julia> dΩ = Measure(Ω, 2)
```
"""
Measure(domain::AbstractDomain, degree::Integer) = Measure(domain, Val(degree))
function Measure(domain::AbstractDomain, degree::Val{D}) where {D}
    Measure(domain, Quadrature(degree))
end

function get_face_normals(measure::Measure{<:AbstractFaceDomain})
    get_face_normals(get_domain(measure))
end

function get_cell_normals(measure::Measure{<:AbstractCellDomain})
    get_cell_normals(get_domain(measure))
end
