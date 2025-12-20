abstract type AbstractBCType end

struct DirichletBCType <: AbstractBCType end

struct NeumannBCType <: AbstractBCType end

struct PeriodicBCType{T <: AbstractAffineTransformation, T2, T3, T4} <: AbstractBCType
    transformation::T
    labels_master::T2
    labels_slave::T3
    tol::T4
end
@inline transformation(perio::PeriodicBCType) = perio.transformation
@inline labels_master(perio::PeriodicBCType) = keys(perio.labels_master)
@inline labels_slave(perio::PeriodicBCType) = keys(perio.labels_slave)
@inline get_tolerance(perio::PeriodicBCType) = perio.tol

function PeriodicBCType(
    transformation::AbstractAffineTransformation,
    labelMaster::String,
    labelSlave::String,
    tol = 1e-9,
)
    PeriodicBCType(transformation, (labelMaster,), (labelSlave,), tol)
end

function PeriodicBCType(
    transformation::AbstractAffineTransformation,
    labelMaster::Tuple{String, Vararg{String, N}},
    labelSlave::Tuple{String, Vararg{String, N}},
    tol = 1e-9,
) where {N}
    #labels are storeds as keys of a namedtuple to ensure that
    #`PeriodicBCType` is `isbits`
    _labelMaster = (; zip(map(Symbol, labelMaster), ntuple(identity, N + 1))...)
    _labelSlave = (; zip(map(Symbol, labelSlave), ntuple(identity, N + 1))...)
    PeriodicBCType(transformation, _labelMaster, _labelSlave, tol)
end

abstract type AbstractBoundaryCondition end

"""
    Structure representing a boundary condition. Its purpose is to be attached to geometric entities
    (nodes, faces, ...).
"""
struct BoundaryCondition{T, F} <: AbstractBoundaryCondition
    type::T # Dirichlet, Neumann, other? For the moment, this property is entirely free
    func::F # This is a function depending on space and time : func(x,t)
end
@inline apply(bnd::BoundaryCondition, x, t) = bnd.func(x, t)
@inline apply(bnd::BoundaryCondition, x) = bnd.func(x, 0.0)

@inline type(bnd::BoundaryCondition) = bnd.type
