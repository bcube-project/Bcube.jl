abstract type AbstractBCType end

struct DirichletBCType <: AbstractBCType end

struct NeumannBCType <: AbstractBCType end

struct PeriodicBCType{T <: AbstractAffineTransformation, T2, T3} <: AbstractBCType
    transformation::T
    labels_master::T2
    labels_slave::T3
end
@inline transformation(perio::PeriodicBCType) = perio.transformation
@inline labels_master(perio::PeriodicBCType) = keys(perio.labels_master)
@inline labels_slave(perio::PeriodicBCType) = keys(perio.labels_slave)

function PeriodicBCType(
    transformation::AbstractAffineTransformation,
    labelMaster::String,
    labelSlave::String,
)
    PeriodicBCType(transformation, (labelMaster,), (labelSlave,))
end

function PeriodicBCType(
    transformation::AbstractAffineTransformation,
    labelMaster::Tuple{String, Vararg{String, N}},
    labelSlave::Tuple{String, Vararg{String, N}},
) where {N}
    #labels are storeds as keys of a namedtuple to ensure that
    #`PeriodicBCType` is `isbits`
    _labelMaster = (; zip(map(Symbol, labelMaster), ntuple(identity, N + 1))...)
    _labelSlave = (; zip(map(Symbol, labelSlave), ntuple(identity, N + 1))...)
    #    labels = (Symbol(labelMaster) = 1, Symbol(labelSlave) = 2)
    PeriodicBCType(transformation, _labelMaster, _labelSlave)
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
