abstract type AbstractBCType end

struct DirichletBCType <: AbstractBCType end

struct NeumannBCType <: AbstractBCType end

struct PeriodicBCType{T <: AbstractAffineTransformation, T2, T3} <: AbstractBCType
    transformation::T
    labels_master::T2
    labels_slave::T3
end
@inline transformation(perio::PeriodicBCType) = perio.transformation
@inline labels_master(perio::PeriodicBCType) = perio.labels_master
@inline labels_slave(perio::PeriodicBCType) = perio.labels_slave

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
