#_scalar_shape_functions
_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Prism, ξ) = SA[one(eltype(ξ))]

function _scalar_shape_functions(::FunctionSpace{<:Lagrange, 1}, ::Prism, ξηζ)
    ξ, η, ζ = ξηζ
    return SA[
        (1 - ξ - η) * (1 - ζ)
        ξ * (1 - ζ)
        η * (1 - ζ)
        (1 - ξ - η) * (1 + ζ)
        ξ * (1 + ζ)
        η * (1 + ζ)
    ] ./ 2.0
end

# ∂λξ_∂ξ
function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Prism, ξ)
    _zero = zero(eltype(ξ))
    SA[_zero _zero _zero]
end

function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 1}, ::Val{1}, ::Prism, ξηζ)
    ξ, η, ζ = ξηζ
    return SA[
        (-(1 - ζ)) (-(1 - ζ)) (-(1 - ξ - η))
        ((1-ζ)) (0.0) (-ξ)
        (0.0) ((1-ζ)) (-η)
        (-(1 + ζ)) (-(1 + ζ)) ((1 - ξ-η))
        ((1+ζ)) (0.0) (ξ)
        (0.0) ((1+ζ)) (η)
    ] ./ 2.0
end

# get_ndofs
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Prism) = 1
get_ndofs(::FunctionSpace{<:Lagrange, 1}, ::Prism) = 6

# idof_by_vertex
function idof_by_vertex(::FunctionSpace{<:Lagrange, 0}, shape::Prism)
    ntuple(i -> SA[], nvertices(shape))
end

function idof_by_vertex(::FunctionSpace{<:Lagrange, degree}, shape::Prism) where {degree}
    ntuple(i -> SA[i], nvertices(shape))
end

# idof_by_edge
function idof_by_edge(::FunctionSpace{<:Lagrange, 0}, shape::Prism)
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_edge(::FunctionSpace{<:Lagrange, 1}, shape::Prism)
    ntuple(i -> SA[], nedges(shape))
end

# idof_by_edge_with_bounds
function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 0}, shape::Prism)
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 1}, shape::Prism)
    (
        SA[1, 2],
        SA[2, 3],
        SA[3, 1],
        SA[1, 4],
        SA[2, 5],
        SA[3, 6],
        SA[4, 5],
        SA[5, 6],
        SA[6, 4],
    )
end

# idof_by_face
function idof_by_face(::FunctionSpace{<:Lagrange, 0}, shape::Prism)
    ntuple(i -> SA[], nfaces(shape))
end

function idof_by_face(::FunctionSpace{<:Lagrange, 1}, shape::Prism)
    ntuple(i -> SA[], nfaces(shape))
end

# idof_by_face_with_bounds
function idof_by_face_with_bounds(::FunctionSpace{<:Lagrange, 0}, shape::Prism)
    ntuple(i -> SA[], nfaces(shape))
end

function idof_by_face_with_bounds(::FunctionSpace{<:Lagrange, 1}, shape::Prism)
    (SA[1, 2, 5, 4], SA[2, 3, 6, 5], SA[3, 1, 4, 6], SA[1, 3, 2], SA[4, 5, 6])
end

# get_coords
get_coords(::FunctionSpace{<:Lagrange, 0}, shape::Prism) = (center(shape),)
get_coords(::FunctionSpace{<:Lagrange, 1}, shape::Prism) = get_coords(shape)