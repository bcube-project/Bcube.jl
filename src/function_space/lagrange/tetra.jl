# _scalar_shape_functions
_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Tetra, ξ) = SA[one(eltype(ξ))]

function _scalar_shape_functions(::FunctionSpace{<:Lagrange, 1}, ::Tetra, ξηζ)
    ξ, η, ζ = ξηζ
    return SA[
        1 - ξ - η - ζ
        ξ
        η
        ζ
    ]
end

# ∂λξ_∂ξ
function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Tetra, ξ)
    _zero = zero(eltype(ξ))
    return SA[_zero _zero _zero]
end

function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 1}, ::Val{1}, ::Tetra, ξηζ)
    return SA[
        -1 -1 -1
        1 0 0
        0 1 0
        0 0 1
    ]
end

# get_ndofs
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Tetra) = 1

get_ndofs(::FunctionSpace{<:Lagrange, 1}, ::Tetra) = 4

# idof_by_vertex
function idof_by_vertex(::FunctionSpace{<:Lagrange, 0}, shape::Tetra)
    ntuple(i -> SA[], nvertices(shape))
end

function idof_by_vertex(::FunctionSpace{<:Lagrange, degree}, shape::Tetra) where {degree}
    ntuple(i -> SA[i], nvertices(shape))
end

# idof_by_edge
function idof_by_edge(::FunctionSpace{<:Lagrange, 0}, shape::Tetra)
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_edge(::FunctionSpace{<:Lagrange, 1}, shape::Tetra)
    ntuple(i -> SA[], nedges(shape))
end

# idof_by_edge_with_bounds
function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 0}, shape::Tetra)
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 1}, shape::Tetra)
    (SA[1, 2], SA[2, 3], SA[3, 1], SA[1, 4], SA[2, 4], SA[3, 4])
end

# idof_by_face
function idof_by_face(::FunctionSpace{<:Lagrange, 0}, shape::Tetra)
    ntuple(i -> SA[], nfaces(shape))
end

function idof_by_face(::FunctionSpace{<:Lagrange, 1}, shape::Tetra)
    ntuple(i -> SA[], nfaces(shape))
end

# idof_by_face_with_bounds
function idof_by_face_with_bounds(::FunctionSpace{<:Lagrange, 0}, shape::Tetra)
    ntuple(i -> SA[], nfaces(shape))
end

function idof_by_face_with_bounds(::FunctionSpace{<:Lagrange, 1}, shape::Tetra)
    (SA[1, 3, 2], SA[1, 2, 4], SA[2, 3, 4], SA[3, 1, 4])
end

# get_coords
get_coords(::FunctionSpace{<:Lagrange, 0}, shape::Tetra) = (center(shape),)

get_coords(::FunctionSpace{<:Lagrange, 1}, shape::Tetra) = get_coords(shape)
