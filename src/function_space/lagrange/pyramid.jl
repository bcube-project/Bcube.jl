# _scalar_shape_functions
_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Pyramid, ξ) = SA[one(eltype(ξ))]

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

# ∂λξ_∂ξ
function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Pyramid, ξ)
    _zero = zero(eltype(ξ))
    SA[_zero _zero _zero]
end

# get_ndofs
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Pyramid) = 1
get_ndofs(::FunctionSpace{<:Lagrange, 1}, ::Pyramid) = 5

# idof_by_vertex
function idof_by_vertex(::FunctionSpace{<:Lagrange, 0}, shape::Pyramid)
    ntuple(i -> SA[], nvertices(shape))
end

function idof_by_vertex(::FunctionSpace{<:Lagrange, degree}, shape::Pyramid) where {degree}
    ntuple(i -> SA[i], nvertices(shape))
end

# idof_by_edge
function idof_by_edge(::FunctionSpace{<:Lagrange, 0}, shape::Pyramid)
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_edge(::FunctionSpace{<:Lagrange, 1}, shape::Pyramid)
    ntuple(i -> SA[], nedges(shape))
end

# idof_by_edge_with_bounds
function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 0}, shape::Pyramid)
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 1}, shape::Pyramid)
    (SA[1, 2], SA[2, 3], SA[3, 4], SA[4, 1], SA[1, 5], SA[2, 5], SA[3, 5], SA[4, 5])
end

# idof_by_face
function idof_by_face(::FunctionSpace{<:Lagrange, 0}, shape::Pyramid)
    ntuple(i -> SA[], nfaces(shape))
end

function idof_by_face(::FunctionSpace{<:Lagrange, 1}, shape::Pyramid)
    ntuple(i -> SA[], nfaces(shape))
end

# idof_by_face_with_bounds
function idof_by_face_with_bounds(::FunctionSpace{<:Lagrange, 0}, shape::Pyramid)
    ntuple(i -> SA[], nfaces(shape))
end

function idof_by_face_with_bounds(::FunctionSpace{<:Lagrange, 1}, shape::Pyramid)
    (SA[1, 4, 3, 2], SA[1, 2, 5], SA[2, 3, 5], SA[3, 4, 5], SA[4, 1, 5])
end

# get_coords
get_coords(::FunctionSpace{<:Lagrange, 0}, shape::Pyramid) = (center(shape),)
get_coords(::FunctionSpace{<:Lagrange, 1}, shape::Pyramid) = get_coords(shape)
