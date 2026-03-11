# _scalar_shape_functions
_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Triangle, ξ) = SA[one(eltype(ξ))]

function _scalar_shape_functions(::FunctionSpace{<:Lagrange, 1}, ::Triangle, ξ)
    return SA[
        1 - ξ[1] - ξ[2]
        ξ[1]
        ξ[2]
    ]
end

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

# ∂λξ_∂ξ
function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Triangle, ξ)
    _zero = zero(eltype(ξ))
    return SA[_zero _zero]
end

function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 1}, ::Val{1}, ::Triangle, ξ)
    return SA[
        -1.0 -1.0
        1.0 0.0
        0.0 1.0
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

# get_ndofs
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Triangle) = 1
get_ndofs(::FunctionSpace{<:Lagrange, 1}, ::Triangle) = 3
get_ndofs(::FunctionSpace{<:Lagrange, 2}, ::Triangle) = 6
get_ndofs(::FunctionSpace{<:Lagrange, 3}, ::Triangle) = 10

# idof_by_vertex
function idof_by_vertex(::FunctionSpace{<:Lagrange, 0}, shape::Triangle)
    ntuple(i -> SA[], nvertices(shape))
end

function idof_by_vertex(::FunctionSpace{<:Lagrange, degree}, shape::Triangle) where {degree}
    ntuple(i -> SA[i], nvertices(shape))
end

# idof_by_edge
function idof_by_edge(::FunctionSpace{<:Lagrange, 0}, shape::Triangle)
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_edge(::FunctionSpace{<:Lagrange, 1}, shape::Triangle)
    ntuple(i -> SA[], nedges(shape))
end

idof_by_edge(::FunctionSpace{<:Lagrange, 2}, ::Triangle) = (SA[4], SA[5], SA[6])

idof_by_edge(::FunctionSpace{<:Lagrange, 3}, ::Triangle) = (SA[4, 5], SA[6, 7], SA[8, 9])

# idof_by_edge_with_bounds
function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 0}, shape::Triangle)
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 1}, shape::Triangle)
    (SA[1, 2], SA[2, 3], SA[3, 1])
end

function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 2}, ::Triangle)
    (SA[1, 2, 4], SA[2, 3, 5], SA[3, 1, 6])
end

function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 3}, ::Triangle)
    (SA[1, 2, 4, 5], SA[2, 3, 6, 7], SA[3, 1, 8, 9])
end

# idof_by_face
function idof_by_face(::FunctionSpace{<:Lagrange, 0}, shape::Triangle)
    ntuple(i -> SA[], nfaces(shape))
end

# idof_by_face_with_bounds
function idof_by_face_with_bounds(::FunctionSpace{<:Lagrange, 0}, shape::Triangle)
    ntuple(i -> SA[], nfaces(shape))
end

# get_coords
get_coords(::FunctionSpace{<:Lagrange, 0}, shape::Triangle) = (center(shape),)

get_coords(::FunctionSpace{<:Lagrange, 1}, shape::Triangle) = get_coords(shape)

function get_coords(::FunctionSpace{<:Lagrange, 2}, shape::Triangle)
    (
        get_coords(shape)...,
        sum(get_coords(shape, [1, 2])) / 2,
        sum(get_coords(shape, [2, 3])) / 2,
        sum(get_coords(shape, [3, 1])) / 2,
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