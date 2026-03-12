
# _scalar_shape_functions
#
#
# _scalar_shape_functions
#   -> shape_functions_symbolic
#     -> _shape_functions_symbolic_*
#       -> _lagrange_poly

function _lagrange_poly(j, ξ, nodes)
    coef = 1.0 / prod((nodes[j] - nodes[k]) for k in eachindex(nodes) if k ≠ j)
    terms = [:($ξ - $(nodes[k])) for k in eachindex(nodes) if k ≠ j]
    if isempty(terms)
        prod_expr = 1.0
    else
        prod_expr = foldl((x, y) -> :($x * $y), terms)
    end
    return :($coef * $prod_expr)
end

function __shape_functions_symbolic(
    quadrule::Bcube.AbstractQuadratureRule,
    ::Type{T},
) where {T}
    nodes = get_nodes(quadrule)
    expr_l = map(j -> _lagrange_poly(j, :ξ, nodes), eachindex(nodes))
    return :(SA[$(expr_l...)])
end

@generated function _shape_functions_symbolic(
    ::Line,
    ::Q,
    ξ::T,
) where {Q <: AbstractQuadrature, T}
    quadrule = QuadratureRule(Line(), Q())
    __shape_functions_symbolic(quadrule, T)
end

"""
    shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, ::Shape, ξ) where {D, Shape<:Line}
    ∂λξ_∂ξ_symbolic(fs::FunctionSpace{<:Lagrange, D}, ::Shape, ξ) where {D, Shape<:Line}

# Implementation
Based on `Symbolic.jl`. First tests show that this version is slower than the implementation based on `meta`
when `D` is greater. Further investigations are needed to understand this behavior.

`shape_functions_symbolic` uses a "generated" function named `_shape_functions_symbolic`. The core of the
generated function is an `Expression` that is created by `__shape_functions_symbolic`. This latter function
uses the `Symbolics` package and the lagrange polynomials (defined in `_lagrange_poly`).
"""
function shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, ::Line, ξ) where {D}
    quadtype = lagrange_quadrature_type(fs)
    quad = Quadrature(quadtype, Val(D))
    _shape_functions_symbolic(Line(), quad, ξ[1])
end

function _shape_functions_symbolic_square(
    l1::SVector{N1, T},
    l2::SVector{N2, T},
) where {N1, N2, T}
    N = N1 * N2
    if N < MAX_LENGTH_STATICARRAY
        return SVector{N, T}(l1[i] * l2[j] for i in 1:N1, j in 1:N2)
    else
        _l1 = Array(l1)
        _l2 = Array(l2)
        return vec([i * j for i in _l1, j in _l2])
    end
end

function shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, ::Square, ξ) where {D}
    l1 = shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, Line(), ξ[1])
    l2 = shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, Line(), ξ[2])
    return _shape_functions_symbolic_square(l1, l2)
end

function shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, ::Cube, ξ) where {D}
    l1 = shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, Line(), ξ[1])
    l2 = shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, Line(), ξ[2])
    l3 = shape_functions_symbolic(fs::FunctionSpace{<:Lagrange, D}, Line(), ξ[3])
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(Cube(), Quadrature(quadtype, Val(D)))
    N = length(quadrule)
    if N < MAX_LENGTH_STATICARRAY
        return SVector{N}(i * j * k for i in l1, j in l2, k in l3)
    else
        _l1 = Array(l1)
        _l2 = Array(l2)
        _l3 = Array(l3)
        return vec([i * j * k for i in _l1, j in _l2, k in _l3])
    end
end

_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Line, ξ) = SA[one(eltype(ξ))]
_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Square, ξ) = SA[one(eltype(ξ))]
_scalar_shape_functions(::FunctionSpace{<:Lagrange, 0}, ::Cube, ξ) = SA[one(eltype(ξ))]

function _scalar_shape_functions(
    fs::FunctionSpace{<:Lagrange, D},
    shape::Union{Line, Square, Cube},
    ξ,
) where {D}
    shape_functions_symbolic(fs, shape, ξ)
end

# ∂λξ_∂ξ
function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Line, ξ::Number)
    SA[zero(ξ)]
end

function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Square, ξ)
    _zero = zero(eltype(ξ))
    return SA[_zero _zero]
end

function ∂λξ_∂ξ(::FunctionSpace{<:Lagrange, 0}, ::Val{1}, ::Cube, ξ)
    _zero = zero(eltype(ξ))
    SA[_zero _zero _zero]
end

# get_ndofs
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Line) = 1
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Square) = 1
get_ndofs(::FunctionSpace{<:Lagrange, 0}, ::Cube) = 1

function get_ndofs(
    fs::FunctionSpace{<:Lagrange, D},
    shape::Union{Line, Square, Cube},
) where {D}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(D)))
    return length(quadrule)
end

# idof_by_vertex
function _idof_by_vertex(fs::FunctionSpace{<:Lagrange, degree}, shape::Line) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n = _get_num_nodes_per_dim(quadrule)
    return :(SA[1], SA[$n])
end

function _idof_by_vertex(
    fs::FunctionSpace{<:Lagrange, degree},
    shape::Square,
) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n1, n2 = _get_num_nodes_per_dim(quadrule)
    I = LinearIndices((1:n1, 1:n2))
    p1, p2, p3, p4 = I[1, 1], I[n1, 1], I[n1, n2], I[1, n2]
    return :(SA[$p1], SA[$p2], SA[$p3], SA[$p4])
end

function _idof_by_vertex(fs::FunctionSpace{<:Lagrange, degree}, shape::Cube) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n1, n2, n3 = _get_num_nodes_per_dim(quadrule)
    I = LinearIndices((1:n1, 1:n2, 1:n3))
    p1, p2, p3, p4 = I[1, 1, 1], I[n1, 1, 1], I[n1, n2, 1], I[1, n2, 1]
    p5, p6, p7, p8 = I[1, 1, n3], I[n1, 1, n3], I[n1, n2, n3], I[1, n2, n3]
    return :(SA[$p1], SA[$p2], SA[$p3], SA[$p4], SA[$p5], SA[$p6], SA[$p7], SA[$p8])
end

function idof_by_vertex(::FunctionSpace{<:Lagrange, 0}, shape::Union{Line, Square, Cube})
    ntuple(i -> SA[], nvertices(shape))
end
@generated function idof_by_vertex(
    fs::FunctionSpace{<:Lagrange},
    shape::Union{Line, Square, Cube},
)
    return _idof_by_vertex(fs(), shape())
end

# idof_by_edge
function __idof_by_edge(n1, n2, range1, range2)
    I = LinearIndices((1:n1, 1:n2))
    e1 = I[range1, 1]            # edge x⁻
    e2 = I[n1, range2]           # edge y⁺
    e3 = I[reverse(range1), n2]  # edge x⁺
    e4 = I[1, reverse(range2)]   # edge y⁻
    return e1, e2, e3, e4
end

function __idof_by_edge(n1, n2, n3, range1, range2, range3)
    I   = LinearIndices((1:n1, 1:n2, 1:n3))
    e1  = I[range1, 1, 1]
    e2  = I[n1, range2, 1]
    e3  = I[reverse(range1), n2, 1]
    e4  = I[1, reverse(range2), 1]
    e5  = I[1, 1, range3]
    e6  = I[n1, 1, range3]
    e7  = I[n1, n2, range3]
    e8  = I[1, n2, range3]
    e9  = I[range1, 1, n3]
    e10 = I[n1, range2, n3]
    e11 = I[reverse(range1), n2, n3]
    e12 = I[1, reverse(range2), n3]
    return e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12
end

function _idof_by_edge(
    fs::FunctionSpace{<:Lagrange, degree},
    shape::Union{Square, Cube},
) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n = _get_num_nodes_per_dim(quadrule)
    range = ntuple(i -> 2:(n[i] - 1), length(n))
    e = __idof_by_edge(n..., range...)
    expr = [:(SA[$(x...)]) for x in e]
    return :(tuple($(expr...)))
end

function idof_by_edge(::FunctionSpace{<:Bcube.Lagrange, 0}, shape::Union{Cube, Square})
    ntuple(i -> SA[], nedges(shape))
end

@generated function idof_by_edge(
    fs::FS,
    shape::Shape,
) where {FS <: FunctionSpace{<:Lagrange}, Shape <: Union{Square, Cube}}
    return _idof_by_edge(fs(), shape())
end

# idof_by_edge_with_bounds
function idof_by_edge_with_bounds(
    ::FunctionSpace{<:Lagrange, 0},
    shape::Union{Line, Square, Cube},
)
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_edge_with_bounds(
    fs::FunctionSpace{<:Lagrange, degree},
    shape::Line,
) where {degree}
    idof_by_vertex(fs, shape)
end

function _idof_by_edge_with_bounds(
    fs::FunctionSpace{<:Lagrange, degree},
    shape::Union{Square, Cube},
) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n = _get_num_nodes_per_dim(quadrule)
    range = ntuple(i -> 1:n[i], length(n))
    e = __idof_by_edge(n..., range...)
    # put extrema nodes at first and second position :
    # e = map(e) do x
    #     length(x) == 2  && return x
    #     return [x[1],x[end],x[2:end-1]...]
    # end
    expr = [:(SA[$(x...)]) for x in e]
    return :(tuple($(expr...)))
end

@generated function idof_by_edge_with_bounds(
    fs::FS,
    shape::Shape,
) where {FS <: FunctionSpace{<:Lagrange}, Shape <: Union{Square, Cube}}
    return _idof_by_edge_with_bounds(fs(), shape())
end

# idof_by_face
function __idof_by_face(n1, n2, n3, range1, range2, range3)
    I  = LinearIndices((1:n1, 1:n2, 1:n3))
    f5 = I[1, range2, range3]  # face x⁻
    f3 = I[n1, range2, range3]  # face x⁺
    f2 = I[range1, 1, range3]  # face y⁻
    f4 = I[range1, n2, range3]  # face y⁺
    f1 = I[range1, range2, 1]  # face z⁻
    f6 = I[range1, range2, n3]  # face z⁺
    return f1, f2, f3, f4, f5, f6
end

function _idof_by_face(fs::FunctionSpace{<:Lagrange, degree}, shape::Cube) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n = _get_num_nodes_per_dim(quadrule)
    range = ntuple(i -> 2:(n[i] - 1), length(n))
    f = vec.(__idof_by_face(n..., range...))
    expr = [:(SA[$(x...)]) for x in f]
    return :(tuple($(expr...)))
end

function idof_by_face(::FunctionSpace{<:Lagrange, 0}, shape::Union{Line, Square, Cube})
    ntuple(i -> SA[], nfaces(shape))
end

@generated function idof_by_face(fs::FunctionSpace{<:Lagrange}, shape::Union{Cube})
    return _idof_by_face(fs(), shape())
end

# idof_by_face_with_bounds
function _idof_by_face_with_bounds(
    fs::FunctionSpace{<:Lagrange, degree},
    shape::Cube,
) where {degree}
    quadtype = lagrange_quadrature_type(fs)
    quadrule = QuadratureRule(shape, Quadrature(quadtype, Val(degree)))
    n = _get_num_nodes_per_dim(quadrule)
    range = ntuple(i -> 1:n[i], length(n))
    f = vec.(__idof_by_face(n..., range...))
    expr = [:(SA[$(x...)]) for x in f]
    return :(tuple($(expr...)))
end

function idof_by_face_with_bounds(
    ::FunctionSpace{<:Lagrange, 0},
    shape::Union{Line, Square, Cube},
)
    ntuple(i -> SA[], nfaces(shape))
end

@generated function idof_by_face_with_bounds(fs::FunctionSpace{<:Lagrange}, shape::Cube)
    return _idof_by_face_with_bounds(fs(), shape())
end

# get_coords
# Rq: it's tempting to use `Union` in some places, but it leads to ambiguities
function _make_staticvector(x)
    if isa(x, SVector) || !(length(x) < MAX_LENGTH_STATICARRAY)
        return x
    else
        return SA[x]
    end
end

function _coords_gen_impl(shape::AbstractShape, quad::AbstractQuadrature)
    quadrule = QuadratureRule(shape, quad)
    nodes = _make_staticvector.(get_nodes(quadrule))
    return :(tuple($(nodes...)))
end

@generated function _coords_gen(
    ::Shape,
    ::Quad,
) where {Shape <: AbstractShape, Quad <: AbstractQuadrature}
    _coords_gen_impl(Shape(), Quad())
end

function _get_coords(
    fs::FunctionSpace{<:Lagrange, D},
    shape::Union{Line, Square, Cube},
) where {D}
    quadtype = lagrange_quadrature_type(fs)
    quad = Quadrature(quadtype, Val(D))
    _coords_gen(shape, quad)
end

get_coords(::FunctionSpace{<:Lagrange, 0}, shape::Line) = (center(shape),)
get_coords(::FunctionSpace{<:Lagrange, 0}, shape::Square) = (center(shape),)
get_coords(::FunctionSpace{<:Lagrange, 0}, shape::Cube) = (center(shape),)

get_coords(fs::FunctionSpace{<:Lagrange, 1}, shape::Line) = _get_coords(fs, shape)
get_coords(fs::FunctionSpace{<:Lagrange, 1}, shape::Square) = _get_coords(fs, shape)
get_coords(fs::FunctionSpace{<:Lagrange, 1}, shape::Cube) = _get_coords(fs, shape)

get_coords(fs::FunctionSpace{<:Lagrange, D}, shape::Line) where {D} = _get_coords(fs, shape)
function get_coords(fs::FunctionSpace{<:Lagrange, D}, shape::Square) where {D}
    _get_coords(fs, shape)
end
get_coords(fs::FunctionSpace{<:Lagrange, D}, shape::Cube) where {D} = _get_coords(fs, shape)