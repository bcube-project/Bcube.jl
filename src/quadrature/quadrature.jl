abstract type AbstractQuadratureType end
struct QuadratureLegendre <: AbstractQuadratureType end
struct QuadratureLobatto <: AbstractQuadratureType end
struct QuadratureUniform <: AbstractQuadratureType end

"""
    AbstractQuadrature{T,D}

Abstract type representing quadrature of type `T`
and degree `D`
"""
abstract type AbstractQuadrature{T, D} end
get_quadtype(::AbstractQuadrature{T}) where {T} = T()
get_degree(::AbstractQuadrature{T, D}) where {T, D} = D

"""
    Quadrature{T,D}

Quadrature of type `T` and degree `D`
"""
struct Quadrature{T, D} <: AbstractQuadrature{T, D} end

function Quadrature(qt::AbstractQuadratureType, ::Val{D}) where {D}
    @assert D isa Integer && D ≥ 0 "'D' must be a positive Integer"
    Quadrature{typeof(qt), D}()
end
Quadrature(d::Val{D}) where {D} = Quadrature(QuadratureLegendre(), d)

Quadrature(qt::AbstractQuadratureType, d::Integer) = Quadrature(qt, Val(d))
Quadrature(d::Integer) = Quadrature(Val(d))

"""
    AbstractQuadratureRule{S,Q}

Abstract type representing a quadrature rule for a shape `S`
and quadrature `Q`.

Derived types must implement the following method:
    - [`get_weights(qr::AbstractQuadratureRule)`]
    - [`get_nodes(qr::AbstractQuadratureRule)`]
    - [`Base.length(qr::AbstractQuadratureRule)`]
"""
abstract type AbstractQuadratureRule{S, Q} end
get_shape(::AbstractQuadratureRule{S, Q}) where {S, Q} = S
get_quadrature(::AbstractQuadratureRule{S, Q}) where {S, Q} = Q

"""
    get_weights(qr::AbstractQuadratureRule)

Returns an array containing the weights of all quadrature nodes of `qr`.
"""
function get_weights(qr::AbstractQuadratureRule)
    error("`get_weights` is not implemented for $(typeof(qr))")
end

"""
    get_nodes(qr::AbstractQuadratureRule)

Returns an array containing the coordinates of all quadrature nodes of `qr`.
"""
function get_nodes(qr::AbstractQuadratureRule)
    error("`get_nodes` is not implemented for $(typeof(qr))")
end

"""
    length(qr::AbstractQuadratureRule)

Returns the number of quadrature nodes of `qr`.
"""
function Base.length(qr::AbstractQuadratureRule)
    error("`length` is not implemented for $(typeof(qr))")
end

"""
    QuadratureRule{S,Q}

Abstract type representing a quadrature rule for a shape `S`
and quadrature `Q`
"""
struct QuadratureRule{S, Q, N, W <: AbstractArray, X <: AbstractArray} <:
       AbstractQuadratureRule{S, Q}
    weights::W
    nodes::X
end
get_weights(qr::QuadratureRule) = qr.weights
get_nodes(qr::QuadratureRule) = qr.nodes
Base.length(::QuadratureRule{S, Q, N}) where {S, Q, N} = N

"""
    QuadratureRule(shape::AbstractShape, q::AbstractQuadrature)

Return the quadrature rule corresponding to the given `Shape` according to
the selected quadrature 'q'.
"""
function QuadratureRule(shape::AbstractShape, q::AbstractQuadrature)
    w, x = quadrature_rule(shape, Val(get_degree(q)), get_quadtype(q))
    QuadratureRule(shape, q, w, x)
end

function QuadratureRule(shape::AbstractShape, degree::Val{D}) where {D}
    QuadratureRule(shape, Quadrature(degree))
end

function QuadratureRule(
    shape::AbstractShape,
    q::AbstractQuadrature,
    w::AbstractVector,
    x::AbstractVector,
)
    @assert length(w) == length(x) "Dimension mismatch for 'w' and 'x'"
    QuadratureRule{typeof(shape), typeof(q), length(x), typeof(w), typeof(x)}(w, x)
end

function QuadratureRule(
    shape::AbstractShape,
    q::AbstractQuadrature,
    w::NTuple{N, T1},
    x::NTuple{N, T2},
) where {N, T1, T2}
    _w = SVector{N, T1}(w)
    _x = SVector{N, T2}(x)
    QuadratureRule(shape, q, _w, _x)
end

function QuadratureRule(shape::AbstractShape, degree::Integer)
    QuadratureRule(shape, Quadrature(degree))
end

# To integrate a constant, we use the first order quadrature
function QuadratureRule(
    shape::AbstractShape,
    ::Q,
) where {Q <: AbstractQuadrature{T, 0}} where {T}
    QuadratureRule(shape, Q(T, 1))
end

"""
    quadrature_rule_bary(::Int, ::AbstractShape, degree::Val{N}) where N

Return the quadrature rule, computed with barycentric coefficients, corresponding to the given boundary of a
shape and the given degree.

This function returns the quadrature weights and the barycentric weights to apply
to each vertex of the reference shape. Hence, to apply the quadrature using this function,
one needs to do :
`
for (weight, l) in quadrature_rule_bary(iside, shape(etype), degree)
    xp = zeros(SVector{td})
    for i=1:nvertices
        xp += l[i]*vertices[i]
    end
    # weight, xp is the quadrature couple (weight, node)
end
`
"""
function quadrature_rule_bary(::Int, shape::AbstractShape, degree::Val{N}) where {N}
    error(
        "Function 'quadrature_rule_bary' is not implemented for shape $shape and degree $N",
    )
end

"""
    quadrature_rule(iside::Int, shape::AbstractShape, degree::Val{N}) where N

Return the quadrature rule, computed with barycentric coefficients, corresponding to
the given boundary of a shape and the given `degree`.
"""
function quadrature_rule(iside::Int, shape::AbstractShape, degree::Val{N}) where {N}
    # Get coordinates of the face's nodes of the shape
    fvertices = get_coords(shape, faces2nodes(shape)[iside])

    # Get barycentric quadrature
    quadBary = quadrature_rule_bary(iside, shape, degree)

    nq = length(quadBary)

    weights = SVector{nq, eltype(first(quadBary)[1])}(w for (w, l) in quadBary)
    xq = SVector{nq, eltype(fvertices)}(sum(l .* fvertices) for (w, l) in quadBary)

    return weights, xq
end

# Point quadratures
function quadrature_rule(::Point, ::Val{D}, ::AbstractQuadratureType) where {D}
    return SA[1.0], SA[0.0]
end

# Line quadratures
"""
    _gausslegendre1D(::Val{N}) where N
    _gausslobatto1D(::Val{N}) where N

Return `N`-point Gauss quadrature weights and nodes on the domain [-1:1].
"""
@generated function _gausslegendre1D(::Val{N}) where {N}
    x, w = FastGaussQuadrature.gausslegendre(N)
    w0 = SVector{length(w), eltype(w)}(w)
    x0 = SVector{length(x), eltype(x)}(x)
    return :($w0, $x0)
end

@generated function _gausslobatto1D(::Val{N}) where {N}
    x, w = FastGaussQuadrature.gausslobatto(N)
    w0 = SVector{length(w), eltype(w)}(w)
    x0 = SVector{length(x), eltype(x)}(x)
    return :($w0, $x0)
end

@generated function _uniformpoints(::Val{N}) where {N}
    xmin = -1.0
    xmax = 1.0
    if N == 1
        x0 = SA[(xmin + xmax) / 2]
    else
        x0 = SVector{N}(collect(LinRange(xmin, xmax, N)))
    end
    w0 = @. one(x0) / N
    return :($w0, $x0)
end

_quadrature_rule(n::Val{N}, ::QuadratureLegendre) where {N} = _gausslegendre1D(n)
_quadrature_rule(n::Val{N}, ::QuadratureLobatto) where {N} = _gausslobatto1D(n)
_quadrature_rule(n::Val{N}, ::QuadratureUniform) where {N} = _uniformpoints(n)

"""
Gauss-Legendre formula with ``n`` nodes has degree of exactness ``2n-1``.
Then, to obtain a given degree ``D``, the number of nodes must satisfy:
`` 2n-1 ≥ D`` or equivalently ``n ≥ (D+1)/2``
"""
function _get_num_nodes(::Line, ::Val{degree}, ::QuadratureLegendre) where {degree}
    ceil(Int, (degree + 1) / 2)
end

"""
Gauss-Lobatto formula with ``n+1`` nodes has degree of exactness ``2n-1``,
which equivalent to a degree of ``2n-3`` with ``n`` nodes.
Then, to obtain a given degree ``D``, the number of nodes must satisfy:
`` 2n-3 ≥ D`` or equivalently ``n ≥ (D+3)/2``
"""
function _get_num_nodes(::Line, ::Val{degree}, ::QuadratureLobatto) where {degree}
    ceil(Int, (degree + 3) / 2)
end
_get_num_nodes(::Line, ::Val{degree}, ::QuadratureUniform) where {degree} = degree + 1

"""
    get_num_nodes_per_dim(quadrule::AbstractQuadratureRule{S}) where S<:Shape

Returns the number of nodes per dimension. This function is defined for shapes for which quadratures
are based on a cartesian product : `Line`, `Square`, `Cube`

Remark :
Here we assume that the same `degree` is used along each dimension (no anisotropy for now!)
"""
function _get_num_nodes_per_dim(::AbstractQuadratureRule{<:S}) where {S}
    error("Not implemented for shape $S")
end
_get_num_nodes_per_dim(quadrule::AbstractQuadratureRule{<:Line}) = length(quadrule)
function _get_num_nodes_per_dim(::AbstractQuadratureRule{<:Square, Q}) where {Q}
    ntuple(i -> _get_num_nodes_per_dim(QuadratureRule(Line(), Q())), Val(2))
end
function _get_num_nodes_per_dim(::AbstractQuadratureRule{<:Cube, Q}) where {Q}
    ntuple(i -> _get_num_nodes_per_dim(QuadratureRule(Line(), Q())), Val(3))
end

function quadrature_rule(line::Line, degree::Val{D}, quad::AbstractQuadratureType) where {D}
    @assert D isa Integer && D ≥ 0 "'D' must be a positive Integer"
    n = _get_num_nodes(line, degree, quad)
    return _quadrature_rule(Val(n), quad)
end

# Triangle quadratures
function quadrature_rule(::Triangle, ::Val{1}, ::QuadratureLegendre)
    raw_unzip(get_quadrature_points(Val{:GLTRI1}))
end
function quadrature_rule(::Triangle, ::Val{2}, ::QuadratureLegendre)
    raw_unzip(get_quadrature_points(Val{:GLTRI3}))
end
function quadrature_rule(::Triangle, ::Val{3}, ::QuadratureLegendre)
    raw_unzip(get_quadrature_points(Val{:GLTRI4}))
end
function quadrature_rule(::Triangle, ::Val{4}, ::QuadratureLegendre)
    get_quadrature_points_gausslegendre(Val{:GLTRI6})
end #quadrature_points(Triangle(), Val(4), Val(:Lobatto))
function quadrature_rule(::Triangle, ::Val{5}, ::QuadratureLegendre)
    get_quadrature_points_gausslegendre(Val{:GLTRI7})
end
function quadrature_rule(::Triangle, ::Val{6}, ::QuadratureLegendre)
    get_quadrature_points_gausslegendre(Val{:GLTRI12})
end
function quadrature_rule(::Triangle, ::Val{7}, ::QuadratureLegendre)
    get_quadrature_points_gausslegendre(Val{:GLTRI16})
end
function quadrature_rule(::Triangle, ::Val{8}, ::QuadratureLegendre)
    get_quadrature_points_gausslegendre(Val{:GLTRI16})
end

function get_quadrature_points_gausslegendre(::Type{Val{:GLTRI6}})
    P1      = 0.22338158967801146569500700843312 / 2
    P2      = 0.10995174365532186763832632490021 / 2
    A       = 0.44594849091596488631832925388305
    B       = 0.09157621350977074345957146340220
    weights = SA[P2, P2, P2, P1, P1, P1]
    points  = SA[(B, B), (1.0 - 2.0 * B, B), (B, 1.0 - 2.0 * B), (A, 1.0 - 2 * A), (A, A), (1.0 - 2.0 * A, A)]
    return weights, points
end

""" Gauss-Legendre quadrature, 7 point rule on triangle. """
function get_quadrature_points_gausslegendre(::Type{Val{:GLTRI7}})
    A       = 0.47014206410511508977044120951345
    B       = 0.10128650732345633880098736191512
    P1      = 0.13239415278850618073764938783315 / 2
    P2      = 0.12593918054482715259568394550018 / 2
    weights = SA[9 / 80, P1, P1, P1, P2, P2, P2]
    points  = SA[(1 / 3, 1 / 3), (A, A), (1.0 - 2.0 * A, A), (A, 1.0 - 2.0 * A), (B, B), (1.0 - 2.0 * B, B), (B, 1.0 - 2.0 * B)]
    return weights, points
end

""" Gauss-Legendre quadrature, 12 point rule on triangle.

Ref: Witherden, F. D.; Vincent, P. E.
     On the identification of symmetric quadrature rules for finite element methods.
     Comput. Math. Appl. 69 (2015), no. 10, 1232–1241
"""
function get_quadrature_points_gausslegendre(::Type{Val{:GLTRI12}})
    A       = (-0.87382197101699554331933679425836168532 + 1) / 2
    B       = (-0.50142650965817915741672289378596184782 + 1) / 2
    C       = (-0.37929509793243118916678453208689569359 + 1) / 2
    D       = (-0.89370990031036610529350065673720370601 + 1) / 2
    P1      = 0.10168981274041363384187361821373796809 / 4
    P2      = 0.23357255145275873205057922277115888265 / 4
    P3      = 0.16570215123674715038710691284088490796 / 4
    weights = SA[P1, P1, P1, P2, P2, P2, P3, P3, P3, P3, P3, P3]
    points  = SA[(A, A), (1.0 - 2.0 * A, A), (A, 1.0 - 2.0 * A), (B, B), (1.0 - 2.0 * B, B), (B, 1.0 - 2.0 * B), (C, D), (D, C), (1.0 - C - D, C), (1.0 - C - D, D), (C, 1.0 - C - D), (D, 1.0 - C - D)]
    return weights, points
end

""" Gauss-Legendre quadrature, 16 point rule on triangle, degree 8.

Ref: Witherden, F. D.; Vincent, P. E.
     On the identification of symmetric quadrature rules for finite element methods.
     Comput. Math. Appl. 69 (2015), no. 10, 1232–1241

Note : quadrature is rescale to match our reference triangular shape
which is defined in [0:1]² instead of [-1:1]²
"""
function get_quadrature_points_gausslegendre(::Type{Val{:GLTRI16}})
    x = SA[
        -0.33333333333333333333333333333333333333 -0.33333333333333333333333333333333333333 0.2886312153555743365021822209781292496
        -0.081414823414553687942368971011661355879 -0.83717035317089262411526205797667728824 0.1901832685345692495877922087771686332
        -0.83717035317089262411526205797667728824 -0.081414823414553687942368971011661355879 0.1901832685345692495877922087771686332
        -0.081414823414553687942368971011661355879 -0.081414823414553687942368971011661355879 0.1901832685345692495877922087771686332
        -0.65886138449647958675541299701707099796 0.31772276899295917351082599403414199593 0.20643474106943650056358310058425806003
        0.31772276899295917351082599403414199593 -0.65886138449647958675541299701707099796 0.20643474106943650056358310058425806003
        -0.65886138449647958675541299701707099796 -0.65886138449647958675541299701707099796 0.20643474106943650056358310058425806003
        -0.89890554336593804908315289880680210631 0.79781108673187609816630579761360421262 0.064916995246396160621851856683561193593
        0.79781108673187609816630579761360421262 -0.89890554336593804908315289880680210631 0.064916995246396160621851856683561193593
        -0.89890554336593804908315289880680210631 -0.89890554336593804908315289880680210631 0.064916995246396160621851856683561193593
        -0.98321044518008478932557233092141110162 0.45698478591080856248200075835212392604 0.05446062834886998852968938014781784832
        0.45698478591080856248200075835212392604 -0.98321044518008478932557233092141110162 0.05446062834886998852968938014781784832
        -0.47377434073072377315642842743071282442 0.45698478591080856248200075835212392604 0.05446062834886998852968938014781784832
        0.45698478591080856248200075835212392604 -0.47377434073072377315642842743071282442 0.05446062834886998852968938014781784832
        -0.47377434073072377315642842743071282442 -0.98321044518008478932557233092141110162 0.05446062834886998852968938014781784832
        -0.98321044518008478932557233092141110162 -0.47377434073072377315642842743071282442 0.05446062834886998852968938014781784832
    ]
    points = _rescale_tri.(x[:, 1], x[:, 2])
    weights = x[:, 3] ./ 4
    return weights, points
end
_rescale_tri(x, y) = (0.5 * (x + 1), 0.5 * (y + 1))

"""
 ref : https://www.math.umd.edu/~tadmor/references/files/Chen%20&%20Shu%20entropy%20stable%20DG%20JCP2017.pdf
"""
function quadrature_points(tri::Triangle, ::Val{4}, ::QuadratureLobatto)
    p1, p2, p3 = get_coords(tri)
    s12_1 = 1.0 / 2
    s12_2 = 0.4384239524408185
    s12_3 = 0.1394337314154536
    s111_1 = (0.0, 0.230765344947159)
    s111_2 = (0.0, 0.046910077030668)
    points = SA[
        _triangle_orbit_3(p1, p2, p3),
        _triangle_orbit_12(p1, p2, p3, s12_1),
        _triangle_orbit_12(p2, p3, p1, s12_1),
        _triangle_orbit_12(p3, p1, p2, s12_1),
        _triangle_orbit_12(p1, p2, p3, s12_2),
        _triangle_orbit_12(p2, p3, p1, s12_2),
        _triangle_orbit_12(p3, p1, p2, s12_2),
        _triangle_orbit_12(p1, p2, p3, s12_3),
        _triangle_orbit_12(p2, p3, p1, s12_3),
        _triangle_orbit_12(p3, p1, p2, s12_3),
        _triangle_orbit_111(p1, p2, p3, s111_1...),
        _triangle_orbit_111(p2, p3, p1, s111_1...),
        _triangle_orbit_111(p3, p1, p2, s111_1...),
        _triangle_orbit_111(p1, p3, p2, s111_1...),
        _triangle_orbit_111(p2, p1, p3, s111_1...),
        _triangle_orbit_111(p3, p2, p1, s111_1...),
        _triangle_orbit_111(p1, p2, p3, s111_2...),
        _triangle_orbit_111(p2, p3, p1, s111_2...),
        _triangle_orbit_111(p3, p1, p2, s111_2...),
        _triangle_orbit_111(p1, p3, p2, s111_2...),
        _triangle_orbit_111(p2, p1, p3, s111_2...),
        _triangle_orbit_111(p3, p2, p1, s111_2...),
    ]

    w3, w12_1, w12_2, w12_3, w111_1, w111_2 = (
        0.0455499555988567,
        0.00926854241697489,
        0.0623683661448868,
        0.0527146648104222,
        0.0102652298402145,
        0.00330065754050081,
    )
    weights = SA[
        w3,
        w12_1,
        w12_1,
        w12_1,
        w12_2,
        w12_2,
        w12_2,
        w12_3,
        w12_3,
        w12_3,
        w111_1,
        w111_1,
        w111_1,
        w111_1,
        w111_1,
        w111_1,
        w111_2,
        w111_2,
        w111_2,
        w111_2,
        w111_2,
        w111_2,
    ]
    return weights, points
end

_triangle_orbit_3(p1, p2, p3) = 1.0 / 3 * (p1 + p2 + p3)
_triangle_orbit_12(p1, p2, p3, α) = α * p1 + α * p2 + (1.0 - 2 * α) * p3
_triangle_orbit_111(p1, p2, p3, α, β) = α * p1 + β * p2 + (1.0 - α - β) * p3

# Square quadratures : generete quadrature rules for square
# from the cartesian product of line quadratures (i.e. Q-element)
function quadrature_rule(::Square, deg::Val{D}, quadtype::AbstractQuadratureType) where {D}
    quadline = QuadratureRule(Line(), Quadrature(quadtype, deg))
    N = length(quadline)
    w1 = get_weights(quadline)
    x1 = get_nodes(quadline)
    NQ = N^2
    if NQ < MAX_LENGTH_STATICARRAY
        w = SVector{NQ}(w1[i] * w1[j] for i in 1:N, j in 1:N)
        x = SVector{NQ}(SA[x1[i], x1[j]] for i in 1:N, j in 1:N)
    else
        w = vec([w1[i] * w1[j] for i in 1:N, j in 1:N])
        x = vec([SA[x1[i], x1[j]] for i in 1:N, j in 1:N])
    end
    return w, x
end

# Cube quadratures : generete quadrature rules for cube
# from the cartesian product of line quadratures (i.e. Q-element)
function quadrature_rule(::Cube, deg::Val{D}, quadtype::AbstractQuadratureType) where {D}
    quadline = QuadratureRule(Line(), Quadrature(quadtype, deg))
    N = length(quadline)
    w1 = get_weights(quadline)
    x1 = get_nodes(quadline)
    NQ = N^3
    if NQ < MAX_LENGTH_STATICARRAY
        w = SVector{NQ}(w1[i] * w1[j] * w1[k] for i in 1:N, j in 1:N, k in 1:N)
        x = SVector{NQ}(SA[x1[i], x1[j], x1[k]] for i in 1:N, j in 1:N, k in 1:N)
    else
        w = vec([w1[i] * w1[j] * w1[k] for i in 1:N, j in 1:N, k in 1:N])
        x = vec([SA[x1[i], x1[j], x1[k]] for i in 1:N, j in 1:N, k in 1:N])
    end
    return w, x
end

# Prism quadratures
function quadrature_rule(::Prism, ::Val{1}, ::QuadratureLegendre)
    w, x = raw_unzip(get_quadrature_points(Val{:GLWED6}))
    _w = SVector{length(w), eltype(w)}(w)
    _x = SVector{length(x), eltype(x)}(x)
    return _w, _x
end
function quadrature_rule(::Prism, ::Val{2}, ::QuadratureLegendre)
    w, x = raw_unzip(get_quadrature_points(Val{:GLWED21}))
    _w = SVector{length(w), eltype(w)}(w)
    _x = SVector{length(x), eltype(x)}(x)
    return _w, _x
end
# To be completed (available in FEMQuad): tetra, hexa, prism

# Conversion to barycentric rules for 'face' integration
function quadrature_rule_bary(iside::Int, shape::Triangle, ::Val{1})
    area = face_area(shape)[iside]
    weights = area .* (1.0,)
    baryCoords = ((0.5, 0.5),)
    return weights, baryCoords
end

function quadrature_rule_bary(iside::Int, shape::Triangle, ::Val{2})
    area = face_area(shape)[iside]
    weights = area .* (0.5, 0.5)
    baryCoords = (
        (0.7886751345948129, 0.21132486540518708),
        (0.21132486540518713, 0.7886751345948129),
    )
    return weights, baryCoords
end

function quadrature_rule_bary(iside::Int, shape::Triangle, ::Val{3})
    area = face_area(shape)[iside]
    weights = area .* (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)
    baryCoords = (
        (0.8872983346207417, 0.1127016653792583),
        (0.5, 0.5),
        (0.1127016653792583, 0.8872983346207417),
    )
    return weights, baryCoords
end

function quadrature_rule_bary(iside::Int, shape::Square, ::Val{1})
    area = face_area(shape)[iside]
    weights = area .* (1.0,)
    baryCoords = ((0.5, 0.5),)
    return weights, baryCoords
end

function quadrature_rule_bary(iside::Int, shape::Square, ::Val{2})
    area = face_area(shape)[iside]
    weights = area .* (0.5, 0.5)
    baryCoords = (
        (0.7886751345948129, 0.21132486540518708),
        (0.21132486540518713, 0.7886751345948129),
    )
    return weights, baryCoords
end

function quadrature_rule_bary(iside::Int, shape::Square, ::Val{3})
    area = face_area(shape)[iside]
    weights = area .* (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)
    baryCoords = (
        (0.8872983346207417, 0.1127016653792583),
        (0.5, 0.5),
        (0.1127016653792583, 0.8872983346207417),
    )
    return weights, baryCoords
end

function quadrature_rule_bary(iside::Int, shape::Tetra, ::Val{1})
    area = face_area(shape)[iside]
    weights = area .* (1.0,)
    baryCoords = ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),)
    return weights, baryCoords
end

function quadrature_rule_bary(iside::Int, shape::Tetra, ::Val{2})
    area = face_area(shape)[iside]
    weights = area .* (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    baryCoords = (
        (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
    )
    return weights, baryCoords
end

function quadrature_rule(::Tetra, ::Val{1}, ::QuadratureLegendre)
    raw_unzip(get_quadrature_points(Val{:GLTET1}))
end
function quadrature_rule(::Tetra, ::Val{2}, ::QuadratureLegendre)
    raw_unzip(get_quadrature_points(Val{:GLTET4}))
end
function quadrature_rule(::Tetra, ::Val{3}, ::QuadratureLegendre)
    raw_unzip(get_quadrature_points(Val{:GLTET5}))
end
function quadrature_rule(::Tetra, ::Val{4}, ::QuadratureLegendre)
    raw_unzip(get_quadrature_points(Val{:GLTET15}))
end

#  To be adapted to implement integration on square faces
#function quadrature_rule(::Square, ::Val{1}, ::Barycentric)
#    weights = (1.0,)
#    baryCoords = ((0.25, 0.25, 0.25, 0.25),)
#    return zip(weights, baryCoords)
#end
#
#function quadrature_rule(::Square, ::Val{2}, ::Barycentric)
#    weights = (0.25, 0.25, 0.25, 0.25)
#    w1 = sqrt(3.0)/3.0
#    w2 = 0.5*(1.0-w1)
#    baryCoords = (
#        (w1, w2, 0.0, w2),
#        (w2, w1, w2, 0.0),
#        (0.0, w2, w1, w2),
#        (w2, 0.0, w2, w1),
#    )
#    return zip(weights, baryCoords)
#end

"""
    AbstractQuadratureNode{S,Q}

Abstract type representing a quadrature node
for a shape `S` and a quadrature `Q`.
This type is used to represent and identify
easily a quadrature node in a quadrature rules.

Derived types must implement the following method:

    - get_index(quadnode::AbstractQuadratureNode{S,Q})
    - get_coord(quadnode::AbstractQuadratureNode)
"""
abstract type AbstractQuadratureNode{S, Q} end

get_shape(::AbstractQuadratureNode{S, Q}) where {S, Q} = S
get_quadrature(::AbstractQuadratureNode{S, Q}) where {S, Q} = Q

"""
    get_index(quadnode::AbstractQuadratureNode{S,Q})

Returns the index of `quadnode` in the parent
quadrature rule `AbstractQuadRules{S,Q}`
"""
function get_index(quadnode::AbstractQuadratureNode)
    error("'get_index' is not defined for type $(typeof(quadnode))")
end

"""
    get_coord(quadnode::AbstractQuadratureNode)

Returns the coordinates of `quadnode`.
"""
function get_coord(quadnode::AbstractQuadratureNode)
    error("'get_coord' is not defined for type $(typeof(quadnode))")
end

"""
    evalquadnode(f, quadnode::AbstractQuadratureNode)

Evaluate the function `f` at the coordinates of `quadnode`.

Basically, it computes:
```jl
f(get_coord(quadnode))
```

# Remark:

Optimization could be applied if `f` is a function
based on a nodal basis such as one of the DoF and `quadnode` are collocated.
"""
evalquadnode(f, quadnode::AbstractQuadratureNode) = f(get_coord(quadnode))
get_quadrature_rule(::AbstractQuadratureNode{S, Q}) where {S, Q} = QuadratureRule(S(), Q())

# default rules for retro-compatibility
get_coord(x::Number) = x
get_coord(x::AbstractArray{T}) where {T <: Number} = x

Base.getindex(quadnode::AbstractQuadratureNode, i) = get_coord(quadnode)[i]
Base.IteratorEltype(x::Bcube.AbstractQuadratureNode) = Base.IteratorEltype(get_coord(x))
Base.IteratorSize(x::Bcube.AbstractQuadratureNode) = Base.IteratorSize(get_coord(x))
Base.size(x::Bcube.AbstractQuadratureNode) = Base.size(get_coord(x))
Base.iterate(x::Bcube.AbstractQuadratureNode) = Base.iterate(get_coord(x))
function Base.iterate(x::Bcube.AbstractQuadratureNode, i::Integer)
    Base.iterate(get_coord(x), i::Integer)
end
function Base.Broadcast.broadcastable(x::AbstractQuadratureNode)
    Base.Broadcast.broadcastable(get_coord(x))
end

"""
    QuadratureNode{S,Q}

Type representing a quadrature node
for a shape `S` and a quadrature `Q`.
This type can be used to represent and identify
easily a quadrature node in the corresponding
parent quadrature rule.
"""
struct QuadratureNode{S, Q, Ti, Tx} <: AbstractQuadratureNode{S, Q}
    i::Ti
    x::Tx
end

function QuadratureNode(shape, quad, i::Integer, x)
    S, Q, Ti, Tx = typeof(shape), typeof(quad), typeof(i), typeof(x)
    QuadratureNode{S, Q, Ti, Tx}(i, x)
end

get_index(quadnode::QuadratureNode) = quadnode.i
get_coord(quadnode::QuadratureNode) = quadnode.x

function get_quadnodes_impl(::Type{<:QuadratureRule{S, Q, N}}) where {S, Q, N}
    quad = Q()
    _, x = quadrature_rule(S(), Val(get_degree(quad)), get_quadtype(quad))
    x = map(a -> SVector{length(a), typeof(a[1])}(a), x)
    quadnodes =
        [QuadratureNode{S, Q, typeof(i), typeof(_x)}(i, _x) for (i, _x) in enumerate(x)]
    return :(SA[$(quadnodes...)])
end

"""
    get_quadnodes(qr::QuadratureRule{S,Q,N}) where {S,Q,N}

Returns an vector containing each `QuadratureNode` of `qr`
"""
@generated function get_quadnodes(qr::QuadratureRule{S, Q, N}) where {S, Q, N}
    expr = get_quadnodes_impl(qr)
    return expr
end

# ismapover(::Tuple{Vararg{AbstractQuadratureNode}}) = MapOverStyle()

function _map_quadrature_node_impl(
    t::Type{<:Tuple{Vararg{AbstractQuadratureNode, N}}},
) where {N}
    expr = ntuple(i -> :(f(t[$i])), N)
    return :(SA[$(expr...)])
end

@generated function Base.map(f, t::Tuple{Vararg{AbstractQuadratureNode, N}}) where {N}
    return _map_quadrature_node_impl(t)
end

abstract type AbtractCollocatedStyle end
struct IsCollocatedStyle <: AbtractCollocatedStyle end
struct IsNotCollocatedStyle <: AbtractCollocatedStyle end

is_collocated(::AbstractQuadrature, ::AbstractQuadrature) = IsNotCollocatedStyle() #default
is_collocated(::Q, ::Q) where {Q <: AbstractQuadrature} = IsCollocatedStyle()

is_collocated(::AbstractQuadratureRule, ::AbstractQuadratureRule) = IsNotCollocatedStyle() #default

# for two AbstractQuadratureRule defined on the same shape:
function is_collocated(
    q1::AbstractQuadratureRule{S},
    q2::AbstractQuadratureRule{S},
) where {S}
    return is_collocated(get_quadrature(q1), get_quadrature(q2))
end
