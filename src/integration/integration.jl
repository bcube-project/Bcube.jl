abstract type AbstractComputeQuadratureStyle end
#struct BroadcastComputeQuadratureStyle <: AbstractComputeQuadratureStyle end
struct MapComputeQuadratureStyle <: AbstractComputeQuadratureStyle end
ComputeQuadratureStyle(::Type{<:Any}) = MapComputeQuadratureStyle() #default
ComputeQuadratureStyle(op) = ComputeQuadratureStyle(typeof(op))

"""
    integrate_on_ref_element(g, cellinfo::CellInfo, quadrature::AbstractQuadrature, [::T]) where {N,[T<:AbstractComputeQuadratureStyle]}
    integrate_on_ref_element(g, faceinfo::FaceInfo, quadrature::AbstractQuadrature, [::T]) where {N,[T<:AbstractComputeQuadratureStyle]}

Integrate a function `g` over a cell/face described by `cellinfo`/`faceinfo`.

The function `g` can be expressed in the reference or the physical space corresponding to the cell,
both cases are automatically handled by applying necessary mapping when needed.

If the last argument is given, computation is optimized according to the given concrete type `T<:AbstractComputeQuadratureStyle`.
"""
function integrate_on_ref_element(g, eltInfo, quadrature::AbstractQuadrature)
    integrate_on_ref_element(g, eltInfo, quadrature, ComputeQuadratureStyle(g))
end

function integrate_on_ref_element(
    g,
    cInfo::CellInfo,
    quadrature::AbstractQuadrature,
    mapstyle::MapComputeQuadratureStyle,
)
    _g(ξ) = g(CellPoint(ξ, cInfo, ReferenceDomain()))
    integrate_on_ref_element(_g, celltype(cInfo), nodes(cInfo), quadrature, mapstyle)
end

function integrate_on_ref_element(
    g,
    fInfo::FaceInfo,
    quadrature::AbstractQuadrature,
    mapstyle::MapComputeQuadratureStyle,
)
    _g(ξ) = g(FacePoint(ξ, fInfo, ReferenceDomain()))
    integrate_on_ref_element(_g, facetype(fInfo), nodes(fInfo), quadrature, mapstyle)
end

function integrate_on_ref_element(
    g,
    ctype,
    cnodes,
    quadrature::AbstractQuadrature,
    mapstyle::MapComputeQuadratureStyle,
)
    f = Base.Fix1(_apply_metric, (g, ctype, cnodes))
    int = apply_quadrature(f, ctype, cnodes, quadrature, mapstyle)
    return int
end

# Integration on a node is immediate
function integrate_on_ref_element(
    g,
    ::AbstractEntityType{0},
    cnodes,
    ::AbstractQuadrature,
    ::MapComputeQuadratureStyle,
)
    # Whatever the "reference coordinate" ξ that we choose, it will always map to the correct
    # node physical coordinate. So we chose to evaluate in ξ = 0.
    return g(0.0)
end

function _apply_metric(g_and_c::T, qnode::T1) where {T, T1}
    g, ctype, cnodes = g_and_c
    ξ = get_coord(qnode)
    m = mapping_det_jacobian(topology_style(ctype, cnodes), ctype, cnodes, ξ)
    m * g(ξ)
end

"""
    apply_quadrature(
        g_ref,
        shape::AbstractShape,
        quadrature::AbstractQuadrature,
        ::MapComputeQuadratureStyle,
    )

Apply quadrature rule to function `g_ref` expressed on reference shape `shape`.
Computation is optimized according to the given concrete type `T<:AbstractComputeQuadratureStyle`.
"""
function apply_quadrature(
    g_ref,
    shape::AbstractShape,
    quadrature::AbstractQuadrature,
    ::MapComputeQuadratureStyle,
)
    quadrule = QuadratureRule(shape, quadrature)
    # --> TEMPORARY: ALTERING THE QUADNODES TO BYPASS OPERATORS / TestFunctionInterpolator
    quadnodes = map(get_coord, get_quadnodes(quadrule))
    # <-- TEMPORARY:
    _apply_quadrature(g_ref, get_weights(quadrule), quadnodes, g_ref(quadnodes[1]))
end
# splitting the previous function to have function barrier...
@inline function _apply_quadrature(g_ref, ω, xq::SVector{N}, ::T) where {N, T}
    u = map(g_ref, xq)::SVector{N, T}
    _apply_quadrature1(ω, u)
end
@inline function _apply_quadrature1(ω, u)
    wu = map(_mapquad, ω, u)
    _apply_quadrature2(wu)
end
_apply_quadrature2(wu) = reduce(_mapsum, wu)
_mapsum(a, b) = map(+, a, b)
_mapquad(ω, u) = map(Base.Fix1(*, ω), u)

"""
    apply_quadrature_v2(
        g_ref,
        shape::AbstractShape,
        quadrature::AbstractQuadrature,
        ::MapComputeQuadratureStyle,
    )

Alternative version of `apply_quadrature` thats seems to be more efficient
for face integration (this observation is not really understood)
"""
function apply_quadrature_v2(
    g_ref,
    shape::AbstractShape,
    quadrature::AbstractQuadrature,
    ::MapComputeQuadratureStyle,
)
    quadrule = QuadratureRule(shape, quadrature)
    # --> TEMPORARY: ALTERING THE QUADNODES TO BYPASS OPERATORS / TestFunctionInterpolator
    quadnodes = map(get_coord, get_quadnodes(quadrule))
    # <-- TEMPORARY:
    _apply_quadrature_v2(g_ref, get_weights(quadrule), quadnodes)
end
# splitting the previous function to have function barrier...
function _apply_quadrature_v2(g_ref, ω, xq)
    fquad = (w, x) -> _mapquad(w, g_ref(x))
    mapreduce(fquad, _mapsum, ω, xq)
end

# Below are "dispatch functions" to select the more appropriate quadrature function
function apply_quadrature(
    g,
    ctype::AbstractEntityType,
    cnodes,
    quadrature::AbstractQuadrature,
    qStyle::AbstractComputeQuadratureStyle,
)
    apply_quadrature(topology_style(ctype, cnodes), g, shape(ctype), quadrature, qStyle)
end

function apply_quadrature(
    ::isVolumic,
    g,
    shape::AbstractShape,
    quadrature::AbstractQuadrature,
    qStyle::MapComputeQuadratureStyle,
)
    apply_quadrature(g, shape, quadrature, qStyle)
end

function apply_quadrature(
    ::isSurfacic,
    g,
    shape::AbstractShape,
    quadrature::AbstractQuadrature,
    qStyle::MapComputeQuadratureStyle,
)
    apply_quadrature_v2(g, shape, quadrature, qStyle)
end

function apply_quadrature(
    ::isCurvilinear,
    g,
    shape::AbstractShape,
    quadrature::AbstractQuadrature,
    qStyle::MapComputeQuadratureStyle,
)
    apply_quadrature_v2(g, shape, quadrature, qStyle)
end

struct Integrand{N, F}
    f::F
end
Integrand(f::Tuple) = Integrand{length(f), typeof(f)}(f)
Integrand(f) = Integrand{1, typeof(f)}(f)
get_function(integrand::Integrand) = integrand.f
Base.getindex(integrand::Integrand, i) = get_function(integrand)[i]
@inline function Base.getindex(integrand::Integrand{N, F}, i) where {N, F <: Tuple}
    map(_f -> _f[i], get_function(integrand))
end

const ∫ = Integrand

"""
    -(a::Integrand)

Soustraction on an `Integrand` is treated as a multiplication by "(-1)" :
`-a ≡ ((-1)*a)`
"""
Base.:-(a::Integrand) = Integrand((-1) * get_function(a))
Base.:+(a::Integrand) = a

struct Integration{I <: Integrand, M <: Measure}
    integrand::I
    measure::M
end
get_integrand(integration::Integration) = integration.integrand
get_measure(integration::Integration) = integration.measure
Base.getindex(integration::Integration, i) = get_integrand(integration)[i]

Base.:*(integrand::Integrand, measure::Measure) = Integration(integrand, measure)

"""
    *(a::Number, b::Integration)
    *(a::Integration, b::Number)

Multiplication of an `Integration` is based on a rewriting rule
following the linearity rules of integration :
`k*∫(f(x))dx => ∫(k*f(x))dx`
"""
function Base.:*(a::Number, b::Integration)
    Integration(Integrand(a * get_function(get_integrand(b))), get_measure(b))
end
function Base.:*(a::Integration, b::Number)
    Integration(Integrand(get_function(get_integrand(a)) * b), get_measure(a))
end

"""
    -(a::Integration)

Soustraction on an `Integration` is treated as a multiplication by "(-1)" :
`-a ≡ ((-1)*a)`
"""
Base.:-(a::Integration) = (-1) * a
Base.:+(a::Integration) = a

struct MultiIntegration{N, I <: Tuple{Vararg{Integration, N}}}
    integrations::I
end

# implement AbstractLazy inteface:
LazyOperators.pretty_name(a::Integration) = "Integration: " * pretty_name(get_measure(a))
LazyOperators.pretty_name_style(a::Integration) = Dict(:color => :yellow)
function LazyOperators.show_lazy_operator(
    a::Integration;
    level = 1,
    indent = 4,
    islast = (true,),
)
    level == 1 && println("\n---------------")
    print_tree_prefix(level, indent, islast)
    printstyled(pretty_name(a) * "  \n"; pretty_name_style(a)...)
    show_lazy_operator(
        get_function(get_integrand(a));
        level = level + 1,
        indent = indent,
        islast = (islast..., true),
    )
    level == 1 && println("---------------")
end

function MultiIntegration(a::NTuple{N, <:Integration}) where {N}
    MultiIntegration{length(a), typeof(a)}(a)
end
function MultiIntegration(a::Integration, b::Vararg{Integration, N}) where {N}
    MultiIntegration((a, b...))
end

Base.getindex(a::MultiIntegration, ::Val{I}) where {I} = a.integrations[I]
Base.iterate(a::MultiIntegration, i) = iterate(a.integrations, i)
Base.iterate(a::MultiIntegration) = iterate(a.integrations)

# rewriting rules to deal with the additions of several `Integration`
Base.:+(a::Integration, b::Integration) = MultiIntegration(a, b)
Base.:+(a::MultiIntegration, b::Integration) = MultiIntegration(a.integrations..., b)
Base.:+(a::Integration, b::MultiIntegration) = MultiIntegration(a, b.integrations...)
function Base.:+(a::MultiIntegration, b::MultiIntegration)
    MultiIntegration((a.integrations..., b.integrations...))
end
Base.:+(a::MultiIntegration) = a

Base.:-(a::Integration, b::Integration) = MultiIntegration(a, -b)
Base.:-(a::MultiIntegration, b::Integration) = MultiIntegration(a.integrations..., -b)
function Base.:-(a::Integration, b::MultiIntegration)
    MultiIntegration(a, map(-, b.integrations)...)
end
function Base.:-(a::MultiIntegration, b::MultiIntegration)
    MultiIntegration((a.integrations..., map(-, b.integrations)...))
end
Base.:-(a::MultiIntegration) = MultiIntegration(map(-, a.integrations)...)

# implement AbstractLazy inteface:
LazyOperators.pretty_name(::MultiIntegration{N}) where {N} = "MultiIntegration: (N=$N)"
LazyOperators.pretty_name_style(a::MultiIntegration) = Dict(:color => :yellow)
function LazyOperators.show_lazy_operator(
    multiIntegration::MultiIntegration;
    level = 1,
    indent = 4,
    islast = (true,),
)
    level == 1 && println("\n---------------")
    print_tree_prefix(level, indent, islast)
    printstyled(
        pretty_name(multiIntegration) * ": \n";
        pretty_name_style(multiIntegration)...,
    )
    show_lazy_operator(
        multiIntegration.integrations;
        level = level + 1,
        islast = islast,
        printTupleOp = false,
    )
    level == 1 && println("---------------")
    nothing
end
