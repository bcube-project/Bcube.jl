abstract type AbstractComputeQuadratureStyle end
#struct BroadcastComputeQuadratureStyle <: AbstractComputeQuadratureStyle end
struct MapComputeQuadratureStyle <: AbstractComputeQuadratureStyle end
ComputeQuadratureStyle(::Type{<:Any}) = MapComputeQuadratureStyle() #default
ComputeQuadratureStyle(op) = ComputeQuadratureStyle(typeof(op))

"""
    apply_quadrature(g_ref, shape::AbstractShape, quadrature::AbstractQuadrature, ::T) where{N,T<:AbstractComputeQuadratureStyle}

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
    quadnodes = map(get_coords, get_quadnodes(quadrule))
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
    apply_quadrature2(g_ref, shape::AbstractShape, quadrature::AbstractQuadrature, ::MapComputeQuadratureStyle) where{N}

Alternative version of `apply_quadrature` thats seems to be more efficient
for face integration (this observation is not really understood)
"""
function apply_quadrature2(
    g_ref,
    shape::AbstractShape,
    quadrature::AbstractQuadrature,
    ::MapComputeQuadratureStyle,
)
    quadrule = QuadratureRule(shape, quadrature)
    # --> TEMPORARY: ALTERING THE QUADNODES TO BYPASS OPERATORS / TestFunctionInterpolator
    quadnodes = map(get_coords, get_quadnodes(quadrule))
    # <-- TEMPORARY:
    _apply_quadrature_v2(g_ref, get_weights(quadrule), quadnodes)
end
# splitting the previous function to have function barrier...
function _apply_quadrature_v2(g_ref, ω, xq)
    fquad = (w, x) -> _mapquad(w, g_ref(x))
    mapreduce(fquad, _mapsum, ω, xq)
end

@inline function apply_quadrature(
    detJ,
    g_ref,
    shape::AbstractShape,
    quadrature::AbstractQuadrature,
)
    quad = QuadratureRule(shape, quadrature)
    ω = get_weights(quad)
    x = get_nodes(quad)
    res = ω[1] .* detJ(x[1]) .* g_ref(x[1])
    for i in (firstindex(x) + 1):lastindex(x)
        res = res .+ (ω[i] * detJ(x[i])) .* g_ref(x[i])
    end
    return res
end

# Alias to apply quadrature rule
function integrate(g_ref, shape::AbstractShape, quadrature::AbstractQuadrature)
    apply_quadrature(g_ref, shape, quadrature, ComputeQuadratureStyle(g_ref))
end

"""
    integrate(g, ctype::AbstractEntityType, cnodes, quadrature::AbstractQuadrature)

Integrate function `g` expressed in local element. Depending on the cell type and the space
dimension, a volumic or a 'surfacic' integration is performed.
"""
function integrate(g, ctype::AbstractEntityType, cnodes, quadrature::AbstractQuadrature)
    return integrate_ref(x -> g(mapping(ctype, cnodes, x)), ctype, cnodes, quadrature)
end

"""
    integrate_ref(g_ref, ctype::AbstractEntityType, cnodes, quadrature::AbstractQuadrature, [::T]) where {[T<:AbstractComputeQuadratureStyle]}

Integrate function `g_ref` expressed in reference element. A variable substitution (involving Jacobian & Cie) is still
applied, but the function is considered to be already mapped.

This function is helpfull to integrate shape functions (for instance ``\\int \\lambda_i \\lambda_j``) when the inverse
mapping is not known explicitely (hence only ``\\hat{lambda}`` are known, not ``\\lambda``).

If the last argument is given, computation is optimized according to the given concrete type `T<:AbstractComputeQuadratureStyle`.
"""
function integrate_ref(
    g_ref,
    ctype::AbstractEntityType,
    cnodes,
    quadrature::AbstractQuadrature,
)
    integrate_ref(g_ref, ctype, cnodes, quadrature, ComputeQuadratureStyle(g_ref))
end
function integrate_ref(
    g_ref,
    ctype::AbstractEntityType,
    cnodes,
    quadrature::AbstractQuadrature,
    cqStyle::AbstractComputeQuadratureStyle,
)
    integrate_ref(topology_style(ctype, cnodes), g_ref, ctype, cnodes, quadrature, cqStyle)
end

"""
    integrate_on_ref(g, cellinfo::CellInfo, quadrature::AbstractQuadrature, [::T]) where {N,[T<:AbstractComputeQuadratureStyle]}

Integrate a function `g` over a cell decribed by `cellinfo`. The function `g` can be expressed in the reference or
the physical space corresponding to the cell, both cases are automatically handled by applying necessary mapping when needed.

This function is helpfull to integrate shape functions (for instance ``\\int \\lambda_i \\lambda_j``) when the inverse
mapping is not known explicitely (hence only ``\\hat{lambda}`` are known, not ``\\lambda``).

If the last argument is given, computation is optimized according to the given concrete type `T<:AbstractComputeQuadratureStyle`.
"""
function integrate_on_ref(g, cellinfo::CellInfo, quadrature::AbstractQuadrature)
    integrate_on_ref(g, cellinfo, quadrature, ComputeQuadratureStyle(g))
end
function integrate_on_ref(
    g,
    cellinfo::CellInfo,
    quadrature::AbstractQuadrature,
    cqStyle::AbstractComputeQuadratureStyle,
)
    integrate_on_ref(
        topology_style(celltype(cellinfo), nodes(cellinfo)),
        g,
        cellinfo,
        quadrature,
        cqStyle,
    )
end

"""
    integrate(g, iside::Int, ctype::AbstractEntityType, cnodes, quadrature::AbstractQuadrature)

Integrate function `g` on the `iside`-th side of the cell defined by its nodes `cnodes` and its
type `ctype`. Function `g(x)` is expressed in the local element.
"""
function integrate(
    g,
    iside::Int,
    ctype::AbstractEntityType,
    cnodes,
    quadrature::AbstractQuadrature,
)
    # Get cell shape
    cshape = shape(ctype)

    # Get face type
    ftype = facetypes(ctype)[iside]

    # Get face parametrisation
    fp = mapping_face(cshape, iside) # mapping face-ref -> cell-ref

    # Since we want to integrate in the face-ref-element, and since `g` is expressed in local element,
    # we need to send x, from face-ref-element to cell-ref-element to local-element.
    # @ghislainb : need better solution to index with tuple
    return integrate_ref(
        x -> g(mapping(ctype, cnodes, fp(x))),
        ftype,
        [cnodes[i] for i in faces2nodes(ctype)[iside]],
        quadrature,
    )
end

"""
    integrate_ref(g_ref, iside::Int, ctype::AbstractEntityType, cnodes, quadrature::AbstractQuadrature)

Integrate function `g_ref` on the `iside`-th side of the cell defined by its nodes `cnodes` and its
type `ctype`. Function `g_ref(x)` is expressed in the cell-reference element (not the face reference).

This function is helpfull to integrate shape functions (for instance ``\\int \\lambda_i \\lambda_j``) when the inverse
    mapping is not known explicitely (hence only ``\\hat{lambda}`` are known, not ``\\lambda``).
"""
function integrate_ref(
    g_ref,
    iside::Int,
    ctype::AbstractEntityType,
    cnodes,
    quadrature::AbstractQuadrature,
)
    # Get cell shape
    cshape = shape(ctype)

    # Get face type
    ftype = facetypes(ctype)[iside]

    # Get face parametrisation
    fp = mapping_face(cshape, iside) # mapping face-ref -> cell-ref

    # `g` is expressed in the cell-reference element. So we need to send x from the face-ref-element to
    # the cell-ref-element
    return integrate_ref(
        x -> g_ref(fp(x)),
        ftype,
        [cnodes[i] for i in faces2nodes(ctype)[iside]],
        quadrature,
    )
end

"""
    integrate_n(g, iside::Int, ctype::AbstractEntityType, cnodes, quadrature::AbstractQuadrature)

Perform an integration over the `iside`th face of an element (defined by `cnodes` and `ctype`).

Here `g` is expressed in the cell-local element : `n` is the normal vector in the local element,
and `x` is in the local element as well.

# Dev notes:
This method is DEPRECATED : never used, except in the unit tests...
"""
function integrate_n(
    g,
    iside::Int,
    ctype::AbstractEntityType,
    cnodes,
    quadrature::AbstractQuadrature,
)
    # Get cell shape
    cshape = shape(ctype)

    # Get face type and nodes
    ftype = facetypes(ctype)[iside]
    fnodes = [cnodes[i] for i in faces2nodes(ctype)[iside]]

    # Get face parametrisation
    fp = mapping_face(cshape, iside) # mapping face-ref -> cell-ref

    # `g` is express in the cell-local-element. The function `normal` takes input from the face-ref-element
    # so it doesn't need to be mapped. However for the second argument of `g` we need to send ξ from
    # the face-reference-element to the cell-reference-element to the cell-local-element.
    return integrate_ref(
        ξ -> g(normal(ctype, cnodes, iside, ξ), mapping(ctype, cnodes, fp(ξ))),
        ftype,
        fnodes,
        quadrature,
    )
end

"""
    integrate_n_ref(g_ref, iside::Int, ctype::AbstractEntityType, cnodes, quadrature::AbstractQuadrature)

Perform an integration over the `iside`th face of an element (defined by `cnodes` and `ctype`).

Here `g_ref` is expressed in the cell-reference element but `n` is the normal vector in the local element.
"""
function integrate_n_ref(
    g_ref,
    iside::Int,
    ctype::AbstractEntityType,
    cnodes,
    quadrature::AbstractQuadrature,
)
    # Get cell shape
    cshape = shape(ctype)

    # Get face type and nodes
    ftype = facetypes(ctype)[iside]
    fnodes = [cnodes[i] for i in faces2nodes(ctype)[iside]] # @ghislainb : need better solution to index with tuple

    # Get face parametrisation
    fp = mapping_face(cshape, iside) # mapping face-ref -> cell-ref

    # `g` is express in the cell-ref-element. The function `normal` takes input from the face-ref-element
    # so it doesn't need to be mapped. However for the second argument of `gref` we need to send x from
    # the face-reference-element to the cell-reference-element
    return integrate_ref(
        ξ -> g_ref(normal(ctype, cnodes, iside, ξ), fp(ξ)),
        ftype,
        fnodes,
        quadrature,
    )
end

"""
    integrate_ref(::isVolumic, g_ref, ctype::AbstractEntityType, cnodes, quadrature::AbstractQuadrature, ::T) where{N, T<:AbstractComputeQuadratureStyle}

Integrate function `g_ref` (expressed in reference element) on mesh element of type `ctype` defined by
its `cnodes` at the `quadrature`.
Computation is optimized according to the given concrete type `T<:AbstractComputeQuadratureStyle`.

To do so, a variable substitution is performed to integrate on the reference element.

# Implementation
It has been checked that calling the `apply_quadrature` method within this
function instead of directly applying the quadrature rule (i.e without the anonymous function) does not
decrease performance nor allocation.
"""
function integrate_ref(
    ::isVolumic,
    g_ref,
    ctype::AbstractEntityType,
    cnodes,
    quadrature::AbstractQuadrature,
    mapstyle::MapComputeQuadratureStyle,
)
    # the metric should be apply to `g_ref` before the integration:
    # compute the determinant of the jacobian
    # of the mapping only once,
    # and multiply all components of `gref(x)`
    f = x -> begin
        m = mapping_det_jacobian(ctype, cnodes, get_coords(x))
        map(gx -> m * gx, g_ref(x))
    end
    int = apply_quadrature(f, shape(ctype), quadrature, mapstyle)
    return int
end

"""
    getcache_∫(ctype::AbstractEntityType, cnodes, quadrature::AbstractQuadrature)

Return the data cache for function `∫`
"""
function getcache_∫(ctype::AbstractEntityType, cnodes, quadrature::AbstractQuadrature)
    qrule = QuadratureRule(shape(ctype), quadrature)

    qnodes = get_nodes(qrule)
    qweight = get_weights(qrule)

    T = typeof(mapping_det_jacobian(ctype, cnodes, qnodes[1]))
    qmap = SVector{length(qrule), T}(mapping_det_jacobian(ctype, cnodes, ξ) for ξ in qnodes)

    return qmap, qweight, qnodes
end

# (Remark: call overloading cannot be documented with Documenter,
#  see https://github.com/JuliaDocs/Documenter.jl/issues/228)
#
#     (t::NTuple{N, Function})(x)
#
# Make a tuple of functions callable
#
@generated (tup::NTuple{N, Function})(x) where {N} = :($((:(tup[$i](x)) for i in 1:N)...),)

# (Remark: call overloading cannot be documented with Documenter,
#  see https://github.com/JuliaDocs/Documenter.jl/issues/228)
#
#     (t::NTuple{N, NTuple{N1,Function}})(x)
#
# Make a tuple of tuples of functions callable
#
@generated function (tup::NTuple{N, NTuple{N1, Function}})(x) where {N, N1}
    :($((:(tup[$i](x)) for i in 1:N)...),)
end

function ∫v(g, cache)
    detJ, wq, xq = cache
    int = detJ[1] * wq[1] .* g(xq[1])
    n = length(wq)
    if n > 1
        for i in 2:n
            int = int .+ detJ[i] * wq[i] .* g(xq[i])
        end
    end
    return int
end

# Temporary : either we need to dispatch it, or we must force the user to choose between ∫v and ∫s
integrate(g, cache) = ∫v(g, cache)

"""
    integrate_ref(g_ref, ctype, cnodes, quadrature::AbstractQuadrature)

Integration on a node in a ``\\mathbb{R}^n`` space. This trivial function is only to simplify the 'side integral' expression.

# Implementation
For consistency reasons, `g_ref` is a function but it doesnt actually use its argument : the "reference-element" of a Node
can be anything. For instance consider integrating `g(x) = x` on a node named `node`. Then `g_ref(ξ) = g ∘ node.x`. As you can
see, `g_ref` doesnt actually depend on `ξ`
"""
function integrate_ref(
    ::isNodal,
    g_ref,
    ctype,
    cnodes,
    quadrature::AbstractQuadrature,
    ::AbstractComputeQuadratureStyle,
)
    return g_ref(nothing)
end

"""
    integrate_ref(::isCurvilinear, g_ref, ctype::AbstractEntityType{1}, cnodes, quadrature::AbstractQuadrature, ::T) where {T<:AbstractComputeQuadratureStyle}

Perform an integration of the function `g_ref` (expressed in local element) over a line in a ``\\matbb{R}^n`` space.

The applied formulae is: ``\\int_\\Gamma g(x) dx = \\int_l ||F'(l)|| g_ref(l) dl``
where ``F ~:~ \\mathbb{R} \\rightarrow \\mathbb{R}^n`` is the reference segment [-1,1] to the R^n line mapping.

Computation is optimized according to the given concrete type `T<:AbstractComputeQuadratureStyle`.
"""
function integrate_ref(
    ::isCurvilinear,
    g_ref,
    ctype::AbstractEntityType{1},
    cnodes,
    quadrature::AbstractQuadrature,
    mapstyle::MapComputeQuadratureStyle,
)
    # the metric should be apply to `g_ref` before the integration:
    # compute the determinant of the jacobian
    # of the mapping only once,
    # and multiply all components of `gref(x)`
    f = Base.Fix1(_apply_curvilinear_metric1, (g_ref, cnodes, ctype))
    return apply_quadrature2(f, shape(ctype), quadrature, mapstyle)
end
function _apply_curvilinear_metric1((g_ref, cnodes, ctype), x)
    m = norm(mapping_jacobian(ctype, cnodes, x))
    map(gx -> m * gx, g_ref(x)) # `map` is needed by legacy
    #m*g_ref(x) ok for new api
end

function integrate_on_ref(
    ::isCurvilinear,
    g,
    cellinfo::CellInfo{<:AbstractEntityType{1}},
    quadrature::AbstractQuadrature,
    mapstyle::MapComputeQuadratureStyle,
)
    # the metric should be apply to `g` before the integration:
    # compute the determinant of the jacobian
    # of the mapping only once,
    # and multiply all components of `g(x)`
    f = Base.Fix1(_apply_curvilinear_metric, (g, cellinfo))
    return apply_quadrature2(f, shape(celltype(cellinfo)), quadrature, mapstyle)
end

function _apply_curvilinear_metric((g, cellinfo), x)
    m = norm(mapping_jacobian(celltype(cellinfo), nodes(cellinfo), get_coords(x)))
    cellpoint = CellPoint(x, cellinfo, ReferenceDomain())
    m * g(cellpoint)
end

function integrate_on_ref(
    ::isVolumic,
    g::G,
    cellinfo::CellInfo,
    quadrature::AbstractQuadrature,
    mapstyle::MapComputeQuadratureStyle,
) where {G}
    # the metric should be apply to `g` before the integration:
    # compute the determinant of the jacobian
    # of the mapping only once, and multiply `g(x)`
    f = Base.Fix1(_apply_volume_metric, (g, cellinfo))
    int = apply_quadrature(f, shape(celltype(cellinfo)), quadrature, mapstyle)
    return int
end

function _apply_volume_metric(g_and_c::T, x::T1) where {T, T1}
    g, cellinfo = g_and_c
    m = mapping_det_jacobian(celltype(cellinfo), nodes(cellinfo), get_coords(x))
    cellpoint = CellPoint(x, cellinfo, ReferenceDomain())
    m * g(cellpoint)
end

function integrate_ref(
    ::isSurfacic,
    g_ref,
    ctype::AbstractEntityType,
    cnodes,
    quadrature::AbstractQuadrature,
    mapstyle::MapComputeQuadratureStyle,
)
    I = function (ξ)
        J = mapping_jacobian(ctype, cnodes, ξ)
        return norm(J[:, 1] × J[:, 2])
    end
    return apply_quadrature(ξ -> I(ξ), g_ref, shape(ctype), quadrature)
end

"""
Integration on a surface in a volume. We consider that we integrate on the negative side of the face.

WARNING : I need this now, but I am not satisfied. We need to rethink the whole integration API
"""
function integrate_face_ref(g_ref, finfo::FaceInfo, quadrature::AbstractQuadrature)
    _g_ref = ξ -> g_ref(FacePoint(ξ, finfo, ReferenceDomain()))
    return integrate_ref(
        _g_ref,
        facetype(finfo),
        nodes(finfo),
        quadrature,
        ComputeQuadratureStyle(_g_ref),
    )
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
