struct CacheLinearScalingLimiter{CM}
    cacheCellMean::CM
end
function CacheLinearScalingLimiter(u::SingleFieldFEFunction, dω::Measure)
    cacheCellMean = build_cell_mean_cache(u, dω)
    CacheLinearScalingLimiter{typeof(cacheCellMean)}(cacheCellMean)
end

# REF:
# https://www.brown.edu/research/projects/scientific-computing/sites/brown.edu.research.projects.scientific-computing/files/uploads/Maximum-principle-satisfying%20and%20positivity-preserving.pdf
function linear_scaling_limiter_coef(
    v::SingleFieldFEFunction,
    dω::Measure,
    cellQuadratures,
    faceQuadratures,
    bounds = nothing,
    DMPrelax = zero(eltype(get_dof_values(v))),
    periodicBCs = nothing,
    check = true;
    coefmax = one(eltype(get_dof_values(v))),
    cache = CacheLinearScalingLimiter(v, dω),
)
    @assert is_discontinuous(get_fespace(v)) "LinearScalingLimiter only support discontinuous variables"

    mesh = get_mesh(get_domain(dω))

    mean = get_values(cell_mean(v, cache.cacheCellMean))
    limiter = similar(mean)

    minval = copy(mean)
    maxval = copy(mean)
    _minmax_cells!(minval, maxval, v, get_domain(dω), cellQuadratures)
    _minmax_faces!(minval, maxval, v, get_domain(dω), faceQuadratures)
    if !isnothing(periodicBCs)
        for domain in periodicBCs
            _minmax_faces_periodic!(minval, maxval, v, domain, faceQuadratures)
        end
    end

    minval_mean = copy(mean)
    maxval_mean = copy(mean)
    _mean_minmax_cells!(minval_mean, maxval_mean, mean, mesh)
    if !isnothing(periodicBCs)
        for domain in periodicBCs
            _mean_minmax_cells_periodic!(minval_mean, maxval_mean, mean, domain)
        end
    end

    # relax DMP
    @. minval_mean = minval_mean - DMPrelax
    @. maxval_mean = maxval_mean + DMPrelax

    # impose strong physical bounds
    if !isnothing(bounds)
        @. minval_mean = max(minval_mean, bounds[1])
        @. maxval_mean = min(maxval_mean, bounds[2])
    end

    for i in 1:ncells(mesh)
        limiter[i] = _compute_scalar_limiter(
            mean[i],
            minval[i],
            maxval[i],
            minval_mean[i],
            maxval_mean[i],
            coefmax,
            check,
        )
    end

    MeshCellData(limiter), MeshCellData(mean)
end

"""
    _mean_minmax_cells!(minval_mean, maxval_mean, mean, mesh)

For each cell, compute the min and max of mean values (in the `mean` array)
of the neighbor cells.

So `minval_mean[i]` is the minimum of the mean values of cells surrounding cell `i`.
"""
function _mean_minmax_cells!(minval_mean, maxval_mean, mean, mesh)
    f2c = connectivities_indices(mesh, :f2c)

    minval_mean .= mean
    maxval_mean .= mean

    for kface in 1:nfaces(mesh)
        _f2c = f2c[kface]

        if length(_f2c) > 1
            i = _f2c[1]
            j = _f2c[2]

            minval_mean[i] = min(mean[j], minval_mean[i])
            maxval_mean[i] = max(mean[j], maxval_mean[i])

            minval_mean[j] = min(mean[i], minval_mean[j])
            maxval_mean[j] = max(mean[i], maxval_mean[j])
        end
    end
    return nothing
end

function _mean_minmax_cells_periodic!(minval_mean, maxval_mean, mean, periodicBcDomain)
    error("TODO")
    # # TODO : add a specific API for the domain cache:
    # perio_cache = get_cache(periodicBcDomain)
    # _1, _2, _3, bnd_f2c, _5, _6 = perio_cache

    # for kface in axes(bnd_f2c,1)

    #     i = bnd_f2c[kface, 1]
    #     j = bnd_f2c[kface, 2]

    #     minval_mean[i] = min(mean[j], minval_mean[i])
    #     maxval_mean[i] = max(mean[j], maxval_mean[i])

    #     minval_mean[j] = min(mean[i], minval_mean[j])
    #     maxval_mean[j] = max(mean[i], maxval_mean[j])
    # end
    return nothing
end

"""
    _minmax_cells!(minval, maxval, v, domain, quadratures)

Compute the min and max values of `v` interpolated at
`quadratures` points in each cell of `domain`
"""
function _minmax_cells!(minval, maxval, v, domain, quadratures)
    foreach_element(domain) do cellInfo, _, _
        # mᵢ, Mᵢ : min/max at cell quadrature points
        vᵢ = materialize(v, cellInfo)
        fᵢ(ξ) = vᵢ(CellPoint(ξ, cellInfo, ReferenceDomain()))
        icell = cellindex(cellInfo)
        for quadrature in quadratures
            quadrule = QuadratureRule(shape(celltype(cellInfo)), quadrature)
            mᵢ, Mᵢ = _minmax(fᵢ, quadrule)
            minval[icell] = min(mᵢ, minval[icell])
            maxval[icell] = max(Mᵢ, maxval[icell])
        end
    end
    return nothing
end

function _minmax_faces!(minval, maxval, v, Ω::AbstractCellDomain, faceQuadratures)
    Γ = InteriorFaceDomain(get_mesh(Ω))
    _minmax_faces!(minval, maxval, v, Γ, faceQuadratures)
    Γb = BoundaryFaceDomain(get_mesh(Ω))
    _minmax_faces!(minval, maxval, v, Γb, faceQuadratures)
end

function _minmax_faces!(minval, maxval, v, faceDomain::AbstractFaceDomain, quadratures)
    foreach_element(faceDomain) do faceInfo, _, _
        i = cellindex(get_cellinfo_n(faceInfo))
        if has_opposite_side(faceInfo)
            oppositeFaceInfo = opposite_side(faceInfo)
            j = cellindex(get_cellinfo_p(faceInfo))
        else
            oppositeFaceInfo = nothing
            j = -1
        end

        for quadrature in quadratures
            mᵢⱼ, Mᵢⱼ, mⱼᵢ, Mⱼᵢ = _minmax_on_face(
                side_n(v),
                quadrature,
                facetype(faceInfo),
                faceInfo,
                oppositeFaceInfo,
            )
            minval[i] = min(mᵢⱼ, minval[i])
            maxval[i] = max(Mᵢⱼ, maxval[i])
            if has_opposite_side(faceInfo)
                minval[j] = min(mⱼᵢ, minval[j])
                maxval[j] = max(Mⱼᵢ, maxval[j])
            end
        end
    end
    return nothing
end

function _minmax_faces_periodic!(minval, maxval, v, periodicBcDomain, quadratures)
    error("TODO")
    # mesh = get_mesh(v)
    # c2n = connectivities_indices(mesh,:c2n)
    # f2n = connectivities_indices(mesh,:f2n)
    # f2c = connectivities_indices(mesh,:f2c)

    # # TODO : add a specific API for the domain cache:
    # perio_cache = get_cache(periodicBcDomain)
    # A = transformation(get_bc(periodicBcDomain))
    # bndf2f, bnd_f2n1, bnd_f2n2, bnd_f2c, bnd_ftypes, bnd_n2n = perio_cache

    # cellTypes = cells(mesh)
    # faceTypes = faces(mesh)

    # for kface in axes(bnd_f2c,1)

    #     ftype = faceTypes[kface]
    #     _f2c = f2c[kface]

    #     # Neighbor cell i
    #     i = bnd_f2c[kface, 1]
    #     cnodesᵢ = get_nodes(mesh, c2n[i])
    #     ctypeᵢ = cellTypes[i]

    #     # Neighbor cell j
    #     j = bnd_f2c[kface, 2]
    #     cnodesⱼ = get_nodes(mesh, c2n[j])
    #     cnodesⱼ = map(n->Node(A(get_coords(n))), cnodesⱼ)
    #     ctypeⱼ = cellTypes[j]

    #     mᵢⱼ, Mᵢⱼ, mⱼᵢ, Mⱼᵢ = _minmax_on_face_periodic(v, degquad, i, j, kface, ftype, ctypeᵢ, ctypeⱼ, bnd_f2n1, bnd_f2n2, c2n, cnodesᵢ, cnodesⱼ, bnd_n2n)

    #     minval[i] = min(mᵢⱼ, minval[i])
    #     maxval[i] = max(Mᵢⱼ, maxval[i])
    #     minval[j] = min(mⱼᵢ, minval[j])
    #     maxval[j] = max(Mⱼᵢ, maxval[j])
    # end
    return nothing
end

function _minmax_on_face(v, quadrature, ftype, finfo_ij, finfo_ji)
    quadrule = QuadratureRule(shape(ftype), quadrature)

    face_map_ij(ξ) = FacePoint(ξ, finfo_ij, ReferenceDomain())
    v_ij = materialize(v, finfo_ij)
    m_ij, M_ij = _minmax(v_ij ∘ face_map_ij, quadrule)

    if !isa(finfo_ji, Nothing)
        face_map_ji(ξ) = FacePoint(ξ, finfo_ji, ReferenceDomain())
        v_ji = materialize(v, finfo_ji)
        m_ji, M_ji = _minmax(v_ji ∘ face_map_ji, quadrule)
    else
        m_ji = M_ji = nothing
    end

    return m_ij, M_ij, m_ji, M_ji
end

function _minmax_on_face_periodic(
    v,
    quadrature,
    i,
    j,
    faceᵢⱼ,
    ftypeᵢⱼ,
    ctypeᵢ,
    ctypeⱼ,
    bnd_f2n1,
    bnd_f2n2,
    c2n,
    cnodesᵢ,
    cnodesⱼ,
    bnd_n2n,
)
    c2nᵢ = c2n[i, Val(nnodes(ctypeᵢ))]
    c2nⱼ = c2n[j, Val(nnodes(ctypeⱼ))]
    c2nⱼ_perio = map(k -> get(bnd_n2n, k, k), c2nⱼ)

    nnodes_f = Val(nnodes(ftype))
    sideᵢ = cell_side(ctypeᵢ, c2nᵢ, bnd_f2n1[faceᵢⱼ, nnodes_f])
    csᵢ = CellSide(i, sideᵢ, ctypeᵢ, cnodesᵢ, c2nᵢ)
    sideⱼ = cell_side(ctypeⱼ, c2nⱼ, bnd_f2n2[faceᵢⱼ, nnodes_f])
    csⱼ = CellSide(j, sideⱼ, ctypeⱼ, cnodesⱼ, c2nⱼ_perio)

    fp = FaceParametrization()
    quadrule = QuadratureRule(shape(ftypeᵢⱼ), quadrature)

    vᵢⱼ = (v ∘ fp)[Side(Side⁻(), (csᵢ, csⱼ))]
    mᵢⱼ, Mᵢⱼ = _minmax(vᵢⱼ, quadrule)

    vⱼᵢ = (v ∘ fp)[Side(Side⁻(), (csⱼ, csᵢ))]
    mⱼᵢ, Mⱼᵢ = _minmax(vⱼᵢ, quadrule)

    return mᵢⱼ, Mᵢⱼ, mⱼᵢ, Mⱼᵢ
end

# here we assume that f is define in ref. space
_minmax(f, quadrule::AbstractQuadratureRule) = extrema(f(ξ) for ξ in get_nodes(quadrule))

"""
    _compute_scalar_limiter(v̅ᵢ, mᵢ, Mᵢ, m, M, checkmean = true)

v̅ᵢ = mean
mᵢ = minval
Mᵢ = maxval
m = minval_mean
M = maxval_mean
"""
function _compute_scalar_limiter(v̅ᵢ, mᵢ, Mᵢ, m, M, coefmax, checkmean = true)
    _0 = zero(v̅ᵢ)
    eps0 = 2 * eps(eltype(v̅ᵢ))
    if checkmean
        ((Mᵢ - v̅ᵢ) < (-10eps() * max(Mᵢ, one(Mᵢ)))) &&
            error("Invalid max value :  Mᵢ=$Mᵢ, v̅ᵢ=$v̅ᵢ")
        ((v̅ᵢ - mᵢ) < (-10eps() * max(v̅ᵢ, one(v̅ᵢ)))) &&
            error("Invalid min value :  mᵢ=$mᵢ, v̅ᵢ=$v̅ᵢ")
        if !(m ≤ v̅ᵢ ≤ M) ||
           !((M - v̅ᵢ) ≥ _0) ||
           !((Mᵢ - v̅ᵢ) ≥ -eps0) ||
           !((v̅ᵢ - m) ≥ _0) ||
           !((v̅ᵢ - mᵢ) ≥ -eps0)
            @show m ≤ v̅ᵢ ≤ M
            @show (M - v̅ᵢ) ≥ _0
            @show (Mᵢ - v̅ᵢ) ≥ -eps0
            @show (v̅ᵢ - m) ≥ _0
            @show (v̅ᵢ - mᵢ) ≥ -eps0
            @show m, M
            @show mᵢ, v̅ᵢ, Mᵢ
            error("Limiter values are out of range")
        end
    end
    #(Mᵢ - v̅ᵢ) < eps(v̅ᵢ) && return zero(v̅ᵢ)
    #(v̅ᵢ - mᵢ) < eps(v̅ᵢ) && return zero(v̅ᵢ)
    _v̅ᵢ = v̅ᵢ# max(mᵢ, min(Mᵢ, v̅ᵢ))
    #abs(Mᵢ-v̅ᵢ) > 10*eps(typeof(M)) ? coef⁺ = abs((M-v̅ᵢ)/(Mᵢ-v̅ᵢ)) : coef⁺ = zero(M)
    #abs(v̅ᵢ-mᵢ) > 10*eps(typeof(M)) ? coef⁻ = abs((v̅ᵢ-m)/(v̅ᵢ-mᵢ)) : coef⁻ = zero(M)
    return max(_0, min(_ratio(M - _v̅ᵢ, Mᵢ - _v̅ᵢ), _ratio(_v̅ᵢ - m, _v̅ᵢ - mᵢ), coefmax))
end

_ratio(x, y) = (x / (y + eps(eltype(y))))

"""
    linear_scaling_limiter(
        u::SingleFieldFEFunction,
        dω::Measure;
        bounds::Union{Tuple{<:Number, <:Number}, Nothing} = nothing,
        DMPrelax = 0.0,
        periodicBCs::Union{Nothing, NTuple{N, <:BoundaryFaceDomain{Me, BC}}} = nothing,
        mass = nothing,
        checkmean = true
    ) where {N, Me, BC <: PeriodicBCType}

Apply the linear scaling limiter (see "Maximum-principle-satisfying and positivity-preserving high order schemes for
conservation laws: Survey and new developments", Zhang & Shu).

`u_limited = u̅ + lim_u * (u - u̅)`

The first returned argument is the coefficient `lim_u`, and the second is `u_limited`.
"""
function linear_scaling_limiter(
    u::SingleFieldFEFunction,
    dω::Measure,
    cellQuadratures::NTuple{Nq, AbstractQuadrature} = (get_quadrature(dω),),
    faceQuadratures::NTuple{Nq, AbstractQuadrature} = (get_quadrature(dω),);
    bounds::Union{Tuple{<:Number, <:Number}, Nothing} = nothing,
    DMPrelax = 0.0,
    periodicBCs = nothing,
    mass = nothing,
    checkmean = true,
    coefmax = one(eltype(get_dof_values(u))),
    cache = CacheLinearScalingLimiter(u, dω),
) where {Nq}
    lim_u, u̅ = linear_scaling_limiter_coef(
        u,
        dω,
        cellQuadratures,
        faceQuadratures,
        bounds,
        DMPrelax,
        periodicBCs,
        checkmean;
        coefmax = coefmax,
        cache = cache,
    )
    u_lim = FEFunction(get_fespace(u), get_dof_type(u))
    projection_l2!(u_lim, u̅ + lim_u * (u - u̅), dω; mass = mass)
    lim_u, u_lim, u̅
end
