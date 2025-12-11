# REF:
# https://www.brown.edu/research/projects/scientific-computing/sites/brown.edu.research.projects.scientific-computing/files/uploads/Maximum-principle-satisfying%20and%20positivity-preserving.pdf

function linear_scaling_limiter_coef(
    v::SingleFieldFEFunction,
    dω::Measure,
    bounds,
    DMPrelax,
    periodicBCs::Union{Nothing, NTuple{N, <:BoundaryFaceDomain{Me, BC}}},
    check = true;
) where {N, Me, BC <: PeriodicBCType}
    @assert is_discontinuous(get_fespace(v)) "LinearScalingLimiter only support discontinuous variables"

    mesh = get_mesh(get_domain(dω))

    mean = get_values(cell_mean(v, dω))
    limiter = similar(mean)

    minval = similar(mean)
    minval .= typemax(eltype(minval))
    maxval = similar(mean)
    maxval .= -minval
    _minmax_cells!(minval, maxval, v, dω)
    _minmax_faces!(minval, maxval, v, dω)
    if !isnothing(periodicBCs)
        for domain in periodicBCs
            _minmax_faces_periodic!(minval, maxval, v, degquad, domain)
        end
    end

    minval_mean = similar(mean)
    minval_mean .= typemax(eltype(minval))
    maxval_mean = similar(mean)
    maxval_mean .= -minval_mean
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

function _minmax_cells(v, mesh, quadrature)
    c2n = connectivities_indices(mesh, :c2n)
    cellTypes = cells(mesh)

    val = map(1:ncells(mesh)) do i
        # mᵢ, Mᵢ : min/max at cell quadrature points
        ctypeᵢ = cellTypes[i]
        cnodesᵢ = get_nodes(mesh, c2n[i])
        cᵢ = CellInfo(i, ctypeᵢ, cnodesᵢ)
        vᵢ = materialize(v, cᵢ)
        fᵢ(ξ) = vᵢ(CellPoint(ξ, cᵢ, ReferenceDomain()))
        quadrule = QuadratureRule(shape(ctypeᵢ), quadrature)
        mᵢ, Mᵢ = _minmax(fᵢ, quadrule)
        mᵢ, Mᵢ
    end
    return val
end

"""
    _minmax_cells!(minval, maxval, v, dω)

Compute the min and max values of `v` in each cell of `dω`
"""
function _minmax_cells!(minval, maxval, v, dω)
    domain = get_domain(dω)
    quadrature = get_quadrature(dω)

    foreach_element(domain) do cellInfo, _, _
        # mᵢ, Mᵢ : min/max at cell quadrature points
        vᵢ = materialize(v, cellInfo)
        fᵢ(ξ) = vᵢ(CellPoint(ξ, cellInfo, ReferenceDomain()))
        quadrule = QuadratureRule(shape(celltype(cellInfo)), quadrature)
        mᵢ, Mᵢ = _minmax(fᵢ, quadrule)
        icell = cellindex(cellInfo)
        minval[icell] = min(mᵢ, minval[icell])
        maxval[icell] = max(Mᵢ, maxval[icell])
    end
    return nothing
end

function _minmax_faces!(minval, maxval, v, dω::AbstractMeasure{<:AbstractCellDomain})
    dΓ = Measure(AllFaceDomain(get_mesh(get_domain(dω))), get_quadrature(dω))
    _minmax_faces!(minval, maxval, v, dΓ)
end

function _minmax_faces!(minval, maxval, v, dω::AbstractMeasure{<:AbstractFaceDomain})
    quadrature = get_quadrature(dω)

    foreach_element(get_domain(dω)) do faceInfo, _, _
        i = cellindex(get_cellinfo_n(faceInfo))
        j = cellindex(get_cellinfo_p(faceInfo))

        mᵢⱼ, Mᵢⱼ, mⱼᵢ, Mⱼᵢ = _minmax_on_face(
            side_n(v),
            quadrature,
            facetype(faceInfo),
            faceInfo,
            opposite_side(faceInfo),
        )

        minval[i] = min(mᵢⱼ, minval[i])
        maxval[i] = max(Mᵢⱼ, maxval[i])
        minval[j] = min(mⱼᵢ, minval[j])
        maxval[j] = max(Mⱼᵢ, maxval[j])
    end
    return nothing
end

function _minmax_faces_periodic!(minval, maxval, v, degquad, periodicBcDomain)
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
    face_map_ji(ξ) = FacePoint(ξ, finfo_ji, ReferenceDomain())

    v_ij = materialize(v, finfo_ij)
    m_ij, M_ij = _minmax(v_ij ∘ face_map_ij, quadrule)

    v_ji = materialize(v, finfo_ji)
    m_ji, M_ji = _minmax(v_ji ∘ face_map_ji, quadrule)

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
function _compute_scalar_limiter(v̅ᵢ, mᵢ, Mᵢ, m, M, checkmean = true)
    if checkmean
        ((Mᵢ - v̅ᵢ) < (-10eps() * max(Mᵢ, one(Mᵢ)))) &&
            error("Invalid max value :  Mᵢ=$Mᵢ, v̅ᵢ=$v̅ᵢ")
        ((v̅ᵢ - mᵢ) < (-10eps() * max(v̅ᵢ, one(v̅ᵢ)))) &&
            error("Invalid min value :  mᵢ=$mᵢ, v̅ᵢ=$v̅ᵢ")
    end
    (Mᵢ - v̅ᵢ) < eps() && return zero(v̅ᵢ)
    (v̅ᵢ - mᵢ) < eps() && return zero(v̅ᵢ)
    _v̅ᵢ = max(mᵢ, min(Mᵢ, v̅ᵢ))
    #abs(Mᵢ-v̅ᵢ) > 10*eps(typeof(M)) ? coef⁺ = abs((M-v̅ᵢ)/(Mᵢ-v̅ᵢ)) : coef⁺ = zero(M)
    #abs(v̅ᵢ-mᵢ) > 10*eps(typeof(M)) ? coef⁻ = abs((v̅ᵢ-m)/(v̅ᵢ-mᵢ)) : coef⁻ = zero(M)
    return min(_ratio(M - _v̅ᵢ, Mᵢ - _v̅ᵢ), _ratio(_v̅ᵢ - m, _v̅ᵢ - mᵢ), 1.0)
end

_ratio(x, y) = abs(x / (y + eps(y)))

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
    dω::Measure;
    bounds::Union{Tuple{<:Number, <:Number}, Nothing} = nothing,
    DMPrelax = 0.0,
    periodicBCs::Union{Nothing, NTuple{N, <:BoundaryFaceDomain{Me, BC}}} = nothing,
    mass = nothing,
    checkmean = true,
) where {N, Me, BC <: PeriodicBCType}
    lim_u, u̅ = linear_scaling_limiter_coef(u, dω, bounds, DMPrelax, periodicBCs, checkmean)
    u_lim = FEFunction(get_fespace(u), get_dof_type(u))
    projection_l2!(u_lim, u̅ + lim_u * (u - u̅), dω; mass = mass)
    lim_u, u_lim
end
