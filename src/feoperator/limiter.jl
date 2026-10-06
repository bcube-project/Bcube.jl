# REF:
# https://www.brown.edu/research/projects/scientific-computing/sites/brown.edu.research.projects.scientific-computing/files/uploads/Maximum-principle-satisfying%20and%20positivity-preserving.pdf

"""
    linear_scaling_limiter(
        u::SingleFieldFEFunction,
        u_mean::MeshData{<:CellData},
        domain::AbstractCellDomain,
        targetMeasures::NTuple{N, AbstractMeasure};
        periodicBCs::Union{NTuple{Nbc, BoundaryFaceDomain{M, <:PeriodicBCType}}, Nothing} = nothing,
        bounds::Union{Tuple{<:Number, <:Number}, Nothing} = nothing,
        DMPrelax = zero(get_dof_type(u)),
        coefmax = one(get_dof_type(u)),
        checkvalues = false,
    )

Apply the linear scaling limiter (see "Maximum-principle-satisfying and positivity-preserving
high-order schemes for conservation laws: survey and new developments", Zhang & Shu) to a
scalar discontinuous `FEFunction`, in order to enforce a local discrete maximum principle
(DMP), i.e. to prevent the creation of spurious extrema — typically in the neighborhood of
discontinuities.

In each cell, the limited solution is obtained by scaling the fluctuation around the cell
mean with a scalar coefficient `lim_u ∈ [0, coefmax]`:

    u_limited = u_mean + lim_u * (u - u_mean)

The cell mean is thus preserved by construction, and `lim_u = 0` corresponds to a
first-order (piecewise constant) solution.

# Arguments
- `u`: the scalar `FEFunction` to limit; its `FESpace` must be discontinuous.
- `u_mean`: the cell mean values of `u`, as a `MeshCellData` (i.e. `MeshData{<:CellData}`).
  It is typically obtained with `cell_mean(u, dΩ)` where `dΩ` is a `Measure` on the cells of
  the mesh.
- `domain`: an `AbstractCellDomain` — typically the whole `CellDomain(mesh)` — used to
  retrieve the mesh and the neighboring cells, needed to compute the min/max of the
  neighboring cell mean values. Restricted `CellDomain`s (a subset of cells) are not
  supported yet: the neighbor search currently runs over all mesh faces.
- `targetMeasures`: the complete list of `Measure`s on which the min and max of `u` are
  evaluated (at the quadrature nodes of each measure). Nothing is automatic: both the cell
  measure (e.g. `dΩ = Measure(CellDomain(mesh), 2 * degree + 1)`) and any face measure
  (e.g. `dΓ = Measure(InteriorFaceDomain(mesh), 2 * degree + 1)`) must be provided
  explicitly, as in `(dΩ, dΓ)`. Note that a `Measure` defined on a periodic
  `BoundaryFaceDomain` is not supported yet (it currently raises an error).

# Keyword arguments
- `periodicBCs`: not yet supported: any value other than `nothing` currently raises an
  error.
- `bounds`: optional `(lower, upper)` strong physical bounds imposed on the solution. The
  bounds are clipped to the range allowed by the local cell mean: if a cell mean does not
  satisfy the bounds, the admissible window of that cell reduces to its mean and the
  limiter degenerates to a first-order (piecewise constant) solution in that cell.
- `DMPrelax`: non-negative relaxation parameter (possibly broadcastable to the number of
  components) that enlarges the admissible range of neighboring cell mean values by
  `- DMPrelax` / `+ DMPrelax` (default: `0`, i.e. strict DMP).
- `coefmax`: upper bound of the limiter coefficient (default: `1`, i.e. the fluctuation is
  never amplified).
- `checkvalues`: if `true`, check that the cell mean values are consistent with the
  min/max bounds and error otherwise (default: `false`).

# Returns
A tuple `(lim_u, u_lim)` where:
- `lim_u` is a `MeshCellData` containing one limiter coefficient in `[0, coefmax]` per cell;
- `u_lim` is the limited field `u_mean + lim_u * (u - u_mean)` given as a lazy expression
  (`AbstractLazy`), and not as an `FEFunction`. It can be used directly inside a weak form
  (e.g. `∫(u_lim ⋅ v)dΩ`); to obtain the dofs of the limited solution, project it for
  instance with `projection_l2!`.

# Example
```julia
mesh = rectangle_mesh(20, 4)
degree = 2
fes = TrialFESpace(FunctionSpace(:Lagrange, degree), mesh, :discontinuous)
u = FEFunction(fes, mesh, PhysicalFunction(x -> x[1])) # the DG field to limit

Ω = CellDomain(mesh)
dΩ = Measure(Ω, 2 * degree + 1)
dΓ = Measure(InteriorFaceDomain(mesh), 2 * degree + 1)

u_mean = cell_mean(u, dΩ)
lim_u, u_lim = linear_scaling_limiter(u, u_mean, Ω, (dΩ, dΓ); bounds = (0.0, 1.0))

u_limited = FEFunction(fes)
projection_l2!(u_limited, u_lim, mesh)
```
"""
function linear_scaling_limiter(
    u::SingleFieldFEFunction,
    u_mean::MeshData{<:CellData},
    domain::AbstractCellDomain,
    targetMeasures::NTuple{N, AbstractMeasure};
    periodicBCs::Union{NTuple{Nbc, BoundaryFaceDomain{M, <:PeriodicBCType}}, Nothing} = nothing,
    bounds::Union{Tuple{<:Number, <:Number}, Nothing} = nothing,
    DMPrelax = zero(get_dof_type(u)),
    coefmax = one(get_dof_type(u)),
    checkvalues = false,
) where {N, Nbc, M}
    @assert is_discontinuous(get_fespace(u)) "linear_scaling_limiter only supports discontinuous FEFunction"
    @assert all(DMPrelax .≥ 0) "DMPrelax must be non-negative"

    mean = get_values(u_mean)
    limiter = similar(mean)

    minval = copy(mean)
    maxval = copy(mean)

    foreach(targetMeasures) do measure
        _minmax_elements!(minval, maxval, u, get_domain(measure), get_quadrature(measure))
    end

    minval_mean = copy(mean)
    maxval_mean = copy(mean)
    _mean_minmax_cells!(minval_mean, maxval_mean, mean, domain)
    # deal with periodic BC for mean values
    if !isnothing(periodicBCs)
        error("PeriodicBC are not yet supported for limiters")
    end

    # relax DMP
    @. minval_mean = minval_mean - DMPrelax
    @. maxval_mean = maxval_mean + DMPrelax

    # Impose strong physical bounds, but clipped to the range allowed by the local cell means
    if !isnothing(bounds)
        @. minval_mean = max(minval_mean, min(mean, bounds[1]))
        @. maxval_mean = min(maxval_mean, max(mean, bounds[2]))
    end

    for i in eachindex(limiter)
        limiter[i] = _compute_scalar_limiter(
            mean[i],
            minval[i],
            maxval[i],
            minval_mean[i],
            maxval_mean[i],
            coefmax,
            checkvalues,
        )
    end

    lim_u = MeshCellData(limiter)
    return lim_u, u_mean + lim_u*(u-u_mean)
end

"""
    _mean_minmax_cells!(minval_mean, maxval_mean, mean, domain)

For each cell, compute the min and max of mean values (in the `mean` array)
of the neighbor cells.

So `minval_mean[i]` is the minimum of the mean values of cells surrounding cell `i`.
The `domain` (an `AbstractCellDomain`) is only used to retrieve the mesh.
"""
function _mean_minmax_cells!(minval_mean, maxval_mean, mean, domain)
    mesh = get_mesh(domain)
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
    error("PeriodicBCs are not yet supported by the limiter (TODO)")
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

function _minmax_elements!(minval, maxval, v, domain::AbstractCellDomain, quadrature)
    _minmax_cells!(minval, maxval, v, domain, (quadrature,))
end
function _minmax_elements!(minval, maxval, v, domain::AbstractFaceDomain, quadrature)
    _minmax_faces!(minval, maxval, v, domain, (quadrature,))
end
function _minmax_elements!(
    minval,
    maxval,
    v,
    domain::BoundaryFaceDomain{M, <:PeriodicBCType},
    quadrature,
) where {M}
    _minmax_faces_periodic!(minval, maxval, v, domain, (quadrature,))
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
    error("PeriodicBCs are not yet supported by the limiter (TODO)")
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
    _compute_scalar_limiter(v̅ᵢ, mᵢ, Mᵢ, m̅, M̅, coefmax, checkvalues = false)

Compute the limiter coefficient θᵢ of cell `i`:

    θᵢ = max(0, min((M̅-v̅ᵢ)/(Mᵢ-v̅ᵢ), (v̅ᵢ-m̅)/(v̅ᵢ-mᵢ), coefmax))

where:
- `v̅ᵢ` is the cell mean value,
- `mᵢ`/`Mᵢ` are the min/max values of `u` in the cell (at the quadrature nodes
  of the target measures),
- `m̅`/`M̅` are the min/max of the neighboring cell mean values (relaxed by
  `DMPrelax`, and possibly clipped by the strong `bounds`).

Each ratio is regularized by `eps(eltype(y))` in the denominator. If `checkvalues`
is `true`, check that `v̅ᵢ` lies within `[mᵢ, Mᵢ]` and `[m̅, M̅]` and error otherwise.
"""
function _compute_scalar_limiter(v̅ᵢ, mᵢ, Mᵢ, m̅, M̅, coefmax, checkvalues = false)
    _0 = zero(eltype(v̅ᵢ))

    if checkvalues
        if !((m̅ ≤ v̅ᵢ ≤ M̅) && (mᵢ ≤ v̅ᵢ ≤ Mᵢ))
            @show m̅ ≤ v̅ᵢ ≤ M̅
            @show (M̅ - v̅ᵢ) ≥ _0
            @show (Mᵢ - v̅ᵢ) ≥ _0
            @show (v̅ᵢ - m̅) ≥ _0
            @show (v̅ᵢ - mᵢ) ≥ _0
            @show m̅, M̅
            @show mᵢ, v̅ᵢ, Mᵢ
            error("Limiter values are out of range")
        end
    end

    return max(_0, min(_ratio(M̅ - v̅ᵢ, Mᵢ - v̅ᵢ), _ratio(v̅ᵢ - m̅, v̅ᵢ - mᵢ), coefmax))
end

_ratio(x, y) = (x / (y + eps(eltype(y))))
