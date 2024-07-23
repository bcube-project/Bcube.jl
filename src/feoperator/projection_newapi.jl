"""
    projection_l2!(u::AbstractSingleFieldFEFunction, f, mesh::AbstractMesh, [degree::Int])
    projection_l2!(
        u::AbstractSingleFieldFEFunction,
        f,
        mesh::AbstractMesh,
        degree::Int;
        kwargs...,
    )
    projection_l2!(u::AbstractSingleFieldFEFunction, f, dΩ::Measure; mass = nothing)
    projection_l2!(u::MultiFieldFEFunction, f, args...; mass = nothing)

Compute dof values of `u` from a projection L2 so that:
```math
    ∫(u ⋅ v)dΩ = ∫(f ⋅ v)dΩ
```
where `dΩ` is the measure defined on the `CellDomain` of the `mesh` with a degree `D`

If `degree` is not given, the default value is set to `2d+1` where `d`
is the degree of the `function_space` of `u`.

Keyword argument `mass` could be used to give a precomputed matrix for
the left-hand term ``∫(u ⋅ v)dΩ``.
"""
function projection_l2!(u::AbstractSingleFieldFEFunction, f, mesh::AbstractMesh; kwargs...)
    degree = 2 * get_degree(get_function_space(get_fespace(u))) + 1
    projection_l2!(u, f, mesh, degree; kwargs...)
end

function projection_l2!(
    u::AbstractSingleFieldFEFunction,
    f,
    mesh::AbstractMesh,
    degree::Int;
    kwargs...,
)
    dΩ = Measure(CellDomain(mesh), degree)
    projection_l2!(u, f, dΩ; kwargs...)
end

function projection_l2!(u::AbstractSingleFieldFEFunction, f, dΩ::Measure; mass = nothing)
    @assert f isa AbstractLazy "`f` must be <:AbstractLazy : a `PhysicalFunction` for instance"

    U = get_fespace(u)
    V = TestFESpace(U)
    if isa(mass, Nothing)
        a(u, v) = ∫(u ⋅ v)dΩ
        A = assemble_bilinear(a, U, V)
    else
        A = mass
    end
    l(v) = ∫(f ⋅ v)dΩ
    T, = get_return_type_and_codim(f, get_domain(dΩ))
    b = assemble_linear(l, V; T = T)
    x = A \ b
    set_dof_values!(u, x)
    return nothing
end

function projection_l2!(u::MultiFieldFEFunction, f, args...; mass = nothing)
    _mass = isa(mass, Nothing) ? ntuple(i -> nothing, length((u...,))) : mass
    foreach(u, f, _mass) do uᵢ, fᵢ, mᵢ
        projection_l2!(uᵢ, fᵢ, args...; mass = mᵢ)
    end
end

struct CellMeanCache{FE, M, MM}
    feSpace::FE
    measure::M
    massMatrix::MM
end
get_fespace(cache::CellMeanCache) = cache.feSpace
get_measure(cache::CellMeanCache) = cache.measure
get_mass_matrix(cache::CellMeanCache) = cache.massMatrix

function build_cell_mean_cache(u::AbstractSingleFieldFEFunction, dΩ::Measure)
    mesh = get_mesh(get_domain(dΩ))
    fs = FunctionSpace(:Lagrange, 0)
    ncomp = get_ncomponents(get_fespace(u))
    Umean = TrialFESpace(fs, mesh, :discontinuous; size = ncomp)
    Vmean = TestFESpace(Umean)
    mass = factorize(build_mass_matrix(Umean, Vmean, dΩ))
    return CellMeanCache(Umean, dΩ, mass)
end
function build_cell_mean_cache(u::MultiFieldFEFunction, dΩ::Measure)
    return map(Base.Fix2(build_cell_mean_cache, dΩ), get_fe_functions(u))
end

"""
    cell_mean(q::MultiFieldFEFunction, dω::Measure)
    cell_mean(q::SingleFieldFEFunction, dω::Measure)

Return a vector containing mean cell values of `q`
computed according quadrature rules defined by measure `dω`.
# Dev note
* This function should be moved to a better file **(TODO)**
"""
function cell_mean(u, dΩ::Measure)
    cache = build_cell_mean_cache(u, dΩ)
    return cell_mean(u, cache)
end
function cell_mean(u::MultiFieldFEFunction, cache::Tuple{Vararg{CellMeanCache}})
    return map(get_fe_functions(u), cache) do uᵢ, cacheᵢ
        cell_mean(uᵢ, cacheᵢ)
    end
end
function cell_mean(u::AbstractFEFunction, cache::CellMeanCache)
    Umean = get_fespace(cache)
    u_mean = FEFunction(Umean, get_dof_type(u))
    projection_l2!(u_mean, u, get_measure(cache); mass = get_mass_matrix(cache))
    values = _reshape_cell_mean(u_mean, Val(get_size(Umean)))
    return MeshCellData(values)
end
function _reshape_cell_mean(u::SingleFieldFEFunction, ncomp::Val{N}) where {N}
    ncell = Int(length(get_dof_values(u)) / N)
    map(1:ncell) do i
        _may_scalar(get_dof_values(u, i, ncomp))
    end
end
_may_scalar(a::SVector{1}) = a[1]
_may_scalar(a) = a

cell_mean(u::MeshData{CellData}, ::Measure) = u

function build_mass_matrix(u::AbstractFEFunction, dΩ::AbstractMeasure)
    U = get_fespace(u)
    V = TestFESpace(U)
    return build_mass_matrix(U, V, dΩ)
end

function build_mass_matrix(U, V, dΩ::AbstractMeasure)
    m(u, v) = ∫(u ⋅ v)dΩ
    return assemble_bilinear(m, U, V)
end
