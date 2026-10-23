"""
    projection_l2!(u::AbstractSingleFieldFEFunction, f, mesh::AbstractMesh)
    projection_l2!(u::AbstractSingleFieldFEFunction, f, mesh::AbstractMesh, degree::Int)
    projection_l2!(u::AbstractSingleFieldFEFunction, f, dΩ::Measure; mass = nothing)
    projection_l2!(u::MultiFieldFEFunction, f, args...; mass = nothing)

Set the dof values of `u` — in place — from the L2-projection of `f` onto the `FESpace` of
`u`, i.e. the dof values solve:
```math
    ∫(u ⋅ v)dΩ = ∫(f ⋅ v)dΩ,   ∀v
```
where `v` spans the test space associated with the `FESpace` of `u`. The integrals are
evaluated with the quadrature rule of the cell `Measure` `dΩ`; for a discontinuous
`FESpace` the mass matrix is block-diagonal and the projection is local to each cell.

When a `mesh` is provided instead of a measure, `dΩ` is built internally on
`CellDomain(mesh)` with the quadrature degree `degree` — `2d+1` by default, where `d` is
the degree of the `FunctionSpace` of `u`. This degree makes the integrals exact for
products of two degree-`d` polynomials — e.g. `f ⋅ v` with `f` polynomial of degree `d` —
which are of degree `2d`.

Compared to the nodal initialization `FEFunction(fes, mesh, f)` — which requires a nodal
`FunctionSpace` and simply evaluates `f` at the dof nodes — `projection_l2!` computes the
L2-optimal dof values and accepts any lazy expression on any `FESpace`.

# Arguments
- `u`: the `FEFunction` whose dof values are computed (modified in place). It can also be
  a `MultiFieldFEFunction`, in which case `f` must be a tuple of lazy expressions (one per
  field) and the projection is applied field by field.
- `f`: the field to project; it must be an `AbstractLazy` expression — e.g. a
  `PhysicalFunction`, an `FEFunction` or a combination of lazy operators — and not a plain
  Julia function.
- `mesh`: the mesh on which the measure `dΩ` is built (it covers the whole `CellDomain`).
- `degree`: the quadrature degree of the measure built from `mesh` (default `2d+1`).
- `dΩ`: the cell `Measure` used to evaluate both integrals.

# Keyword arguments
- `mass`: a precomputed matrix — or, better, a `Factorization`, e.g. obtained with
  `factorize` from `LinearAlgebra` — for the left-hand term ``∫(u ⋅ v)dΩ``. This is useful
  when the projection is repeated with the same `FESpace` and measure, e.g. at each time
  step of an explicit scheme. Note that if `u` is a `MultiFieldFEFunction`, then `mass`
  must be a `Tuple` of the mass matrices of each `FESpace` constituting the `MultiFESpace`
  of `u` (and not the mass matrix of the whole `MultiFESpace`).

# Returns
`nothing`: `u` is modified in place.

# Examples
```julia
mesh = rectangle_mesh(20, 4)
fes = TrialFESpace(FunctionSpace(:Lagrange, 2), mesh) # continuous P2
u = FEFunction(fes)

# the measure is built internally, with the quadrature degree 2d+1 = 5
projection_l2!(u, PhysicalFunction(x -> x[1]^2 + x[2]^2), mesh)

# or explicitly, e.g. to share the same measure between several projections
dΩ = Measure(CellDomain(mesh), 5)
projection_l2!(u, PhysicalFunction(x -> x[1]^2 + x[2]^2), dΩ)
```
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
        @assert mass isa Union{AbstractMatrix, Factorization} "`mass` must be a matrix. If you use `projection_l2!` on a `MultiFieldFEFunction`, make sure to provide a Tuple of mass matrices and not the whole system matrix."
        A = mass
    end
    l(v) = ∫(f ⋅ v)dΩ
    b = zero(get_dof_values(u))
    assemble_linear!(b, l, V)
    x = zero(b)
    __solve!(x, A, b, get_bcube_backend(dΩ))
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

"""
    build_cell_mean_cache(u::AbstractLazy, dΩ::Measure)
    build_cell_mean_cache(u::MultiFieldFEFunction, dΩ::Measure)

Build the cache required to compute the cell mean values of `u` on the cells covered by
the `Measure` `dΩ`: the piecewise-constant discontinuous `FESpace` in which the mean
values live, together with its factorized mass matrix.

This is useful when the mean values are computed repeatedly — e.g. at each time step of
an explicit scheme — as the mass matrix is then assembled and factorized only once:

```julia
cache = build_cell_mean_cache(u, dΩ)
u_mean = cell_mean(u, cache) # cheap: reuses the cache
```

For a `MultiFieldFEFunction`, a tuple of caches (one per field) is returned, to be passed
to the corresponding `cell_mean` method. Note that `cell_mean(u, dΩ)` simply builds the
cache and calls `cell_mean(u, cache)`.
"""
function build_cell_mean_cache(u::AbstractLazy, dΩ::Measure)
    mesh = get_mesh(get_domain(dΩ))
    fs = FunctionSpace(:Lagrange, 0)
    _, N = get_return_type_and_codim(u, mesh)
    @assert length(N) == 1 "ncomponents > 1 not supported yet"
    ncomp = first(N)
    Umean = TrialFESpace(fs, mesh, :discontinuous; size = ncomp)
    Vmean = TestFESpace(Umean)
    mass = factorize(build_mass_matrix(Umean, Vmean, dΩ))
    return CellMeanCache(Umean, dΩ, mass)
end
function build_cell_mean_cache(u::MultiFieldFEFunction, dΩ::Measure)
    return map(Base.Fix2(build_cell_mean_cache, dΩ), get_fe_functions(u))
end

"""
    cell_mean(u::AbstractLazy, dΩ::Measure)
    cell_mean(u::MultiFieldFEFunction, dΩ::Measure)
    cell_mean(u::AbstractLazy, cache::CellMeanCache)
    cell_mean(u::MultiFieldFEFunction, cache::Tuple{Vararg{CellMeanCache}})
    cell_mean(u::MeshData{CellData}, dΩ::Measure)

Compute the mean value of `u` in each cell covered by the cell `Measure` `dΩ`, i.e. the
L2-projection of `u` onto the piecewise-constant (degree 0, discontinuous) space:
```math
    ūᵢ = \\frac{1}{|Kᵢ|} ∫_{Kᵢ}(u) dΩ
```
The integrals are evaluated with the quadrature rule of `dΩ`, which must therefore be
accurate enough — as for `projection_l2!`.

When the mean values are computed repeatedly — e.g. at each time step of an explicit
scheme — the underlying mass matrix can be assembled and factorized only once with
[`build_cell_mean_cache`](@ref), the mean values being then obtained with
`cell_mean(u, cache)`.

# Arguments
- `u`: the field whose mean values are sought: a `SingleFieldFEFunction` or, more
  generally, any lazy expression (`AbstractLazy`, e.g. a `PhysicalFunction`); it can also
  be a `MultiFieldFEFunction` (the mean values are then computed field by field), or even
  a `MeshData{CellData}`, in which case it is returned unchanged.
- `dΩ`: a `Measure` built on a `CellDomain` (e.g. `Measure(CellDomain(mesh), 2d+1)`).
- `cache`: a cache previously built with [`build_cell_mean_cache`](@ref), allowing to
  reuse the assembled and factorized mass matrix instead of rebuilding it at each call.

# Returns
A `MeshCellData` (i.e. a `MeshData{<:CellData}`) containing one mean value per cell — a
scalar for a single-component field, an `SVector` for a multi-component field. For a
`MultiFieldFEFunction`, a tuple of `MeshCellData` (one per field) is returned.

# Examples
```julia
mesh = rectangle_mesh(20, 4)
degree = 2
fes = TrialFESpace(FunctionSpace(:Lagrange, degree), mesh, :discontinuous)
u = FEFunction(fes, mesh, PhysicalFunction(x -> x[1]))

dΩ = Measure(CellDomain(mesh), 2 * degree + 1)

u_mean = cell_mean(u, dΩ) # one mean value per cell

# repeated evaluations (e.g. at each time step): reuse a prebuilt cache
cache = build_cell_mean_cache(u, dΩ)
u_mean = cell_mean(u, cache)
```

# Devs notes
* This function should be moved to a more appropriate file **(TODO)**
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
function cell_mean(u::AbstractLazy, cache::CellMeanCache)
    Umean = get_fespace(cache)
    dΩ = get_measure(cache)
    T, _ = get_return_type_and_codim(u, get_domain(dΩ))
    u_mean = FEFunction(Umean, T)
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

function build_mass_matrix(U, V, dΩ::D) where {D <: AbstractMeasure}
    m(u, v) = ∫(u ⋅ v)dΩ
    return assemble_bilinear(m, U, V)
end

function build_mass_matrix(
    U::MultiFESpace,
    V::MultiFESpace,
    dΩ::D,
) where {D <: AbstractMeasure}
    m(u, v) = ∫(_dot((u...,), (v...,)))dΩ
    return assemble_bilinear(m, U, V)
end

_dot(u::Tuple{T1}, v::Tuple{T2}) where {T1, T2} = first(u) ⋅ first(v)
function _dot(u::Tuple{Vararg{Any, N}}, v::Tuple{Vararg{Any, N}}) where {N}
    first(u) ⋅ first(v) + _dot(Base.tail(u), Base.tail(v))
end
