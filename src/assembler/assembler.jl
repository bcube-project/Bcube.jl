allocate_linear(backend::BcubeBackendCPUSerial, V, T) = allocate_dofs(V, T)

function allocate_bilinear(backend::BcubeBackendCPUSerial, a, U, V, T)
    # Prepare sparse matrix allocation
    I = Int[]
    J = Int[]
    X = T[]

    # Pre-allocate I, J, X
    n = _count_n_elts(U, V, a)
    foreach(Base.Fix2(sizehint!, n), (I, J, X))
    return I, J, X
end

"""
    assemble_bilinear(a::Function, U, V; T = Float64, backend::AbstractBcubeBackend = get_bcube_backend())

Assemble the (sparse) Matrix corresponding to the given bilinear form `a`
on the trial and test finite element spaces `U` and `V`.

For the in-place version, check-out [`assemble_bilinear!`](@ref).

# Arguments
- `a::Function` : function of two variables (u,v) representing the bilinear form
- `U` : trial finite element space (for `u`)
- `V` : test finite element space (for `v`)

# Examples
```jldoctest
julia> mesh = rectangle_mesh(3,3)
julia> U = TrialFESpace(FunctionSpace(:Lagrange, 0), mesh)
julia> V = TestFESpace(U)
julia> dΩ = Measure(CellDomain(mesh), 3)
julia> a(u, v) = ∫(u * v)dΩ
julia> assemble_bilinear(a, U, V)
4×4 SparseArrays.SparseMatrixCSC{Float64, Int64} with 4 stored entries:
 0.25   ⋅     ⋅     ⋅
  ⋅    0.25   ⋅     ⋅
  ⋅     ⋅    0.25   ⋅
  ⋅     ⋅     ⋅    0.25
```
"""
function assemble_bilinear(
    a::Function,
    U::Union{TrialFESpace, AbstractMultiFESpace{N, <:Tuple{Vararg{TrialFESpace, N}}}},
    V::Union{TestFESpace, AbstractMultiFESpace{N, <:Tuple{Vararg{TestFESpace, N}}}};
    T = Float64,
    backend::AbstractBcubeBackend = get_bcube_backend(),
) where {N}
    I, J, X = allocate_bilinear(backend, a, U, V, T)

    # Compute
    assemble_bilinear!(I, J, X, a, U, V, backend)
    nrows = get_ndofs(V)
    ncols = get_ndofs(U)
    return sparse(I, J, X, nrows, ncols)
end

function assemble_bilinear!(I, J, X, a, U, V, backend::AbstractBcubeBackend)
    return_type_a = a(_null_operator(U), _null_operator(V))
    _assemble_bilinear!(I, J, X, a, U, V, return_type_a, backend)
    return nothing
end

function _assemble_bilinear!(
    I::AbstractVector,
    J::AbstractVector,
    X::AbstractVector,
    a::Function,
    U,
    V,
    integration::Integration,
    backend::AbstractBcubeBackend,
)
    f(u, v) = get_function(get_integrand(a(u, v)))
    measure = get_measure(integration)
    assemble_bilinear!(I, J, X, f, measure, U, V, backend)
    return nothing
end

function _assemble_bilinear!(
    I::AbstractVector,
    J::AbstractVector,
    X::AbstractVector,
    a::Function,
    U,
    V,
    multiIntegration::MultiIntegration{N},
    backend::AbstractBcubeBackend,
) where {N}
    for i in 1:N
        ival = Val(i)
        aᵢ(u, v) = a(u, v)[ival]
        _assemble_bilinear!(I, J, X, aᵢ, U, V, multiIntegration[ival], backend)
    end
    nothing
end

"""
    assemble_bilinear!(I, J, X, f, measure::Measure, U::TrialFESpace, V::TestFESpace, backend::BcubeBackendCPUSerial)

In-place version of [`assemble_bilinear`](@ref).
"""
function assemble_bilinear!(
    I,
    J,
    X,
    f,
    measure::Measure,
    U::TrialFESpace,
    V::TestFESpace,
    backend::AbstractBcubeBackend,
)
    # Alias
    quadrature = get_quadrature(measure)
    domain = get_domain(measure)

    # Loop over cells
    foreach_element(domain, backend) do elementInfo
        λu, λv = blockmap_bilinear_shape_functions(U, V, elementInfo)
        g1 = materialize(f(λu, λv), elementInfo)
        values = integrate_on_ref_element(g1, elementInfo, quadrature)
        _append_contribution!(X, I, J, U, V, values, elementInfo, domain, backend)
    end

    return nothing
end

function assemble_bilinear!(
    I::AbstractVector,
    J::AbstractVector,
    X::AbstractVector,
    f::Function,
    measure::Measure,
    U::AbstractMultiFESpace{N, <:Tuple{Vararg{TrialFESpace, N}}},
    V::AbstractMultiFESpace{N, <:Tuple{Vararg{TestFESpace, N}}},
    backend::AbstractBcubeBackend,
) where {N}

    # Loop over all combinations
    for (j, _U) in enumerate(U)
        for (i, _V) in enumerate(V)
            # Materialize function for `(..., uj, ...), (...,vi,...)`
            tuple_u = _get_tuple_var(Val(N), j)
            tuple_v = _get_tuple_var(Val(N), i)
            _f(uj, vi) = f(tuple_u(uj), tuple_v(vi))

            # Skip computation if there is nothing to compute
            isa(_f(maywrap(_U), maywrap(_V)), NullOperator) && continue

            # Local indices / values
            _I = Int[]
            _J = Int[]
            n = _count_n_elts(U, V, get_domain(measure))
            sizehint!.((_I, _J), n)

            # Perform assembly on SingleFESpace
            assemble_bilinear!(_I, _J, X, _f, measure, _U, _V, backend)

            # Update global indices
            push!(I, get_mapping(V, i)[_I]...)
            push!(J, get_mapping(U, j)[_J]...)
        end
    end

    return nothing
end

"""
    assemble_linear(l::Function, V::Union{TestFESpace, AbstractMultiTestFESpace}; T = Float64, backend::AbstractBcubeBackend = get_bcube_backend())

Assemble the vector corresponding to a linear form `l` on the finite element space `V`

For the in-place version, checkout [`assemble_linear!`](@ref).

# Arguments
- `l::Function` : linear form to assemble, a function of one variable `l(v)`
- `V` : test finite element space

# Examples
```jldoctest
julia> mesh = rectangle_mesh(3,3)
julia> U = TrialFESpace(FunctionSpace(:Lagrange, 0), mesh)
julia> V = TestFESpace(U)
julia> dΩ = Measure(CellDomain(mesh), 3)
julia> l(v) = ∫(v)dΩ
julia> assemble_linear(l, V)
4-element Vector{Float64}:
 0.25
 0.25
 0.25
 0.25
```
"""
function assemble_linear(
    l::Function,
    V::Union{TestFESpace, AbstractMultiTestFESpace};
    T = Float64,
    backend::AbstractBcubeBackend = get_bcube_backend(),
)
    b = allocate_linear(backend, V, T)
    assemble_linear!(b, l, V; backend = backend)
    return b
end

"""
    assemble_linear!(b::AbstractVector, l::Function, V::Union{TestFESpace, AbstractMultiTestFESpace}; backend::AbstractBcubeBackend = get_bcube_backend())

In-place version of [`assemble_linear`](@ref).
"""
function assemble_linear!(
    b::AbstractVector,
    l::Function,
    V::Union{TestFESpace, AbstractMultiTestFESpace};
    backend::AbstractBcubeBackend = get_bcube_backend(),
)
    # apply `l` on `NullOperator` to get the type
    # of the result of `l` and use it for dispatch
    # (`Integration` or `MultiIntegration` case).
    _assemble_linear!(b, l, V, l(_null_operator(V)), backend)
    return nothing
end

"""
    _assemble_linear!(b, l, V, integration::Integration, backend::AbstractBcubeBackend)
    _assemble_linear!(b, l, V, integration::MultiIntegration{N}, backend::AbstractBcubeBackend) where {N}

These functions act as a function barrier in order to:
* get the function corresponding to the operand in the linear form
* reshape `b` internally to deal with cases when `V` is a `AbstractMultiTestFESpace`
* call `__assemble_linear!` to apply dispatch on the type of `measure` of the
  integration and improve type stability during the assembling loop.

## Dev note:
The case `integration::MultiIntegration{N}` is treated by looping over
each `Integration` contained in the `MultiIntegration`
"""
function _assemble_linear!(b, l, V, integration::Integration, backend::AbstractBcubeBackend)
    f(v) = get_function(get_integrand(l(v)))
    measure = get_measure(integration)
    __assemble_linear!(_may_reshape_b(b, V), f, V, measure, backend)
    return nothing
end

function _assemble_linear!(
    b,
    l,
    V,
    integration::MultiIntegration{N},
    backend::AbstractBcubeBackend,
) where {N}
    ival = Val(N)
    lᵢ(v) = l(v)[ival]
    _assemble_linear!(b, lᵢ, V, integration[ival], backend)
    if N > 1 # recursive calls
        _assemble_linear!(
            b,
            l,
            V,
            MultiIntegration(Base.front(integration.integrations)),
            backend,
        )
    end
    return nothing
end

"""
# Dev notes
Two levels of "LazyMapOver" because first we LazyMapOver the Tuple of argument of the linear form,
and the for each item of this Tuple we LazyMapOver the shape functions.
"""
function __assemble_linear!(b, f, V, measure::Measure, backend::AbstractBcubeBackend)
    quadrature = get_quadrature(measure)
    domain = get_domain(measure)
    foreach_element(domain, backend) do elementInfo
        vₑ = blockmap_shape_functions(V, elementInfo)
        fᵥ = materialize(f(vₑ), elementInfo)
        values = integrate_on_ref_element(fᵥ, elementInfo, quadrature)
        _update_b!(b, V, values, elementInfo, domain, backend)
    end
    nothing
end

_null_operator(::AbstractFESpace) = NullOperator()
_null_operator(::AbstractMultiFESpace{N}) where {N} = ntuple(i -> NullOperator(), Val(N))

"""
For `AbstractMultiTestFESpace`, it creates a Tuple (of views) of the different
"destination" in the vector: one for each FESpace
"""
_may_reshape_b(b::AbstractVector, V::TestFESpace) = b
function _may_reshape_b(b::AbstractVector, V::AbstractMultiTestFESpace)
    ntuple(i -> view(b, get_mapping(V, i)), Val(get_n_fespace(V)))
end

"""
bilinear case
"""
function _append_contribution!(
    X,
    I,
    J,
    U,
    V,
    values,
    elementInfo::CellInfo,
    domain,
    backend::AbstractBcubeBackend,
)
    icell = cellindex(elementInfo)
    nU = Val(get_ndofs(U, shape(celltype(elementInfo))))
    nV = Val(get_ndofs(V, shape(celltype(elementInfo))))
    Udofs = get_dofs(U, icell, nU) # columns correspond to the TrialFunction
    Vdofs = get_dofs(V, icell, nV) # lines correspond to the TestFunction
    unwrapValues = _unwrap_cell_integrate(V, values)
    _append_bilinear!(I, J, X, Vdofs, Udofs, unwrapValues, backend)
    return nothing
end

function _append_contribution!(
    X,
    I,
    J,
    U,
    V,
    values,
    elementInfo::FaceInfo,
    domain,
    backend::AbstractBcubeBackend,
)
    cellinfo_n = get_cellinfo_n(elementInfo)
    cellinfo_p = get_cellinfo_p(elementInfo)
    cellindex_n = cellindex(cellinfo_n)
    cellindex_p = cellindex(cellinfo_p)

    unwrapValues = _unwrap_face_integrate(U, V, values)

    nU_n = Val(get_ndofs(U, shape(celltype(cellinfo_n))))
    nV_n = Val(get_ndofs(V, shape(celltype(cellinfo_n))))
    nU_p = Val(get_ndofs(U, shape(celltype(cellinfo_p))))
    nV_p = Val(get_ndofs(V, shape(celltype(cellinfo_p))))

    col_dofs_U_n = get_dofs(U, cellindex_n, nU_n) # columns correspond to the TrialFunction on side⁻
    row_dofs_V_n = get_dofs(V, cellindex_n, nV_n) # lines correspond to the TestFunction on side⁻
    col_dofs_U_p = get_dofs(U, cellindex_p, nU_p) # columns correspond to the TrialFunction on side⁺
    row_dofs_V_p = get_dofs(V, cellindex_p, nV_p) # lines correspond to the TestFunction on side⁺

    for (k, (row, col)) in enumerate(
        Iterators.product((row_dofs_V_n, row_dofs_V_p), (col_dofs_U_n, col_dofs_U_p)),
    )
        _append_bilinear!(I, J, X, row, col, unwrapValues[k], backend)
    end
    return nothing
end

function _append_bilinear!(I, J, X, row, col, vals, backend::BcubeBackendCPUSerial)
    _rows, _cols = _cartesian_product(row, col)
    @assert length(_rows) == length(_cols) == sum(length, vals)
    append!(I, _rows)
    append!(J, _cols)
    append!(X, vals...)
end
function _append_bilinear!(
    I,
    J,
    X,
    row,
    col,
    vals::NullOperator,
    backend::AbstractBcubeBackend,
)
    nothing
end

#fix ambiguity:
function _append_bilinear!(
    I,
    J,
    X,
    row,
    col,
    vals::NullOperator,
    ::Bcube.BcubeBackendCPUSerial,
)
    nothing
end

Base.getindex(::Bcube.LazyOperators.NullOperator, i) = NullOperator()

function _unwrap_face_integrate(
    ::Union{TrialFESpace, AbstractMultiTrialFESpace},
    ::Union{TestFESpace, AbstractMultiTestFESpace},
    a,
)
    return _recursive_unwrap(a)
end

_recursive_unwrap(a::LazyOperators.AbstractMapOver) = map(_recursive_unwrap, unwrap(a))
_recursive_unwrap(a) = unwrap(a)

function _update_b!(
    b,
    V,
    values,
    elementInfo::CellInfo,
    domain,
    backend::AbstractBcubeBackend,
)
    idofs = get_dofs(V, cellindex(elementInfo))
    unwrapValues = _unwrap_cell_integrate(V, values)
    __update_b!(b, idofs, unwrapValues, backend)
end
_unwrap_cell_integrate(::TestFESpace, a) = map(unwrap, unwrap(a))
_unwrap_cell_integrate(::AbstractMultiTestFESpace, a) = map(unwrap, unwrap(a))

function _update_b!(
    b::B,
    V::TV,
    values::Tval,
    elementInfo::FaceInfo,
    domain,
    backend::AbstractBcubeBackend,
) where {B, TV, Tval}
    # First, we get the values from the integration on the positive/negative side
    # Then, if the face has two side, we seek the values from the opposite side
    unwrapValues = _unwrap_face_integrate(V, values)
    values_i = map(identity, map(identity, map(first, unwrapValues)))
    idofs = get_dofs(V, cellindex(get_cellinfo_n(elementInfo)))
    __update_b!(b, idofs, values_i, backend)
    if (domain isa InteriorFaceDomain) ||
       (domain isa BoundaryFaceDomain{<:AbstractMesh, <:PeriodicBCType})
        values_j = map(identity, map(identity, map(last, unwrapValues)))
        jdofs = get_dofs(V, cellindex(get_cellinfo_p(elementInfo)))
        __update_b!(b, jdofs, values_j, backend)
    end
end
function _unwrap_face_integrate(::Union{TestFESpace, AbstractMultiTestFESpace}, a)
    return unwrap(unwrap(unwrap(a)))
end

function __update_b!(
    b::Tuple{Vararg{Any, N}},
    idofs::Tuple{Vararg{Any, N}},
    intvals::Tuple{Vararg{Any, N}},
    backend::AbstractBcubeBackend,
) where {N}
    f(x1, x2, x3) = __update_b!(x1, x2, x3, backend)
    map(f, b, idofs, intvals)
    nothing
end
function __update_b!(
    b::AbstractVector,
    idofs,
    intvals::Tuple{Vararg{Tuple, N}},
    backend::AbstractBcubeBackend,
) where {N}
    f(x) = __update_b!(b, idofs, x, backend)
    map(f, intvals)
    nothing
end
function __update_b!(
    b::AbstractVector,
    idofs,
    intvals::Tuple{Vararg{Tuple, N}},
    backend::BcubeBackendCPUSerial,  # to remove method ambiguity
) where {N}
    f(x) = __update_b!(b, idofs, x, backend)
    map(f, intvals)
    nothing
end
function __update_b!(b::AbstractVector, dofs, vals, backend::BcubeBackendCPUSerial)
    for (i, val) in zip(dofs, vals)
        b[i] += val
    end
    nothing
end

function __update_b!(
    b::AbstractVector,
    dofs,
    vals::NullOperator,
    backend::AbstractBcubeBackend,
)
    nothing
end

#fix ambiguity
function __update_b!(
    b::AbstractVector,
    dofs,
    vals::NullOperator,
    backend::BcubeBackendCPUSerial,
)
    nothing
end

"""
    _count_n_elts(
        U::TrialFESpace,
        V::TestFESpace,
        domain::CellDomain{M, IND},
    ) where {M, IND}
    function _count_n_elts(
        U::AbstractMultiFESpace{N, Tu},
        V::AbstractMultiFESpace{N, Tv},
        domain::AbstractDomain,
    ) where {N, Tu <: Tuple{Vararg{TrialFESpace}}, Tv <: Tuple{Vararg{TestFESpace}}}
    function _count_n_elts(
        U::TrialFESpace,
        V::TestFESpace,
        domain::BoundaryFaceDomain{M, BC, L, C},
    ) where {M, BC, L, C}


Count the (maximum) number of elements in the matrix corresponding to the bilinear assembly of U, V
on a domain.

# Arguments
- `U::TrialFESpace` : TrialFESpace associated to the first argument of the bilinear form.
- `V::TestFESpace` : TestFESpace associated to the second argument of the bilinear form.
- `domain`::AbstractDomain : domain of integration of the bilinear form.

# Warning
TO DO: for the moment this function is not really implemented for a BoundaryFaceDomain.
This requires to be able to distinguish between the usual TrialsFESpace and MultiplierFESpace.

"""

function _count_n_elts(
    U::TrialFESpace,
    V::TestFESpace,
    domain::CellDomain{M, IND},
) where {M, IND}
    return sum(
        icell -> length(get_dofs(U, icell)) * length(get_dofs(V, icell)),
        indices(domain),
    )
end

function _count_n_elts(
    U::AbstractMultiFESpace{N, Tu},
    V::AbstractMultiFESpace{N, Tv},
    domain::AbstractDomain,
) where {N, Tu <: Tuple{Vararg{TrialFESpace}}, Tv <: Tuple{Vararg{TestFESpace}}}
    n = 0
    for _U in U
        for _V in V
            n += _count_n_elts(_U, _V, domain)
        end
    end
    return n
end

# todo
function _count_n_elts(U::TrialFESpace, V::TestFESpace, domain::AbstractFaceDomain)
    return 1
end

function _count_n_elts(U, V, a::Function)
    integration = a(_null_operator(U), _null_operator(V))
    _count_n_elts(U, V, integration)
end
function _count_n_elts(U, V, integration::Integration)
    return _count_n_elts(U, V, get_domain(get_measure(integration)))
end
function _count_n_elts(U, V, integrations::MultiIntegration)
    return sum(i -> _count_n_elts(U, V, i), integrations)
end

maywrap(x) = LazyWrap(x)
maywrap(x::AbstractLazyOperator) = x

function _get_tuple_var(::Val{N}, k) where {N}
    x -> ntuple(i -> i == k ? x : NullOperator(), Val(N))
end
_get_tuple_var(λ::Tuple, k) = x -> ntuple(i -> i == k ? x : NullOperator(), length(λ)) #ntuple(i->i==k ? λ[k] : NullOperator(), length(λ))
_get_tuple_var(λ, k) = identity

function _get_tuple_var_impl(k, N)
    exprs = [i == k ? :(diag[$i]) : :(b) for i in 1:N]
    return :(tuple($(exprs...)))
end
@generated function _get_tuple_var(x, ::Val{I}, ::Val{N}) where {I, N}
    _get_tuple_var_impl(I, N)
end

function _get_all_tuple_var_impl(N)
    exprs = [_get_tuple_var_impl(i, N) for i in 1:N]
    return :(tuple($(exprs...)))
end

"""
    _diag_tuples(diag::Tuple{Vararg{Any,N}}, b) where N

Return `N` tuples of length `N`. For each tuple `tᵢ`, its values
are defined so that `tᵢ[k]=diag[k]` if `k==i`, `tᵢ[k]=b` otherwise.
The result can be seen as a dense diagonal-like array using tuple.

# Example for `N=3`:

    (diag[1],  b,       b      ),
    (b,        diag[2], b      ),
    (b,        b,       diag[3]))
"""
@generated function _diag_tuples(diag::Tuple{Vararg{Any, N}}, b) where {N}
    _get_all_tuple_var_impl(N)
end

function _diag_tuples(n::Val{N}, ::Val{i}, a, b) where {N, i}
    _diag_tuples(ntuple(k -> a, n), b)[i]
end

"""
For N=3 for example:
    (LazyMapOver((LazyMapOver(V[1]), NullOperator(),  NullOperator())),
     LazyMapOver((NullOperator(),  LazyMapOver(V[2]), NullOperator())),
     LazyMapOver((NullOperator(),  NullOperator(),  LazyMapOver(V[3]))))
"""
function _get_multi_tuple_var(V::Tuple{Vararg{Any, N}}) where {N}
    map(LazyMapOver, _diag_tuples(map(LazyMapOver, V), NullOperator()))
end
_get_multi_tuple_var(a::LazyMapOver) = _get_multi_tuple_var(unwrap(a))

"""
    blockmap_shape_functions(fespace::AbstractFESpace, cellinfo::AbstractCellInfo)

Return all shape functions `a = LazyMapOver((λ₁, λ₂, …, λₙ))` corresponding to `fespace` in cell
`cellinfo`. These shape functions are wrapped by a `LazyMapOver`
so that for a function `f` it gives:
    `f(a) == map(f, a)`
"""
function blockmap_shape_functions(fespace::AbstractFESpace, cellinfo::AbstractCellInfo)
    cshape = shape(celltype(cellinfo))
    λ = get_cell_shape_functions(fespace, cshape)
    LazyMapOver(λ)
end

"""
    blockmap_shape_functions(multiFESpace::AbstractMultiFESpace, cellinfo::AbstractCellInfo)

Return all shape functions corresponding to each `fespace` in `multiFESSpace`
for cell `cellinfo` :
```math
    ((v₁, ∅, ∅, …), (∅, v₂, ∅, …), …, ( …, ∅, ∅, vₙ))
```
where:
* vᵢ = (λᵢ_₁, λᵢ_₂, …, λᵢ_ₘ) are the shapes functions of the i-th fespace in the cell.
* ∅ are `NullOperator`s

Note that the `LazyMapOver` is used to wrap recursively the result.
"""
function blockmap_shape_functions(
    multiFESpace::AbstractMultiFESpace,
    cellinfo::AbstractCellInfo,
)
    cshape = shape(celltype(cellinfo))
    λ = map(LazyMapOver, get_cell_shape_functions(multiFESpace, cshape))
    map(LazyMapOver, _diag_tuples(λ, NullOperator()))
end

"""
# Dev note :
Materialize the integrand function on all the different possible Tuples of
`v=(v1,0,0,...), (0,v2,0,...), ..., (..., vi, ...)`
"""
function blockmap_shape_functions(feSpace, faceinfo::FaceInfo)
    cshape_i = shape(celltype(get_cellinfo_n(faceinfo)))
    cshape_j = shape(celltype(get_cellinfo_p(faceinfo)))
    _cellpair_blockmap_shape_functions(feSpace, cshape_i, cshape_j)
end

function _cellpair_blockmap_shape_functions(
    multiFESpace::AbstractMultiFESpace,
    cshape_i::AbstractShape,
    cshape_j::AbstractShape,
)
    λi = get_cell_shape_functions(multiFESpace, cshape_i)
    λj = get_cell_shape_functions(multiFESpace, cshape_j)
    λij = map(FaceSidePair, map(LazyMapOver, λi), map(LazyMapOver, λj))
    map(LazyMapOver, _diag_tuples(λij, NullOperator()))
end

function _cellpair_blockmap_shape_functions(
    feSpace::AbstractFESpace,
    cshape_i::AbstractShape,
    cshape_j::AbstractShape,
)
    λi = get_cell_shape_functions(feSpace, cshape_i)
    λj = get_cell_shape_functions(feSpace, cshape_j)
    λij = FaceSidePair(LazyMapOver(λi), LazyMapOver(λj))
    LazyMapOver((λij,))
end

"""
# Dev notes:
Return `blockU` and `blockV` to be able to compute
the local matrix corresponding to the bilinear form :
```math
    A[i,j] = a(λᵤ[j], λᵥ[i])
```
where `λᵤ` and `λᵥ` are the shape functions associated with
the trial `U` and the test `V` function spaces respectively.
In a "map-over" version, it can be written :
```math
    A = a(blockU, blockV)
```
where `blockU` and `blockV` correspond formally to
the lazy-map-over matrices :
```math
    ∀k, blockU[k,j] = λᵤ[j],
        blockV[i,k] = λᵥ[i]
```
"""
function blockmap_bilinear_shape_functions(
    U::AbstractFESpace,
    V::AbstractFESpace,
    cellinfo::AbstractCellInfo,
)
    cshape = shape(celltype(cellinfo))
    λU = get_shape_functions(U, cshape)
    λV = get_shape_functions(V, cshape)
    blockV, blockU = _blockmap_bilinear(λV, λU)
    blockU, blockV
end

function blockmap_bilinear_shape_functions(
    U::AbstractFESpace,
    V::AbstractFESpace,
    cellinfo_u::AbstractCellInfo,
    cellinfo_v::AbstractCellInfo,
)
    λU = get_shape_functions(U, shape(celltype(cellinfo_u)))
    λV = get_shape_functions(V, shape(celltype(cellinfo_v)))
    blockV, blockU = _blockmap_bilinear(λV, λU)
    blockU, blockV
end

function blockmap_bilinear_shape_functions(
    U::AbstractFESpace,
    V::AbstractFESpace,
    faceinfo::FaceInfo,
)
    cellinfo_n = get_cellinfo_n(faceinfo)
    cellinfo_p = get_cellinfo_p(faceinfo)
    U_nn, V_nn = blockmap_bilinear_shape_functions(U, V, cellinfo_n, cellinfo_n)
    U_pn, V_pn = blockmap_bilinear_shape_functions(U, V, cellinfo_n, cellinfo_p)
    U_np, V_np = blockmap_bilinear_shape_functions(U, V, cellinfo_p, cellinfo_n)
    U_pp, V_pp = blockmap_bilinear_shape_functions(U, V, cellinfo_p, cellinfo_p)

    return (
        LazyWrap(BilinearTrialFaceSidePair(U_nn, U_pn, U_np, U_pp)),
        LazyWrap(BilinearTestFaceSidePair(V_nn, V_pn, V_np, V_pp)),
    )
end

"""
From tuples ``a=(a_1, a_2, …, a_i, …, a_m)`` and ``b=(b_1, b_2, …, b_j, …, b_n)``,
it builds `A` and `B` which correspond formally to the following two matrices :
```math
A \\equiv \\begin{pmatrix}
a_1 & a_1 & ⋯ & a_1\\\\
a_2 & a_2 & ⋯ & a_2\\\\
 ⋮  &  ⋮  & ⋮ & ⋮  \\\\
a_m & a_m & ⋯ & a_m
\\end{pmatrix}
\\qquad and \\qquad
B \\equiv \\begin{pmatrix}
b_1 & b_2 & ⋯ & b_n\\\\
b_1 & b_2 & ⋯ & b_n\\\\
 ⋮  &  ⋮  & ⋮ & ⋮  \\\\
b_1 & b_2 & ⋯ & b_n
\\end{pmatrix}
```

`A` and `B` are wrapped in `LazyMapOver` structures so that
all operations on them are done elementwise by default (in other words,
it can be considered that the operations are automatically broadcasted).

# Dev note :
Both `A` and `B` are stored as a tuple of tuples, wrapped by `LazyMapOver`, where
inner tuples correspond to each columns of a matrix.
This hierarchical structure reduces both inference and compile times by avoiding the use of large tuples.
"""
function _blockmap_bilinear(a::NTuple{N1}, b::NTuple{N2}) where {N1, N2}
    _a = ntuple(j -> begin
        LazyMapOver(ntuple(i -> a[i], Val(N1)))
    end, Val(N2))
    _b = ntuple(j -> begin
        LazyMapOver(ntuple(i -> b[j], Val(N1)))
    end, Val(N2))
    LazyMapOver(_a), LazyMapOver(_b)
end

function _cartesian_product(a::NTuple{N1}, b::NTuple{N2}) where {N1, N2}
    _a, _b = _cartesian_product(SVector{N1}(a), SVector{N2}(b))
    Tuple(_a), Tuple(_b)
end
function _cartesian_product(a::SVector{N1}, b::SVector{N2}) where {N1, N2}
    # Return `_a` and `__b` defined as :
    # _a = SVector{N1 * N2}(a[i] for i in 1:N1, j in 1:N2)
    # __b = SVector{N1 * N2}(b[j] for i in 1:N1, j in 1:N2)
    _a = repeat(a, Val(N2))
    _b = repeat(b, Val(N1))
    __b = vec(permutedims(reshape(_b, Size(N2, N1))))
    _a, __b
end

Base.repeat(a::SVector, ::Val{N}) where {N} = reduce(vcat, ntuple(i -> a, Val(N)))

"""
    compute(integration::Integration)

Compute an integral, independently from a FEM/DG framework (i.e without FESpace)

Return a `SparseVector`. The indices of the domain elements are used to store
the result of the integration in this sparse vector.

# Example
Compute volume of each cell and each face.
```julia
mesh = rectangle_mesh(2, 3)
dΩ = Measure(CellDomain(mesh), 1)
dΓ = Measure(BoundaryFaceDomain(mesh), 1)
f = PhysicalFunction(x -> 1)
@show Bcube.compute(∫(f)dΩ)
@show Bcube.compute(∫(side⁻(f))dΓ)
```
"""
function compute(integration::Integration)
    measure = get_measure(integration)
    domain = get_domain(measure)
    f = get_function(get_integrand(integration))
    quadrature = get_quadrature(measure)

    values = map_element(domain) do elementInfo
        _f = materialize(f, elementInfo)
        integrate_on_ref_element(_f, elementInfo, quadrature)
    end
    return SparseVector(_domain_to_mesh_nelts(domain), collect(indices(domain)), values)
end

_domain_to_mesh_nelts(domain::AbstractCellDomain) = ncells(get_mesh(domain))
_domain_to_mesh_nelts(domain::AbstractFaceDomain) = nfaces(get_mesh(domain))

"""
    AbstractFaceSidePair{A} <: AbstractLazyWrap{A}

# Interface:
* `side_n(a::AbstractFaceSidePair)`
* `side_p(a::AbstractFaceSidePair)`
"""
abstract type AbstractFaceSidePair{A} <: AbstractLazyWrap{A} end
LazyOperators.get_args(a::AbstractFaceSidePair) = a.data
LazyOperators.pretty_name(::AbstractFaceSidePair) = "AbstractFaceSidePair"
LazyOperators.pretty_name_style(::AbstractFaceSidePair) = Dict(:color => :yellow)

struct FaceSidePair{A} <: AbstractFaceSidePair{A}
    data::A
end

FaceSidePair(a, b) = FaceSidePair((a, b))
LazyOperators.pretty_name(::FaceSidePair) = "FaceSidePair"

side_n(a::FaceSidePair) = Side⁻(LazyMapOver((a.data[1], NullOperator())))
side_p(a::FaceSidePair) = Side⁺(LazyMapOver((NullOperator(), a.data[2])))

function LazyOperators.materialize(a::FaceSidePair, cPoint::CellPoint)
    _a = (materialize(a.data[1], cPoint), materialize(a.data[2], cPoint))
    return MapOver(_a)
end

function LazyOperators.materialize(a::FaceSidePair, side::Side⁻{Nothing, <:Tuple{FaceInfo}})
    return FaceSidePair(materialize(a.data[1], side), NullOperator())
end

function LazyOperators.materialize(a::FaceSidePair, side::Side⁺{Nothing, <:Tuple{FaceInfo}})
    return FaceSidePair(NullOperator(), materialize(a.data[2], side))
end

function LazyOperators.materialize(
    a::FaceSidePair,
    side::Side⁻{Nothing, <:Tuple{FacePoint}},
)
    return MapOver(materialize(a.data[1], side), NullOperator())
end

function LazyOperators.materialize(
    a::FaceSidePair,
    side::Side⁺{Nothing, <:Tuple{FacePoint}},
)
    return MapOver(NullOperator(), materialize(a.data[2], side))
end

function LazyOperators.materialize(
    a::Gradient{O, <:Tuple{AbstractFaceSidePair}},
    point::AbstractSide{Nothing, <:Tuple{FacePoint}},
) where {O}
    _args, = get_operator(point)(get_args(a))
    __args, = get_args(_args)
    return materialize(∇(__args), point)
end

"""
    AbstractBilinearFaceSidePair{A} <: AbstractLazyWrap{A}

# Interface:
    * get_args_bilinear(a::AbstractBilinearFaceSidePair)
"""
abstract type AbstractBilinearFaceSidePair{A} <: AbstractLazyWrap{A} end

LazyOperators.pretty_name_style(::AbstractBilinearFaceSidePair) = Dict(:color => :yellow)
get_args(a::AbstractBilinearFaceSidePair) = a.data
get_basetype(a::AbstractBilinearFaceSidePair) = get_basetype(typeof(a))
get_basetype(::Type{<:AbstractBilinearFaceSidePair}) = error("To be defined")

function LazyOperators.materialize(
    a::AbstractBilinearFaceSidePair,
    point::AbstractSide{Nothing, <:Tuple{FacePoint}},
)
    args = tuplemap(x -> materialize(x, point), get_args(a))
    return MapOver(args)
end

function LazyOperators.materialize(
    a::Gradient{Nothing, <:Tuple{AbstractBilinearFaceSidePair}},
    point::AbstractSide{Nothing, <:Tuple{FacePoint}},
)
    op_side = get_operator(point)
    args, = get_args(op_side(get_args(a))...)
    return materialize(∇(args), point)
end

function LazyOperators.materialize(
    a::AbstractBilinearFaceSidePair,
    side::AbstractSide{Nothing, <:Tuple{FaceInfo}},
)
    T = get_basetype(a)
    return T(tuplemap(x -> materialize(x, side), get_args(a)))
end

struct BilinearTrialFaceSidePair{A} <: AbstractBilinearFaceSidePair{A}
    data::A
end

BilinearTrialFaceSidePair(a...) = BilinearTrialFaceSidePair(a)
LazyOperators.pretty_name(::BilinearTrialFaceSidePair) = "BilinearTrialFaceSidePair"
get_basetype(::Type{<:BilinearTrialFaceSidePair}) = BilinearTrialFaceSidePair

function side_n(a::BilinearTrialFaceSidePair)
    Side⁻(LazyMapOver((a.data[1], a.data[2], NullOperator(), NullOperator())))
end
function side_p(a::BilinearTrialFaceSidePair)
    Side⁺(LazyMapOver((NullOperator(), NullOperator(), a.data[3], a.data[4])))
end

struct BilinearTestFaceSidePair{A} <: AbstractBilinearFaceSidePair{A}
    data::A
end

BilinearTestFaceSidePair(a...) = BilinearTestFaceSidePair(a)
LazyOperators.pretty_name(::BilinearTestFaceSidePair) = "BilinearTestFaceSidePair"
get_basetype(::Type{<:BilinearTestFaceSidePair}) = BilinearTestFaceSidePair

function side_n(a::BilinearTestFaceSidePair)
    Side⁻(LazyMapOver((a.data[1], NullOperator(), a.data[3], NullOperator())))
end
function side_p(a::BilinearTestFaceSidePair)
    Side⁺(LazyMapOver((NullOperator(), a.data[2], NullOperator(), a.data[4])))
end
