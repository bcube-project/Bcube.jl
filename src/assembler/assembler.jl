"""
    _offsets_bilinear_contribution(U, V, domain, backend::AbstractBcubeBackend)

Compute the cumulative offsets and total number of degrees of freedom
for bilinear form assembly on a domain.

Returns a tuple `(offset, totalDof)` where `offset` is an array of cumulative offsets
for each element and `totalDof` is the total number of local DoFs to store.
"""
function _offsets_bilinear_contribution(U, V, domain, backend::AbstractBcubeBackend)
    ndofs = zeros(Int, Bcube.get_nelements(domain))
    foreach_element((e, i, _) -> _nnz_bilinear_by_element!(ndofs, i, e, U, V), domain)
    offsets = accumulate(+, ndofs)
    totalDof = last(offsets)
    offset = vcat(0, offsets[1:(end - 1)])
    return offset, totalDof
end

"""
    _bilinear_integration_type(a, U, V)

Determine the integration type (Single or Multi) by evaluating the bilinear form
with null operators.
"""
_bilinear_integration_type(a, U, V) = a(_null_operator(U), _null_operator(V))

"""
    _linear_integration_type(a, V)

Determine the integration type (Single or Multi) by evaluating the linear form
with a null operator.
"""
_linear_integration_type(a, V) = a(_null_operator(V))

"""
    allocate_linear(backend::AbstractBcubeBackend, V, T)

Allocate memory for a linear form vector on finite element space `V`
with element type `T`.
"""
allocate_linear(backend::AbstractBcubeBackend, V, T) = allocate_dofs(V, T)

"""
    allocate_bilinear(backend::AbstractBcubeBackend, a, U, V, T)

Allocate memory for bilinear form assembly.

Evaluates the bilinear form `a` with null operators to determine the integration
type (single or multiple domains), then dispatches to the appropriate allocate function.
"""
function allocate_bilinear(backend::AbstractBcubeBackend, a, U, V, T)
    integration = a(Bcube._null_operator(U), Bcube._null_operator(V))
    if integration isa MultiIntegration
        domains =
            map(inte -> Bcube.get_domain(Bcube.get_measure(inte)), integration.integrations)
    else
        domains = (Bcube.get_domain(Bcube.get_measure(integration)),)
    end
    allocate_bilinear(backend, domains, U, V, T)
end

"""
    allocate_bilinear(backend::BcubeBackendCPUSerial, domains, U, V, T)

Allocate COO format arrays for bilinear form assembly on given domains.

Computes the total buffer size needed for all elements in all domains,
and allocates COO format arrays (I, J, X) for storing the sparse matrix.
"""
function allocate_bilinear(
    backend::BcubeBackendCPUSerial,
    domains::Tuple{Vararg{AbstractDomain}},
    U,
    V,
    T,
)
    buffersize = sum(domains) do domain
        ndofs = zeros(Int, Bcube.get_nelements(domain))
        foreach_element(domain) do elt, i, _
            _nnz_bilinear_by_element!(ndofs, i, elt, U, V)
        end
        sum(ndofs; init = zero(eltype(ndofs)))
    end

    I = ones(Int, buffersize)
    J = ones(Int, buffersize)
    X = zeros(T, buffersize)
    return I, J, X
end

"""
    _nnz_bilinear_by_element!(ndofs, i, elementInfo, U, V)

Compute and store the number of nonzero entries for the bilinear form
for element `elementInfo` (ith element in the domain) by trial space `U`
and test space `V`.
"""
function _nnz_bilinear_by_element!(ndofs, i, elementInfo, U, V)
    ndofs[i] = _nnz_bilinear_by_element(elementInfo, U, V)
end

"""
    _nnz_bilinear_by_element(elementInfo::CellInfo, U, V)

Compute the number of nonzero entries for a cell element.

Returns the product of the number of DoFs in trial space `U` and test space `V`.
"""
function _nnz_bilinear_by_element(elementInfo::CellInfo, U, V)
    nU = Bcube.get_ndofs(U, shape(Bcube.celltype(elementInfo)))
    nV = Bcube.get_ndofs(V, shape(Bcube.celltype(elementInfo)))
    return nU * nV
end

"""
    _nnz_bilinear_by_element(elementInfo::FaceInfo, U, V)

Compute the number of nonzero entries for a face element.

Accounts for interactions between both sides of the face (n⁻, n⁺),
returning the sum of all four combinations: nn, np, pn, pp.
"""
function _nnz_bilinear_by_element(elementInfo::FaceInfo, U, V)
    cellInfo_n = get_cellinfo_n(elementInfo)
    cellInfo_p = get_cellinfo_p(elementInfo)
    nU_n = Bcube.get_ndofs(U, shape(Bcube.celltype(cellInfo_n)))
    nV_n = Bcube.get_ndofs(V, shape(Bcube.celltype(cellInfo_n)))
    kdofs = nU_n * nV_n

    # TODO : boundary faces can be skipped:
    # if get_element_index(cellInfo_n) ≠ get_element_index(cellInfo_p)
    nU_p = Bcube.get_ndofs(U, shape(Bcube.celltype(cellInfo_p)))
    nV_p = Bcube.get_ndofs(V, shape(Bcube.celltype(cellInfo_p)))
    kdofs += nU_n * nV_p + nU_p * nV_n + nU_p * nV_p
    # end

    return kdofs
end

"""
    init_bilinear(backend::BcubeBackendCPUSerial, a, U, V, T)

Initialize empty COO (coordinate) format arrays for bilinear form assembly.

Returns empty integer arrays `I, J` for row and column indices, and empty array `X` for values.
"""
function init_bilinear(backend::BcubeBackendCPUSerial, a, U, V, T)
    I = zeros(Int, 0)
    J = zeros(Int, 0)
    X = zeros(T, 0)
    return I, J, X
end

"""
    assemble_bilinear(a::Function, U, V; T = Float64)

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

# Dev note
- step 1: dispatch on `Integration`/`MultiIntegration` with `_assemble_bilinear!`
- step 2: dispatch on `SingleFESpace`/`MultiFESpace` with `__assemble_bilinear!`
"""
function assemble_bilinear(
    a::Function,
    U::Union{TrialFESpace, AbstractMultiFESpace{N}},
    V::Union{TestFESpace, AbstractMultiFESpace{N}};
    T = Float64,
) where {N}
    backend = get_bcube_backend(_bilinear_integration_type(a, U, V))
    I, J, X = allocate_bilinear(backend, a, U, V, T)

    # Compute
    offset = assemble_bilinear!(I, J, X, a, U, V)
    nrows = get_ndofs(V)
    ncols = get_ndofs(U)
    return sparse(I, J, X, nrows, ncols)
end

"""
    assemble_bilinear!(I::AbstractVector, J::AbstractVector, X::AbstractVector,
                       a::Function, U, V) where {N}

In-place assembly of bilinear form `a` on spaces `U` (trial) and `V` (test).

Populates the COO format arrays `I`, `J`, `X` representing row indices, column indices,
and values of the sparse matrix. Returns the final offset in the arrays.
"""
function assemble_bilinear!(
    I::AbstractVector,
    J::AbstractVector,
    X::AbstractVector,
    a::Function,
    U::Union{TrialFESpace, AbstractMultiFESpace{N}},
    V::Union{TestFESpace, AbstractMultiFESpace{N}},
) where {N}
    backend = get_bcube_backend(_bilinear_integration_type(a, U, V))
    integration_type = _bilinear_integration_type(a, U, V)
    offset = 0
    offset = _assemble_bilinear!(I, J, X, offset, a, U, V, integration_type, backend)
    return offset
end

"""
    _assemble_bilinear!(I, J, X, offset, a, U, V, multiIntegration::MultiIntegration{N}, backend)

Internal assembly function that dispatches over multiple integration domains.

Loops over each integration in the `MultiIntegration` and calls `_assemble_bilinear!`
recursively with the extracted bilinear form component.
"""
function _assemble_bilinear!(
    I::AbstractVector,
    J::AbstractVector,
    X::AbstractVector,
    offset::Int,
    a::F,
    U,
    V,
    multiIntegration::MultiIntegration{N},
    backend::AbstractBcubeBackend,
) where {F <: Function, N}
    for i in 1:N
        ival = Val(i)
        aᵢ(u, v) = a(u, v)[ival]
        integration = multiIntegration[ival]
        offset = _assemble_bilinear!(I, J, X, offset, aᵢ, U, V, integration, backend)
    end
    return offset
end

"""
    _assemble_bilinear!(I, J, X, offset, a, U, V, integration::Integration, backend)

Internal assembly function that dispatches over a single integration.

Extracts the function from the integrand and calls `__assemble_bilinear!` with the measure.
"""
function _assemble_bilinear!(
    I::AbstractVector,
    J::AbstractVector,
    X::AbstractVector,
    offset::Int,
    a::F,
    U,
    V,
    integration::Integration,
    backend::AbstractBcubeBackend,
) where {F <: Function}
    f = get_function ∘ get_integrand ∘ a
    measure = get_measure(integration)
    offset = __assemble_bilinear!(I, J, X, offset, f, measure, U, V, backend)
    return offset
end

"""
    __assemble_bilinear!(
        I,
        J,
        X,
        offset0::Int,
        f::F,
        measure::Measure,
        U::TrialFESpace,
        V::TestFESpace,
        backend::AbstractBcubeBackend,
    ) where {F <: Function}


Perform assembly of a bilinear form (function `f` integrated on the `measure`).
"""
function __assemble_bilinear!(
    I,
    J,
    X,
    offset0::Int,
    f::F,
    measure::Measure,
    U::TrialFESpace,
    V::TestFESpace,
    backend::AbstractBcubeBackend,
) where {F <: Function}
    # Alias
    quadrature = get_quadrature(measure)
    domain = get_domain(measure)

    offsets, nDofs = _offsets_bilinear_contribution(U, V, domain, backend)

    # Loop over cells
    foreach_element(domain) do elementInfo, i, _
        λu, λv = blockmap_bilinear_shape_functions(U, V, elementInfo)
        g1 = materialize(f(λu, λv), elementInfo)
        values = integrate_on_ref_element(g1, elementInfo, quadrature)
        offset = offset0 + offsets[i]
        _append_contribution!(X, I, J, offset, U, V, values, elementInfo, domain, backend)
    end

    return offset0 + nDofs
end

"""
    __assemble_bilinear!(
        I::AbstractVector,
        J::AbstractVector,
        X::AbstractVector,
        offset::Int,
        f::F,
        measure::Measure,
        U::AbstractMultiFESpace{N, <:Tuple{Vararg{TrialFESpace, N}}},
        V::AbstractMultiFESpace{N, <:Tuple{Vararg{TestFESpace, N}}},
        backend::AbstractBcubeBackend,
    ) where {F <: Function, N}

Assemble bilinear form over all combinations of component spaces in MultiFESpaces.

Loops over all pairs of component spaces (i, j) and calls `assemble_bilinear_by_singleFE!`
to handle each component.
"""
function __assemble_bilinear!(
    I::AbstractVector,
    J::AbstractVector,
    X::AbstractVector,
    offset::Int,
    f::F,
    measure::Measure,
    U::AbstractMultiFESpace{N, <:Tuple{Vararg{TrialFESpace, N}}},
    V::AbstractMultiFESpace{N, <:Tuple{Vararg{TestFESpace, N}}},
    backend::AbstractBcubeBackend,
) where {F <: Function, N}
    # Loop over all combinations
    for (j, _U) in enumerate(U)
        for (i, _V) in enumerate(V)
            # Materialize function for `(..., uj, ...), (...,vi,...)`
            offset = assemble_bilinear_by_singleFE!(
                I,
                J,
                X,
                offset,
                Val(N),
                Val(i),
                Val(j),
                f,
                measure,
                U,
                V,
                _U,
                _V,
                backend,
            )
        end
    end

    return offset
end

"""
    assemble_bilinear_by_singleFE!(
        I::AbstractVector,
        J::AbstractVector,
        X::AbstractVector,
        offset0::Int,
        valN::Val{VN},
        valI::Val{VI},
        valJ::Val{VJ},
        f::F,
        measure::Measure,
        U,
        V,
        _U,
        _V,
        backend::AbstractBcubeBackend,
    ) where {VN, VI, VJ, F <: Function}

Assemble bilinear form for a specific pair of component spaces in a MultiFESpace.

Constructs the appropriate function materialization for the component spaces,
and performs assembly, then maps the resulting indices back to the global numbering.

`U` is the initial multi TrialFESpace, while `_U` is one of the single TrialFESpace
composing `U`. The same goes with `V` vs `_V`
"""
function assemble_bilinear_by_singleFE!(
    I::AbstractVector,
    J::AbstractVector,
    X::AbstractVector,
    offset0::Int,
    valN::Val{VN},
    valI::Val{VI},
    valJ::Val{VJ},
    f::F,
    measure::Measure,
    U,
    V,
    _U,
    _V,
    backend::AbstractBcubeBackend,
) where {VN, VI, VJ, F <: Function}
    # Materialize function for `(..., uj, ...), (...,vi,...)`
    function _f(uj, vi)
        f(
            _get_tuple_var1(valN, valJ, uj, NullOperator()),
            _get_tuple_var1(valN, valI, vi, NullOperator()),
        )
    end

    # Skip computation if there is nothing to compute
    isa(_f(maywrap(_U), maywrap(_V)), NullOperator) && (return offset0)
    offset = __assemble_bilinear!(I, J, X, offset0, _f, measure, _U, _V, backend)
    I[(offset0 + 1):offset] = get_mapping(V, VI)[I[(offset0 + 1):offset]]
    J[(offset0 + 1):offset] = get_mapping(U, VJ)[J[(offset0 + 1):offset]]
    return offset
end

"""
    assemble_linear(l::Function, V::Union{TestFESpace, AbstractMultiTestFESpace}; T = Float64)

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
)
    backend = get_bcube_backend(_linear_integration_type(l, V))

    b = allocate_linear(backend, V, T)
    assemble_linear!(b, l, V)
    return b
end

"""
    assemble_linear!(b::AbstractVector, l::Function, V::Union{TestFESpace, AbstractMultiTestFESpace})

In-place version of [`assemble_linear`](@ref).
"""
function assemble_linear!(
    b::AbstractVector,
    l::Function,
    V::Union{TestFESpace, AbstractMultiTestFESpace},
)
    backend = get_bcube_backend(_linear_integration_type(l, V))
    _assemble_linear!(b, l, V, _linear_integration_type(l, V), backend)
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
    __assemble_linear!(b, f, V, measure, backend)
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
    foreach_element(domain) do elementInfo, _, _
        vₑ = blockmap_shape_functions(V, elementInfo)
        fᵥ = materialize(f(vₑ), elementInfo)
        values = integrate_on_ref_element(fᵥ, elementInfo, quadrature)
        _update_b!(b, V, values, elementInfo, domain, backend)
    end
    nothing
end

"""
    _null_operator(::AbstractFESpace)
    _null_operator(::AbstractMultiFESpace{N})

Create a null operator placeholder for a finite element space.

For single spaces, returns a `NullOperator()`. For multi spaces, returns a tuple of
`NullOperator`s matching the number of component spaces.
"""
_null_operator(::AbstractFESpace) = NullOperator()
_null_operator(::AbstractMultiFESpace{N}) where {N} = ntuple(i -> NullOperator(), Val(N))

"""
    _append_contribution!(X, I, J, offset, U, V, values, elementInfo::CellInfo, domain, backend)

Append contributions to the COO format arrays for a cell element.

Extracts DoF indices for the trial and test spaces and appends the computed
values to the sparse matrix arrays.
"""
function _append_contribution!(
    X,
    I,
    J,
    offset,
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
    _append_bilinear!(I, J, X, offset, Vdofs, Udofs, unwrapValues, backend)
    return nothing
end

"""
    _append_contribution!(X, I, J, offset, U, V, values, elementInfo::FaceInfo, domain, backend)

Append contributions to the COO format arrays for a face element.

Handles both sides of a face (n⁻ and n⁺) and adds all four combinations of
DoF interactions (n⁻-n⁻, n⁻-n⁺, n⁺-n⁻, n⁺-n⁺) to the sparse matrix arrays.
"""
function _append_contribution!(
    X,
    I,
    J,
    offset,
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

    _offset = offset
    for (k, (row, col)) in enumerate(
        Iterators.product((row_dofs_V_n, row_dofs_V_p), (col_dofs_U_n, col_dofs_U_p)),
    )
        _offset = _append_bilinear!(I, J, X, _offset, row, col, unwrapValues[k], backend)
    end
    return nothing
end

"""
    _append_bilinear!(I, J, X, offset, row, col, vals, backend)

Append a bilinear contribution to the COO format arrays.

Takes row and column DoF arrays and computes their Cartesian product,
then appends the resulting indices and values to the sparse matrix arrays.
"""
function _append_bilinear!(I, J, X, offset, row, col, vals, backend::AbstractBcubeBackend)
    _rows, _cols = Bcube._cartesian_product(row, col)
    for k in eachindex(_rows)
        I[offset + k] = _rows[k]
    end
    for k in eachindex(_rows)
        J[offset + k] = _cols[k]
    end
    k = 0
    if !isa(vals, NullOperator)
        for vi in vals
            for vij in vi
                k += 1
                X[offset + k] += vij
            end
        end
    end
    return offset + length(_rows)
end

Base.getindex(::Bcube.LazyOperators.NullOperator, i) = NullOperator()

"""
    _unwrap_face_integrate(::Union{TrialFESpace, AbstractMultiTrialFESpace},
                           ::Union{TestFESpace, AbstractMultiTestFESpace}, a)

Recursively unwrap the lazy evaluation structure for face integrations.

Extracts the actual computed values from the nested lazy operations.
"""
function _unwrap_face_integrate(
    ::Union{TrialFESpace, AbstractMultiTrialFESpace},
    ::Union{TestFESpace, AbstractMultiTestFESpace},
    a,
)
    return _recursive_unwrap(a)
end

"""
    _recursive_unwrap(a::LazyOperators.AbstractMapOver)
    _recursive_unwrap(a)

Recursively unwrap lazy map operations to extract concrete values.

For `AbstractMapOver` objects, maps `_recursive_unwrap` over the elements.
For other types, applies `unwrap` once.
"""
_recursive_unwrap(a::LazyOperators.AbstractMapOver) = map(_recursive_unwrap, unwrap(a))
_recursive_unwrap(a) = unwrap(a)

"""
    _map_idofs(V::AbstractFESpace, idofs::SVector)

Map DoF indices for a single finite element space (identity operation).

For single spaces, returns the indices unchanged.
"""
_map_idofs(V::AbstractFESpace, idofs::SVector) = idofs

"""
    _map_idofs(V::AbstractMultiFESpace{N}, idofs::Tuple{Vararg{SVector, N}})

Map DoF indices for a multi-space, applying component-wise mappings.

Uses the space's global DoF mappings to renumber local DoFs to global indices.
"""
function _map_idofs(V::AbstractMultiFESpace{N}, idofs::Tuple{Vararg{SVector, N}}) where {N}
    __map_idofs(get_mapping(V), idofs)
end
"""
    __map_idofs(mappings::Tuple{Vararg{AbstractArray, N}}, idofs::Tuple{Vararg{SVector, N}})

Recursively map DoF indices using component-specific mappings.

Applies each mapping to the corresponding component space's DoFs.
"""
function __map_idofs(
    mappings::Tuple{Vararg{AbstractArray, N}},
    idofs::Tuple{Vararg{SVector, N}},
) where {N}
    (first(mappings)[first(idofs)], __map_idofs(Base.tail(mappings), Base.tail(idofs))...)
end
function __map_idofs(
    mappings::Tuple{Vararg{AbstractArray, 1}},
    idofs::Tuple{Vararg{SVector, 1}},
)
    (first(mappings)[first(idofs)],)
end

"""
    _update_b!(b, V, values, elementInfo::CellInfo, domain, backend)

Update the linear form vector for a cell element.

Extracts DoF indices and integrated values, then updates the result vector.
"""
function _update_b!(
    b,
    V,
    values,
    elementInfo::CellInfo,
    domain,
    backend::AbstractBcubeBackend,
)
    idofs = get_dofs(V, cellindex(elementInfo), shape(celltype(elementInfo)))
    _idofs = _map_idofs(V, idofs)
    unwrapValues = _unwrap_cell_integrate(V, values)
    __update_b!(b, _idofs, unwrapValues, backend)
end
"""
    _unwrap_cell_integrate(::TestFESpace, a)
    _unwrap_cell_integrate(::AbstractMultiTestFESpace, a)

Unwrap lazy evaluation structures for cell integrations.

Maps `unwrap` twice over the lazy structure to extract computed values.
"""
_unwrap_cell_integrate(::TestFESpace, a) = map(unwrap, unwrap(a))
_unwrap_cell_integrate(::AbstractMultiTestFESpace, a) = map(unwrap, unwrap(a))

"""
    _update_b!(b, V, values, elementInfo::FaceInfo, domain, backend)

Update the linear form vector for a face element.

Handles both sides of the face (n⁻ and n⁺) and updates the corresponding
DoFs with the integrated values.
"""
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
    idofs = get_dofs(
        V,
        cellindex(get_cellinfo_n(elementInfo)),
        shape(celltype(get_cellinfo_n(elementInfo))),
    )
    idofs = _map_idofs(V, idofs)

    __update_b!(b, idofs, values_i, backend)
    if (domain isa InteriorFaceDomain) ||
       (domain isa BoundaryFaceDomain{<:AbstractMesh, <:PeriodicBCType})
        values_j = map(identity, map(identity, map(last, unwrapValues)))
        jdofs = get_dofs(
            V,
            cellindex(get_cellinfo_p(elementInfo)),
            shape(celltype(get_cellinfo_n(elementInfo))),
        )
        jdofs = _map_idofs(V, jdofs)
        __update_b!(b, jdofs, values_j, backend)
    end
end
"""
    _unwrap_face_integrate(::Union{TestFESpace, AbstractMultiTestFESpace}, a)

Unwrap lazy evaluation structures for face integrations in linear forms.

Maps `unwrap` three times to extract values from nested lazy operations.
"""
function _unwrap_face_integrate(::Union{TestFESpace, AbstractMultiTestFESpace}, a)
    return unwrap(unwrap(unwrap(a)))
end

"""
    __update_b!(b, idofs::Tuple, intvals::Tuple, backend)

Update linear form vector for multi-space with component-wise updates.

Maps over each component space, updating its DoFs with the corresponding values.
"""
function __update_b!(
    b::AbstractVector{T},
    idofs::Tuple{Vararg{Union{Tuple, SVector}, N}},
    intvals::Tuple{Vararg{Tuple, N}},
    backend::AbstractBcubeBackend,
) where {N, T}
    f(x2, x3) = __update_b!(b, x2, x3, backend)
    map(f, idofs, intvals)
    nothing
end

"""
    __update_b!(b, idofs::AbstractVector, intvals::Tuple, backend)

Update linear form vector with integrated values for a single component.

Maps over the tuple of values and recursively calls `__update_b!`.
"""
function __update_b!(
    b::AbstractVector{T},
    idofs::AbstractVector,
    intvals::Tuple{NTuple{N, T}},
    backend::AbstractBcubeBackend,
) where {T, N}
    __update_b!(b, idofs, first(intvals), backend)
    nothing
end

"""
    __update_b!(b, dofs::AbstractVector{<:Integer}, vals::NTuple, backend::BcubeBackendCPUSerial)

Actually update the linear form vector entries for CPU serial backend.

Adds the integrated values to the corresponding DoF entries in the result vector.
"""
function __update_b!(
    b::AbstractVector{T},
    dofs::AbstractVector{<:Integer},
    vals::NTuple{N, T},
    backend::BcubeBackendCPUSerial,
) where {N, T}
    for (i, val) in zip(dofs, vals)
        b[i] += val
    end
    nothing
end

"""
    __update_b!(b, dofs, vals::NullOperator, backend)

No-op update when values are null operators.

Skips processing when there are no actual values to add.
"""
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
    maywrap(x)
    maywrap(x::AbstractLazyOperator)

Wrap a value in a `LazyWrap` unless it is already a lazy operator.

For lazy operators, returns them unchanged. Otherwise wraps in `LazyWrap`.
"""
maywrap(x) = LazyWrap(x)
maywrap(x::AbstractLazyOperator) = x

"""
    _get_tuple_var(::Val{N}, k)

Create a function that wraps a value into a tuple of length N with NullOperators.

The wrapped value is placed at position k, other positions are filled with NullOperators.
"""
function _get_tuple_var(::Val{N}, k) where {N}
    x -> ntuple(i -> i == k ? x : NullOperator(), Val(N))
end

"""
    _get_tuple_var1_impl(N, K)

Generate code expression for creating a tuple with value at position K.

Used by `@generated` function `_get_tuple_var1` to create efficient code.
"""
function _get_tuple_var1_impl(N, K)
    exprs = [i == K ? :x : :y for i in 1:N]
    return :(tuple($(exprs...)))
end
@generated function _get_tuple_var1(n::Val{N}, k::Val{K}, x::X, y::Y) where {N, K, X, Y}
    _get_tuple_var1_impl(N, K)
end
"""
    _get_tuple_var(λ::Tuple, k)
    _get_tuple_var(λ, k)

Create a function that wraps a value into a tuple of NullOperators except at position k.

For tuple input, returns a function creating tuple of same length.
For other inputs, returns identity.
"""
_get_tuple_var(λ::Tuple, k) = x -> ntuple(i -> i == k ? x : NullOperator(), length(λ))
_get_tuple_var(λ, k) = identity

"""
    _get_tuple_var_impl(k, N)

Generate code expression for creating a tuple selecting from diagonal or background value.

Used by `@generated` function to efficiently create diagonal-like tuple.
"""
function _get_tuple_var_impl(k, N)
    exprs = [i == k ? :(diag[$i]) : :(b) for i in 1:N]
    return :(tuple($(exprs...)))
end
@generated function _get_tuple_var(x, ::Val{I}, ::Val{N}) where {I, N}
    _get_tuple_var_impl(I, N)
end

"""
    _get_all_tuple_var_impl(N)

Generate code expression for creating N tuples of diagonal-like structure.

Used by `@generated` function `_diag_tuples` to create efficient code.
"""
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

"""
    _diag_tuples(n::Val{N}, ::Val{i}, a, b)

Get the i-th diagonal-like tuple from the diag_tuples structure.

Selects the i-th row of the N×N diagonal-like tuple structure.
"""
function _diag_tuples(n::Val{N}, ::Val{i}, a, b) where {N, i}
    _diag_tuples(ntuple(k -> a, n), b)[i]
end

"""
For N=3 for example:
    (LazyMapOver((LazyMapOver(V[1]), NullOperator(),  NullOperator())),
     LazyMapOver((NullOperator(),  LazyMapOver(V[2]), NullOperator())),
     LazyMapOver((NullOperator(),  NullOperator(),  LazyMapOver(V[3]))))
"""
"""
    _get_multi_tuple_var(V::Tuple{Vararg{Any, N}})
    _get_multi_tuple_var(a::LazyMapOver)

Create block-diagonal tuple structure for multi-space shape functions.

For use in MultiFESpace assembly, where each component space is
mapped in a diagonal block pattern with NullOperators elsewhere.
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
    blockmap_shape_functions(feSpace, faceinfo::FaceInfo)

Return block-mapped shape functions for a face element.

Combines shape functions from both sides of the face (n⁻ and n⁺).

# Dev note :
Materialize the integrand function on all the different possible Tuples of
`v=(v1,0,0,...), (0,v2,0,...), ..., (..., vi, ...)`
"""
function blockmap_shape_functions(feSpace, faceinfo::FaceInfo)
    cshape_i = shape(celltype(get_cellinfo_n(faceinfo)))
    cshape_j = shape(celltype(get_cellinfo_p(faceinfo)))
    _cellpair_blockmap_shape_functions(feSpace, cshape_i, cshape_j)
end

"""
    _cellpair_blockmap_shape_functions(multiFESpace::AbstractMultiFESpace,
                                       cshape_i::AbstractShape, cshape_j::AbstractShape)

Create block-diagonal shape function pairs for two cell shapes in a MultiFESpace.

Returns FaceSidePairs with LazyMapOver wrappers for each component space.
"""
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

"""
    _cellpair_blockmap_shape_functions(feSpace::AbstractFESpace,
                                       cshape_i::AbstractShape, cshape_j::AbstractShape)

Create shape function pair for two cell shapes in a single FESpace.

Returns a FaceSidePair wrapped in LazyMapOver for use in face integrations.
"""
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
"""
    blockmap_bilinear_shape_functions(U::AbstractFESpace, V::AbstractFESpace,
                                      cellinfo::AbstractCellInfo)

Return block-mapped bilinear shape functions for trial and test spaces on a cell.

Creates lazy-wrapped structures suitable for evaluating bilinear forms:
`A[i,j] = a(λᵤ[j], λᵥ[i])` in block form.
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

"""
    blockmap_bilinear_shape_functions(U::AbstractFESpace, V::AbstractFESpace,
                                      cellinfo_u::AbstractCellInfo, cellinfo_v::AbstractCellInfo)

Return block-mapped bilinear shape functions for possibly different cell shapes.

Useful for cases where trial and test functions may be defined on different cell types.
"""
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

"""
    blockmap_bilinear_shape_functions(U::AbstractFESpace, V::AbstractFESpace, faceinfo::FaceInfo)

Return block-mapped bilinear shape functions for face integrations.

Creates four combinations of shape functions for the two sides of the face (n⁻, n⁺):
nn, pn, np, pp.
"""
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
"""
    _blockmap_bilinear(a::NTuple{N1}, b::NTuple{N2})

Create block-diagonal lazy-mapped bilinear structure from tuple pairs.

Returns two LazyMapOver structures `A` and `B` such that `A[i,j] = a[i]`
and `B[i,j] = b[j]`, forming implicit dense diagonal-like matrices.
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

"""
    _cartesian_product(a::NTuple{N1}, b::NTuple{N2})

Compute the Cartesian product of two tuples.

Returns two tuples where the first contains all combinations from `a` repeated,
and the second contains all combinations from `b` with proper ordering.
"""
function _cartesian_product(a::NTuple{N1}, b::NTuple{N2}) where {N1, N2}
    _a, _b = _cartesian_product(SVector{N1}(a), SVector{N2}(b))
    Tuple(_a), Tuple(_b)
end
"""
    _cartesian_product(a::SVector{N1}, b::SVector{N2})

Compute the Cartesian product of two static vectors.

Efficiently creates the row and column index patterns for Cartesian product
using repetition and permutation.
"""
function _cartesian_product(a::SVector{N1}, b::SVector{N2}) where {N1, N2}
    # Return `_a` and `__b` defined as :
    # _a = SVector{N1 * N2}(a[i] for i in 1:N1, j in 1:N2)
    # __b = SVector{N1 * N2}(b[j] for i in 1:N1, j in 1:N2)
    _a = repeat(a, Val(N2))
    _b = repeat(b, Val(N1))
    __b = vec(permutedims(reshape(_b, Size(N2, N1))))
    _a, __b
end

"""
    Base.repeat(a::SVector, ::Val{N})

Repeat a static vector N times by vertical concatenation.
"""
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
"""
    AbstractFaceSidePair{A} <: AbstractLazyWrap{A}

Abstract base type for wrapping shape functions on face pairs (two sides of a face).

# Interface:
* `side_n(a::AbstractFaceSidePair)` - Extract shape functions from negative side
* `side_p(a::AbstractFaceSidePair)` - Extract shape functions from positive side
"""
abstract type AbstractFaceSidePair{A} <: AbstractLazyWrap{A} end
LazyOperators.get_args(a::AbstractFaceSidePair) = a.data
LazyOperators.pretty_name(::AbstractFaceSidePair) = "AbstractFaceSidePair"
LazyOperators.pretty_name_style(::AbstractFaceSidePair) = Dict(:color => :yellow)

"""
    FaceSidePair{A}

Concrete wrapper for shape functions on face pairs.

Stores data as `(side_n_data, side_p_data)` tuple.
"""
struct FaceSidePair{A} <: AbstractFaceSidePair{A}
    data::A
end

"""
    FaceSidePair(a, b)

Construct a FaceSidePair from two arguments.
"""
FaceSidePair(a, b) = FaceSidePair((a, b))
LazyOperators.pretty_name(::FaceSidePair) = "FaceSidePair"

"""
    side_n(a::FaceSidePair)

Extract shape functions from the negative side of a face.

Returns a Side⁻ wrapped LazyMapOver with the first component and NullOperators for others.
"""
side_n(a::FaceSidePair) = Side⁻(LazyMapOver((a.data[1], NullOperator())))

"""
    side_p(a::FaceSidePair)

Extract shape functions from the positive side of a face.

Returns a Side⁺ wrapped LazyMapOver with the second component and NullOperators for others.
"""
side_p(a::FaceSidePair) = Side⁺(LazyMapOver((NullOperator(), a.data[2])))

"""
    LazyOperators.materialize(a::FaceSidePair, cPoint::CellPoint)

Materialize a FaceSidePair at a cell point.

Evaluates both sides of the pair at the given point and returns a MapOver structure.
"""
function LazyOperators.materialize(a::FaceSidePair, cPoint::CellPoint)
    _a = (materialize(a.data[1], cPoint), materialize(a.data[2], cPoint))
    return MapOver(_a)
end

"""
    LazyOperators.materialize(a::FaceSidePair, side::Side⁻{Nothing, <:Tuple{FaceInfo}})

Materialize a FaceSidePair on the negative side of a face.

Extracts only the negative side data and materializes it.
"""
function LazyOperators.materialize(a::FaceSidePair, side::Side⁻{Nothing, <:Tuple{FaceInfo}})
    return FaceSidePair(materialize(a.data[1], side), NullOperator())
end

"""
    LazyOperators.materialize(a::FaceSidePair, side::Side⁺{Nothing, <:Tuple{FaceInfo}})

Materialize a FaceSidePair on the positive side of a face.

Extracts only the positive side data and materializes it.
"""
function LazyOperators.materialize(a::FaceSidePair, side::Side⁺{Nothing, <:Tuple{FaceInfo}})
    return FaceSidePair(NullOperator(), materialize(a.data[2], side))
end

"""
    LazyOperators.materialize(a::FaceSidePair, side::Side⁻{Nothing, <:Tuple{FacePoint}})

Materialize a FaceSidePair on the negative side at a face point.

Extracts and materializes the negative side data in a MapOver structure.
"""
function LazyOperators.materialize(
    a::FaceSidePair,
    side::Side⁻{Nothing, <:Tuple{FacePoint}},
)
    return MapOver(materialize(a.data[1], side), NullOperator())
end

"""
    LazyOperators.materialize(a::FaceSidePair, side::Side⁺{Nothing, <:Tuple{FacePoint}})

Materialize a FaceSidePair on the positive side at a face point.

Extracts and materializes the positive side data in a MapOver structure.
"""
function LazyOperators.materialize(
    a::FaceSidePair,
    side::Side⁺{Nothing, <:Tuple{FacePoint}},
)
    return MapOver(NullOperator(), materialize(a.data[2], side))
end

"""
    LazyOperators.materialize(a::Gradient{O, <:Tuple{AbstractFaceSidePair}},
                              point::AbstractSide{Nothing, <:Tuple{FacePoint}}) where {O}

Materialize the gradient of a FaceSidePair at a face point.

Applies the side operator to extract the appropriate side data, then computes the gradient.
"""
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

Abstract base type for wrapping bilinear form shape functions on face pairs.

Specializes AbstractFaceSidePair for use in bilinear forms with trial and test functions.

# Interface:
* `get_args_bilinear(a::AbstractBilinearFaceSidePair)` - Extract bilinear arguments
"""
abstract type AbstractBilinearFaceSidePair{A} <: AbstractLazyWrap{A} end

LazyOperators.pretty_name_style(::AbstractBilinearFaceSidePair) = Dict(:color => :yellow)
get_args(a::AbstractBilinearFaceSidePair) = a.data
get_basetype(a::AbstractBilinearFaceSidePair) = get_basetype(typeof(a))
get_basetype(::Type{<:AbstractBilinearFaceSidePair}) = error("To be defined")

"""
    LazyOperators.materialize(a::AbstractBilinearFaceSidePair,
                              point::AbstractSide{Nothing, <:Tuple{FacePoint}})

Materialize a bilinear face side pair at a face point.

Maps the materialize operation over all stored data and returns a MapOver structure.
"""
function LazyOperators.materialize(
    a::AbstractBilinearFaceSidePair,
    point::AbstractSide{Nothing, <:Tuple{FacePoint}},
)
    args = tuplemap(x -> materialize(x, point), get_args(a))
    return MapOver(args)
end

"""
    LazyOperators.materialize(a::Gradient{Nothing, <:Tuple{AbstractBilinearFaceSidePair}},
                              point::AbstractSide{Nothing, <:Tuple{FacePoint}})

Materialize the gradient of a bilinear face side pair at a face point.

Applies the side operator to extract the appropriate data, then computes the gradient.
"""
function LazyOperators.materialize(
    a::Gradient{Nothing, <:Tuple{AbstractBilinearFaceSidePair}},
    point::AbstractSide{Nothing, <:Tuple{FacePoint}},
)
    op_side = get_operator(point)
    args, = get_args(op_side(get_args(a))...)
    return materialize(∇(args), point)
end

"""
    LazyOperators.materialize(a::AbstractBilinearFaceSidePair,
                              side::AbstractSide{Nothing, <:Tuple{FaceInfo}})

Materialize a bilinear face side pair at a face info.

Maps materialize over all data and wraps in the appropriate concrete type.
"""
function LazyOperators.materialize(
    a::AbstractBilinearFaceSidePair,
    side::AbstractSide{Nothing, <:Tuple{FaceInfo}},
)
    T = get_basetype(a)
    return T(tuplemap(x -> materialize(x, side), get_args(a)))
end

"""
    BilinearTrialFaceSidePair{A}

Concrete wrapper for trial shape functions on face pairs.

Stores four combinations: (nn, pn, np, pp) where n denotes negative side and p denotes positive side.
"""
struct BilinearTrialFaceSidePair{A} <: AbstractBilinearFaceSidePair{A}
    data::A
end

"""
    BilinearTrialFaceSidePair(a...)

Construct a BilinearTrialFaceSidePair from variable arguments.
"""
BilinearTrialFaceSidePair(a...) = BilinearTrialFaceSidePair(a)
LazyOperators.pretty_name(::BilinearTrialFaceSidePair) = "BilinearTrialFaceSidePair"
get_basetype(::Type{<:BilinearTrialFaceSidePair}) = BilinearTrialFaceSidePair

"""
    side_n(a::BilinearTrialFaceSidePair)

Extract trial functions from the negative side.

Returns shape functions for (nn, pn) combinations.
"""
function side_n(a::BilinearTrialFaceSidePair)
    Side⁻(LazyMapOver((a.data[1], a.data[2], NullOperator(), NullOperator())))
end

"""
    side_p(a::BilinearTrialFaceSidePair)

Extract trial functions from the positive side.

Returns shape functions for (np, pp) combinations.
"""
function side_p(a::BilinearTrialFaceSidePair)
    Side⁺(LazyMapOver((NullOperator(), NullOperator(), a.data[3], a.data[4])))
end

"""
    BilinearTestFaceSidePair{A}

Concrete wrapper for test shape functions on face pairs.

Stores four combinations: (nn, pn, np, pp) where n denotes negative side and p denotes positive side.
"""
struct BilinearTestFaceSidePair{A} <: AbstractBilinearFaceSidePair{A}
    data::A
end

"""
    BilinearTestFaceSidePair(a...)

Construct a BilinearTestFaceSidePair from variable arguments.
"""
BilinearTestFaceSidePair(a...) = BilinearTestFaceSidePair(a)
LazyOperators.pretty_name(::BilinearTestFaceSidePair) = "BilinearTestFaceSidePair"
get_basetype(::Type{<:BilinearTestFaceSidePair}) = BilinearTestFaceSidePair

"""
    side_n(a::BilinearTestFaceSidePair)

Extract test functions from the negative side.

Returns shape functions for (nn, np) combinations.
"""
function side_n(a::BilinearTestFaceSidePair)
    Side⁻(LazyMapOver((a.data[1], NullOperator(), a.data[3], NullOperator())))
end

"""
    side_p(a::BilinearTestFaceSidePair)

Extract test functions from the positive side.

Returns shape functions for (pn, pp) combinations.
"""
function side_p(a::BilinearTestFaceSidePair)
    Side⁺(LazyMapOver((NullOperator(), a.data[2], NullOperator(), a.data[4])))
end
