"""
    assemble_bilinear(a::Function, U, V)

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
) where {N}

    # Prepare sparse matrix allocation
    I = Int[]
    J = Int[]
    X = T[] # TODO : could be ComplexF64 or Dual

    # Compute
    assemble_bilinear!(I, J, X, a, U, V)

    nrows = get_ndofs(V)
    ncols = get_ndofs(U)
    return sparse(I, J, X, nrows, ncols)
end

function assemble_bilinear!(I, J, X, a, U, V)
    return_type_a = a(_null_operator(U), _null_operator(V))
    _assemble_bilinear!(I, J, X, a, U, V, return_type_a)
    return nothing
end

function _assemble_bilinear!(
    I::Vector{Int},
    J::Vector{Int},
    X::Vector,
    a::Function,
    U,
    V,
    integration::Integration,
)
    f(u, v) = get_function(get_integrand(a(u, v)))
    measure = get_measure(integration)
    assemble_bilinear!(I, J, X, f, measure, U, V)
    return nothing
end

function _assemble_bilinear!(
    I::Vector{Int},
    J::Vector{Int},
    X::Vector,
    a::Function,
    U,
    V,
    multiIntegration::MultiIntegration{N},
) where {N}
    for i in 1:N
        ival = Val(i)
        aᵢ(u, v) = a(u, v)[ival]
        _assemble_bilinear!(I, J, X, aᵢ, U, V, multiIntegration[ival])
    end
    nothing
end

"""
    assemble_bilinear!(
        I::Vector{Int},
        J::Vector{Int},
        X::Vector{T},
        f::Function,
        measure::Measure{<:CellDomain},
        U::TrialFESpace,
        V::TestFESpace,
    )

In-place version of [`assemble_bilinear`](@ref).
"""
function assemble_bilinear!(
    I::Vector{Int},
    J::Vector{Int},
    X::Vector,
    f::Function,
    measure::Measure{<:CellDomain},
    U::TrialFESpace,
    V::TestFESpace,
)
    # Alias
    quadrature = get_quadrature(measure)
    domain = get_domain(measure)

    # Loop over cells
    for (i, cellinfo) in enumerate(DomainIterator(domain))
        _λU, _λV = blockmap_bilinear_shape_functions(U, V, cellinfo)
        g1 = materialize(f(_λU, _λV), cellinfo)
        values = integrate_on_ref(g1, cellinfo, quadrature)
        _append_contribution!(X, I, J, U, V, values, cellinfo, domain)
    end

    return nothing
end

"""
    assemble_bilinear!(
        I::Vector{Int},
        J::Vector{Int},
        X::Vector{T},
        f::Function,
        measure::Measure{<:AbstractFaceDomain},
        U::TrialFESpace,
        V::TestFESpace,
    )

In-place version of [`assemble_bilinear`](@ref).
"""
function assemble_bilinear!(
    I::Vector{Int},
    J::Vector{Int},
    X::Vector,
    f::Function,
    measure::Measure{<:AbstractFaceDomain},
    U::TrialFESpace,
    V::TestFESpace,
)
    # Alias
    quadrature = get_quadrature(measure)
    domain = get_domain(measure)
    mesh = get_mesh(domain)
    c2n = connectivities_indices(mesh, :c2n)
    f2n = connectivities_indices(mesh, :f2n)
    f2c = connectivities_indices(mesh, :f2c)
    celltypes = cells(mesh)
    faceTypes = faces(mesh)

    # Loop over faces
    for kface in indices(domain)
        # Neighbor cell i
        i = f2c[kface][1]
        ctype_i = cellTypes[i]
        nnodes_i = Val(nnodes(ctype_i))
        _c2n_i = c2n[i, nnodes_i]
        cnodes_i = get_nodes(mesh, _c2n_i)
        cinfo_i = CellInfo(i, ctype_i, cnodes_i, _c2n_i)

        # Neighbor cell j
        j = f2c[kface][2]
        ctype_j = cellTypes[j]
        nnodes_j = Val(nnodes(ctype_j))
        _c2n_j = c2n[j, nnodes_j]
        cnodes_j = get_nodes(mesh, _c2n_j)
        cinfo_j = CellInfo(j, ctype_j, cnodes_j, _c2n_j)

        # Face info
        ftype = faceTypes[kface]
        n_fnodes = Val(nnodes(ftype))
        _f2n = f2n[kface, n_fnodes]
        fnodes = get_nodes(mesh, _f2n)
        error("TODO")
        # # Integrate from cell i "point of view"
        # finfo_ij = FaceInfo(cinfo_i, cinfo_j, ftype, fnodes, _f2n)
        # λU_i = get_shape_functions(U, shape(ctype_i))
        # λV_i = get_shape_functions(V, shape(ctype_i))

        # jdofs = get_dofs(U, icell)
        # idofs = get_dofs(V, icell)

        # for (jdof, λⱼ) in zip(jdofs, λU)
        #     for (idof, λᵢ) in zip(idofs, λV)
        #         _g = f(λⱼ, λᵢ)
        #         g = materialize(_g, cellinfo)

        #         push!(I, idof)
        #         push!(J, jdof)
        #         push!(X, integrate_on_ref(g, cellinfo, quadrature))
        #     end
        # end
    end

    return nothing
end

function assemble_bilinear!(
    I::Vector{Int},
    J::Vector{Int},
    X::Vector,
    f::Function,
    measure::Measure,
    U::AbstractMultiFESpace{N, <:Tuple{Vararg{TrialFESpace, N}}},
    V::AbstractMultiFESpace{N, <:Tuple{Vararg{TestFESpace, N}}},
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
            assemble_bilinear!(_I, _J, X, _f, measure, _U, _V)

            # Update global indices
            push!(I, get_mapping(V, i)[_I]...)
            push!(J, get_mapping(U, j)[_J]...)
        end
    end

    return nothing
end

"""
    assemble_linear(l::Function, V::Union{TestFESpace, AbstractMultiTestFESpace})

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
function assemble_linear(l::Function, V::Union{TestFESpace, AbstractMultiTestFESpace})
    b = zeros(get_ndofs(V)) # TODO : specify the eltype (Float64, Dual,...)
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
    # apply `l` on `NullOperator` to get the type
    # of the result of `l` and use it for dispatch
    # (`Integration` or `MultiIntegration` case).
    _assemble_linear!(b, l, V, l(_null_operator(V)))
    return nothing
end

"""
    _assemble_linear!(b, l, V, integration::Integration)
    _assemble_linear!(b, l, V, integration::MultiIntegration{N}) where {N}

These functions act as a function barrier in order to:
* get the function corresponding to the operand in the linear form
* reshape `b` internally to deal with cases when `V` is a `AbstractMultiTestFESpace`
* call `__assemble_linear!` to apply dispatch on the type of `measure` of the
  integration and improve type stability during the assembling loop.

## Dev note:
The case `integration::MultiIntegration{N}` is treated by looping over
each `Integration` contained in the `MultiIntegration`
"""
function _assemble_linear!(b, l, V, integration::Integration)
    f(v) = get_function(get_integrand(l(v)))
    measure = get_measure(integration)
    __assemble_linear!(_may_reshape_b(b, V), f, V, measure)
    return nothing
end

function _assemble_linear!(b, l, V, integration::MultiIntegration{N}) where {N}
    for i in 1:N
        ival = Val(i)
        lᵢ(v) = l(v)[ival]
        _assemble_linear!(b, lᵢ, V, integration[ival])
    end
    return nothing
end

"""
# Dev notes
Two levels of "LazyMapOver" because first we LazyMapOver the Tuple of argument of the linear form,
and the for each item of this Tuple we LazyMapOver the shape functions.
"""
function __assemble_linear!(b, f, V, measure::Measure)
    # Alias
    quadrature = get_quadrature(measure)
    domain = get_domain(measure)

    for elementInfo in DomainIterator(domain)
        # Materialize the operation to perform on the current element
        vₑ = blockmap_shape_functions(V, elementInfo)
        fᵥ = materialize(f(vₑ), elementInfo)
        values = _integrate_on_ref_element(fᵥ, elementInfo, quadrature)
        _update_b!(b, V, values, elementInfo, domain)
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

function _integrate_on_ref_element(f, elementInfo::CellInfo, quadrature)
    integrate_on_ref(f, elementInfo, quadrature)
end
function _integrate_on_ref_element(f, elementInfo::FaceInfo, quadrature)
    integrate_face_ref(f, elementInfo, quadrature)
end

"""
bilinear case
"""
function _append_contribution!(X, I, J, U, V, values, elementInfo::CellInfo, domain)
    icell = cellindex(elementInfo)
    nU = Val(get_ndofs(U, shape(celltype(elementInfo))))
    nV = Val(get_ndofs(V, shape(celltype(elementInfo))))
    jdofs = get_dofs(U, icell, nU)
    idofs = get_dofs(V, icell, nV)
    _idofs, _jdofs = _cartesian_product(idofs, jdofs)
    unwrapValues = _unwrap_cell_integrate(V, values)
    append!(I, _idofs)
    append!(J, _jdofs)
    for _v in unwrapValues
        append!(X, _v)
    end
    return nothing
end

function _update_b!(b, V, values, elementInfo::CellInfo, domain)
    idofs = get_dofs(V, cellindex(elementInfo))
    unwrapValues = _unwrap_cell_integrate(V, values)
    _update_b!(b, idofs, unwrapValues)
end
_unwrap_cell_integrate(::TestFESpace, a) = map(unwrap, unwrap(a))
_unwrap_cell_integrate(::AbstractMultiTestFESpace, a) = map(unwrap, unwrap(a))

function _update_b!(b, V, values, elementInfo::FaceInfo, domain)
    # First, we get the values from the integration on the positive/negative side
    # Then, if the face has two side, we seek the values from the opposite side
    unwrapValues = _unwrap_face_integrate(V, values)
    values_i = map(identity, map(identity, map(first, unwrapValues)))
    idofs = get_dofs(V, cellindex(get_cellinfo_n(elementInfo)))
    _update_b!(b, idofs, values_i)
    if (domain isa InteriorFaceDomain) ||
       (domain isa BoundaryFaceDomain{<:AbstractMesh, <:PeriodicBCType})
        values_j = map(identity, map(identity, map(last, unwrapValues)))
        jdofs = get_dofs(V, cellindex(get_cellinfo_p(elementInfo)))
        _update_b!(b, jdofs, values_j)
    end
end
function _unwrap_face_integrate(::Union{TestFESpace, AbstractMultiTestFESpace}, a)
    return unwrap(unwrap(unwrap(a)))
end

function _update_b!(
    b::Tuple{Vararg{Any, N}},
    idofs::Tuple{Vararg{Any, N}},
    intvals::Tuple{Vararg{Any, N}},
) where {N}
    map(_update_b!, b, idofs, intvals)
    nothing
end
function _update_b!(b::AbstractVector, idofs, intvals::Tuple{Vararg{Tuple, N}}) where {N}
    map(x -> _update_b!(b, idofs, x), intvals)
    nothing
end
function _update_b!(b::AbstractVector, dofs, vals)
    for (i, val) in zip(dofs, vals)
        b[i] += val
    end
    nothing
end

_update_b!(b::AbstractVector, dofs, vals::NullOperator) = nothing

"""
Count the (maximum) number of elements in the matrix corresponding to the bilinear assembly of U, V
on a cell domain, where `U` and `V` are `TrialFESpace` and `TestFESpace`
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

"""
Count the (maximum) number of elements in the matrix corresponding to the bilinear assembly of U, V
on a cell domain, where `U` and `V` are `AbstractMultiFESpace`
"""
function _count_n_elts(
    U::AbstractMultiFESpace{N, Tu},
    V::AbstractMultiFESpace{N, Tv},
    domain::CellDomain{M, IND},
) where {M, IND, N, Tu <: Tuple{Vararg{TrialFESpace}}, Tv <: Tuple{Vararg{TestFESpace}}}
    n = 0
    for _U in U
        for _V in V
            n += _count_n_elts(_U, _V, domain)
        end
    end
    return n
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

function blockmap_bilinear_shape_functions(
    U::AbstractFESpace,
    V::AbstractFESpace,
    cellinfo::AbstractCellInfo,
)
    cshape = shape(celltype(cellinfo))
    λU = get_shape_functions(U, cshape)
    λV = get_shape_functions(V, cshape)
    _blockmap_bilinear(λU, λV)
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

Return an array of the integral evaluated over each cell (or face). To get the sum over the whole
domain, simply apply the `sum` function.
"""
function compute(integration::Integration)
    measure = get_measure(integration)
    domain = get_domain(measure)
    mesh = get_mesh(domain)
    fs = FunctionSpace(:Lagrange, 0)
    V = TestFESpace(fs, mesh)
    return assemble_linear(v -> integration, V)
end

"""
    AbstractFaceSidePair <: AbstractLazy

# Interface:
* `side_n(a::AbstractFaceSidePair)`
* `side_p(a::AbstractFaceSidePair)`
"""
abstract type AbstractFaceSidePair <: AbstractLazy end

struct FaceSidePair{T1, T2}
    data_n::T1
    data_p::T2
end
side_n(a::FaceSidePair) = MapOver((a.data_n, NullOperator()))
side_p(a::FaceSidePair) = MapOver((NullOperator(), a.data_p))

function LazyOperators.materialize(a::FaceSidePair, cPoint::CellPoint)
    _a = (materialize(a.data_n, cPoint), materialize(a.data_p, cPoint))
    return MapOver(_a)
end

function LazyOperators.materialize(a::FaceSidePair, side::Side⁻{Nothing, <:Tuple{FaceInfo}})
    return FaceSidePair(materialize(a.data_n, side), NullOperator())
end

function LazyOperators.materialize(a::FaceSidePair, side::Side⁺{Nothing, <:Tuple{FaceInfo}})
    return FaceSidePair(NullOperator(), materialize(a.data_p, side))
end
