"""
    assemble_dirichlet_vector(
        U,
        V,
        mesh::AbstractMesh,
        t::Number = 0.0;
        T = Float64,
        dt_derivative_order::Int = 0,
    )
    assemble_dirichlet_sparse_vector(
        U,
        V,
        mesh::AbstractMesh,
        t::Number = 0.0;
        T = Float64,
        dt_derivative_order::Int = 0,
    )

Assemble a vector of zeros dofs except on boundary dofs where they take the Dirichlet values.

To obtain a sparse vector, use [`assemble_dirichlet_sparse_vector`](@ref).
"""
function assemble_dirichlet_vector(U, V, mesh::AbstractMesh, args...; kwargs...)
    sparse_values = assemble_dirichlet_sparse_vector(U, V, mesh, args...; kwargs...)
    return dense_dirichlet_vector(sparse_values, get_bcube_backend(mesh))
end

"""
    dense_dirichlet_vector(sparse_values, backend)

Materialize the assembled sparse Dirichlet values as a dense vector.

The default method returns a Julia `Vector`. Backends can specialize the conversion
for their own storage.
"""
dense_dirichlet_vector(sparse_values, ::AbstractBcubeBackend) = Vector(sparse_values)

"""
    assemble_dirichlet_sparse_vector(
        U,
        V,
        mesh::AbstractMesh,
        t::Number = 0.0;
        T = Float64,
        dt_derivative_order::Int = 0,
    )

Assemble a sparse vector of `T` with prescribed boundary values at time `t`.

Only derivative order zero is supported for now.
"""
function assemble_dirichlet_sparse_vector(
    U,
    V,
    mesh::AbstractMesh,
    t::Number = 0.0;
    T = Float64,
    dt_derivative_order::Int = 0,
)
    @assert dt_derivative_order == 0 "`dt_derivative_order` > 0 not implemented yet"
    boundaries = prepare_dirichlet_boundaries(U, mesh)
    I_diri, position_by_dof = dirichlet_data(U, V, mesh, boundaries)
    backend = get_bcube_backend(mesh)
    b = allocate_dirichlet_sparse_vector(U, I_diri, T)
    apply_dirichlet_values!(b, I_diri, position_by_dof, U, V, boundaries, backend, t)
    return b
end

"""
    allocate_dirichlet_sparse_vector(U, I_diri, T)

Allocate sparse global dofs with explicit zero entries at every index in `I_diri`.
"""
function allocate_dirichlet_sparse_vector(U, I_diri, T)
    values = fill!(similar(I_diri, T), zero(T))
    return sparsevec(I_diri, values, get_ndofs(U))
end

"""
    prepare_dirichlet_boundaries(U, mesh)

Prepare one tuple of boundary domains per field, preserving field and tag order.

For a single field the result is `((domain_tag1, domain_tag2, ...),)`. Domains are
ordered like `get_dirichlet_boundary_tags(U)`.
"""
function prepare_dirichlet_boundaries(U::TrialFESpace, mesh)
    return (prepare_dirichlet_boundary_domains(U, mesh),)
end

"""
    prepare_dirichlet_boundaries(U::AbstractMultiFESpace, mesh)

Prepare the domain tuples of the mixed fields, in their original order.

Each inner tuple follows the boundary-tag order of its trial space, as in the
single-field method. Trial/test spaces and mappings are passed separately to helpers.
"""
function prepare_dirichlet_boundaries(U::AbstractMultiFESpace, mesh)
    return map(get_fespace(U)) do trial
        prepare_dirichlet_boundary_domains(trial, mesh)
    end
end

"""
    prepare_dirichlet_boundary_domains(U, mesh)

Build the domains of one trial field in its Dirichlet-tag order.

The result must remain aligned with `get_dirichlet_boundary_tags(U)` until evaluation finishes.
"""
function prepare_dirichlet_boundary_domains(U, mesh)
    return map(Tuple(get_dirichlet_boundary_tags(U))) do tag
        BoundaryFaceDomain(parent(mesh), String(tag))
    end
end

"""
    dirichlet_data(U, V, mesh, boundaries)

Return the Dirichlet indices and their inverse using the backend of `mesh`.
Pass the domains prepared by [`prepare_dirichlet_boundaries`](@ref).
"""
function dirichlet_data(U, V, mesh, boundaries)
    return dirichlet_data(U, V, get_bcube_backend(mesh), boundaries)
end

"""
    dirichlet_data(U, V, backend::AbstractBcubeBackend, boundaries)

Return `(I_diri, position_by_dof)` for the matching trial/test spaces.

`I_diri` contains unique global Dirichlet dofs in encounter order, after mixed-space
mapping. The dense inverse has length `get_ndofs(U)`: `position_by_dof[I_diri[k]] == k`,
with zero outside Dirichlet. Both arrays are rebuilt on each call, without evaluating
boundary functions or changing FE spaces. Pass prepared `boundaries` to reuse the same
domains during evaluation.

Encounter order follows the fields, their boundary tags, the subdomains within each
boundary, and the faces within each subdomain. After the mixed mapping is applied,
`unique` keeps the first occurrence of each global dof.
"""
function dirichlet_data(U, V, backend::AbstractBcubeBackend, boundaries)
    occurrences = collect_dirichlet_indices(U, V, backend, boundaries)
    ndofs = get_ndofs(U)
    I_diri = unique_dirichlet_indices(occurrences, ndofs, backend)

    # Zero marks unconstrained dofs; nonzero entries address the compact buffer.
    position_by_dof = fill!(similar(I_diri, Int, ndofs), 0)
    position_by_dof[I_diri] .= eachindex(I_diri)
    return I_diri, position_by_dof
end

"""
    unique_dirichlet_indices(occurrences, ndofs, backend)

Return unique global dofs in encounter order. Backends can use the total number
of dofs `ndofs` to index temporary storage by global dof.
"""
unique_dirichlet_indices(occurrences, ndofs, ::AbstractBcubeBackend) = unique(occurrences)

"""
    dirichlet_face_offsets(counts, backend)

Return the exclusive offsets and total number of occurrences for per-face counts.
"""
function dirichlet_face_offsets(counts, ::AbstractBcubeBackend)
    offsets = isempty(counts) ? similar(counts, 0) : cumsum_exclusive(counts)
    return offsets, sum(counts)
end

"""
    collect_dirichlet_indices(U::TrialFESpace, V::TestFESpace, backend, boundaries)

Allocate and return boundary dof occurrences using an identity field mapping.

The domain-only context follows [`prepare_dirichlet_boundaries`](@ref).
"""
function collect_dirichlet_indices(U::TrialFESpace, V::TestFESpace, backend, boundaries)
    return collect_dirichlet_indices((V,), (1:get_ndofs(U),), backend, boundaries)
end

"""
    collect_dirichlet_indices(U::AbstractMultiFESpace, V::AbstractMultiFESpace, backend, boundaries)

Collect mixed-field occurrences with the original test spaces and global mappings.

Field order in `boundaries` must match the mixed spaces; no field metadata is stored
in the context itself. Duplicate dofs are retained for subsequent `unique`.
"""
function collect_dirichlet_indices(
    U::AbstractMultiFESpace,
    V::AbstractMultiFESpace,
    backend,
    boundaries,
)
    return collect_dirichlet_indices(get_fespace(V), get_mapping(U), backend, boundaries)
end

"""
    collect_dirichlet_indices(test_spaces::Tuple, mappings::Tuple, backend, boundaries)

Allocate and fill global occurrences on the supplied domain tuples.

Count occurrences per face with a first foreach_element traversal, then compute
exclusive offsets and allocate exact storage. A second traversal fills disjoint
face segments. Concatenation preserves field, boundary and face order.

Within each boundary, faces are visited in subdomain order. The explicit `backend`
is available for dispatch on the collection method. Each `foreach_element` call
uses the backend of its domain.
"""
function collect_dirichlet_indices(test_spaces::Tuple, mappings::Tuple, backend, boundaries)
    occurrences_by_field = map(test_spaces, mappings, boundaries) do test, mapping, domains
        space = get_function_space(test)
        dhl = get_dhl(test)
        component_count = get_ncomponents(dhl)

        result = map(domains) do domain
            # Count each face independently using storage from its boundary indices.
            n_diri_dofs_by_face = similar(indices(domain), Int, get_nelements(domain))
            foreach_element(domain) do face, face_index, _
                cell = get_cellinfo_n(face)
                local_dofs =
                    idof_by_face_with_bounds(space, shape(celltype(cell)))[get_cell_side_n(
                        face,
                    )]
                n_diri_dofs_by_face[face_index] = length(local_dofs) * component_count
            end

            # Prefix offsets give every face its own contiguous segment.
            offsets, occurrence_count = dirichlet_face_offsets(n_diri_dofs_by_face, backend)
            occurrences = similar(n_diri_dofs_by_face, Int, occurrence_count)
            foreach_element(domain) do face, face_index, _
                cell = get_cellinfo_n(face)
                local_dofs =
                    idof_by_face_with_bounds(space, shape(celltype(cell)))[get_cell_side_n(
                        face,
                    )]
                position = offsets[face_index] + 1
                for local_dof in local_dofs, component in 1:component_count
                    occurrences[position] =
                        mapping[get_dof(dhl, cellindex(cell), component, local_dof)]
                    position += 1
                end
            end
            return occurrences
        end
        return result
    end

    # Concatenation preserves field, boundary and face order, including duplicates.
    first_test_dof_handler = get_dhl(first(test_spaces))
    empty_occurrences = similar(first_test_dof_handler.iglob, Int, 0)
    return reduce(vcat, Iterators.flatten(occurrences_by_field); init = empty_occurrences)
end

"""
    allocate_dirichlet_values(I_diri, T)

Allocate zeroed compact boundary values using I_diri as the storage prototype.

The returned dense buffer has element type `T` and length `length(I_diri)`. Entry `k`
corresponds to global dof `I_diri[k]`, independently of sparse storage order.
"""
function allocate_dirichlet_values(I_diri, T)
    return fill!(similar(I_diri, T), zero(T))
end

"""
    set_dirichlet_value!(values, index, value, backend)

Write one evaluated boundary occurrence; accelerated backends specialize the concurrent write.
"""
function set_dirichlet_value!(values, index, value, ::AbstractBcubeBackend)
    values[index] = value
end

"""
    evaluate_dirichlet_values!(V_diri, position_by_dof, U, V, boundaries, backend, t)

Evaluate boundary values at time `t` into the preallocated compact buffer `V_diri`
using prepared domains.

Reuse [`foreach_dirichlet_value!`](@ref) for materialization. Global dofs address
`V_diri` through the dense inverse. This method does not mutate vector structure.
"""
function evaluate_dirichlet_values!(V_diri, position_by_dof, U, V, boundaries, backend, t)
    callback!(array, global_dof, component, evaluated) = set_dirichlet_value!(
        array,
        position_by_dof[global_dof],
        evaluated[component],
        backend,
    )
    foreach_dirichlet_value!((V_diri,), (callback!,), U, V, boundaries, t)
end

"""
    prepare_dirichlet_vector!(b::AbstractVector, I_diri, position_by_dof, backend)

Prepare dense vector writes; return the unique global indices directly, without altering `b`.
"""
function prepare_dirichlet_vector!(b::AbstractVector, I_diri, position_by_dof, ::AbstractBcubeBackend)
    return I_diri
end

"""
    prepare_dirichlet_vector!(b::AbstractSparseVector, I_diri, position_by_dof, backend)

Merge absent sparse entries as explicit zeros, preserving existing values and `b`; return final storage positions.
"""
function prepare_dirichlet_vector!(
    b::AbstractSparseVector,
    I_diri,
    position_by_dof,
    ::AbstractBcubeBackend,
)
    positions = zeros(Int, length(I_diri))
    fill_dirichlet_vector_positions!(positions, b, position_by_dof)
    missing_positions = findall(iszero, positions)

    # Supply only absent dofs, so old values (including signed zeros) are untouched.
    # sparsevec accepts encounter order and canonicalizes its own stored indices.
    if !isempty(missing_positions)
        prepared = sparsevec(
            vcat(SparseArrays.nonzeroinds(b), I_diri[missing_positions]),
            vcat(nonzeros(b), zeros(eltype(b), length(missing_positions))),
            length(b),
        )
        copyto!(b, prepared)
        fill_dirichlet_vector_positions!(positions, b, position_by_dof)
    end
    return positions
end

"""
    fill_dirichlet_vector_positions!(positions, b::AbstractSparseVector, position_by_dof)

Fill compact `positions` by traversing the stored entries of `b` once.

For every constrained global dof, assign its stored position to the slot given by
`position_by_dof`. Missing entries retain their initial zero sentinel. This helper
changes only `positions`, never `b`.
"""
function fill_dirichlet_vector_positions!(
    positions,
    b::AbstractSparseVector,
    position_by_dof,
)
    for (stored_position, global_dof) in enumerate(SparseArrays.nonzeroinds(b))
        compact_position = position_by_dof[global_dof]
        if compact_position != 0
            positions[compact_position] = stored_position
        end
    end
end

"""
    dirichlet_vector_storage(b::AbstractVector)

Return writable dense entries or the values array of sparse vector storage.
"""
dirichlet_vector_storage(b::AbstractVector) = b
dirichlet_vector_storage(b::AbstractSparseVector) = nonzeros(b)

"""
    set_dirichlet_values!(b, positions, V_diri, backend)

Assign `V_diri[k]` to the prepared storage position `positions[k]` in `b`.

The positions must follow the compact Dirichlet order and be unique. Only stored
values are written; no structure changes or atomic operations are needed in this step.
"""
function set_dirichlet_values!(b, positions, V_diri, ::AbstractBcubeBackend)
    storage = dirichlet_vector_storage(b)
    for k in eachindex(V_diri)
        storage[positions[k]] = V_diri[k]
    end
end

"""
    apply_dirichlet_to_vector!(
        b::AbstractVector{<:Number},
        U,
        V,
        mesh::AbstractMesh,
        t::Number = 0.0;
        dt_derivative_order::Int = 0,
    )

Apply Dirichlet condition on vector `b`.

Dirichlet values are applied on dofs lying on a Dirichlet boundary.

Only `dt_derivative_order == 0` is supported for now. Values are evaluated compactly before
storage preparation and assignment; all other dofs and the supplied object are preserved.
"""
function apply_dirichlet_to_vector!(
    b::AbstractVector{<:Number},
    U,
    V,
    mesh::AbstractMesh,
    t::Number = 0.0;
    dt_derivative_order::Int = 0,
)
    @assert dt_derivative_order == 0 "`dt_derivative_order` > 0 not implemented yet"
    boundaries = prepare_dirichlet_boundaries(U, mesh)
    I_diri, position_by_dof = dirichlet_data(U, V, mesh, boundaries)
    backend = get_bcube_backend(mesh)
    apply_dirichlet_values!(b, I_diri, position_by_dof, U, V, boundaries, backend, t)
end

"""
    apply_dirichlet_values!(b, I_diri, position_by_dof, U, V, boundaries, backend, t)

Evaluate compact boundary values, prepare the vector storage, then assign unique positions.

`boundaries` is a tuple of domain tuples, one per field in `U` and `V`. For example,
`((domain_xmin, domain_ymin), (domain_xmax,))` describes two mixed fields with two
and one constrained boundaries. Each inner order matches that field's
`get_dirichlet_boundary_tags`. Spaces, tags and mappings are read from `U` and `V`,
not embedded in the context. The same domains serve topology and value evaluation.
"""
function apply_dirichlet_values!(b, I_diri, position_by_dof, U, V, boundaries, backend, t)

    # Shared face occurrences address the compact buffer through the dense inverse.
    V_diri = allocate_dirichlet_values(I_diri, eltype(b))
    evaluate_dirichlet_values!(V_diri, position_by_dof, U, V, boundaries, backend, t)

    # Complete all structural changes before the final unique-position assignment.
    positions = prepare_dirichlet_vector!(b, I_diri, position_by_dof, backend)
    set_dirichlet_values!(b, positions, V_diri, backend)
end

"""
    apply_homogeneous_dirichlet_to_vector!(b::AbstractVector{<:Number}, U, V, mesh::AbstractMesh)

Apply homogeneous Dirichlet condition on vector `b`

Dirichlet values are applied on dofs lying on a Dirichlet boundary.
"""
function apply_homogeneous_dirichlet_to_vector!(
    b::AbstractVector{<:Number},
    U,
    V,
    mesh::AbstractMesh,
)
    boundaries = prepare_dirichlet_boundaries(U, mesh)
    I_diri, position_by_dof = dirichlet_data(U, V, mesh, boundaries)
    set_homogeneous_dirichlet_values!(b, I_diri, position_by_dof, get_bcube_backend(mesh))
end

"""
    set_homogeneous_dirichlet_values!(b, I_diri, position_by_dof, backend)

Zero the supplied vector at prepared Dirichlet dofs without evaluating boundary values.

Reuse the common storage preparation and compact assignment with a zeroed value
buffer. This shared step also serves the homogeneous matrix/vector system wrapper.
"""
function set_homogeneous_dirichlet_values!(b, I_diri, position_by_dof, backend)
    V_diri = allocate_dirichlet_values(I_diri, eltype(b))
    positions = prepare_dirichlet_vector!(b, I_diri, position_by_dof, backend)
    set_dirichlet_values!(b, positions, V_diri, backend)
end

"""
    prepare_dirichlet_matrix!(matrix::SparseArrays.AbstractSparseMatrixCSC, indices, backend)

Insert only absent CSC diagonals before parallel writes; return their definitive stored positions, preserving `matrix`.
"""
function prepare_dirichlet_matrix!(
    matrix::SparseArrays.AbstractSparseMatrixCSC,
    indices,
    ::AbstractBcubeBackend,
)
    diagonal_positions = zeros(Int, length(indices))
    fill_dirichlet_diagonal_positions!(diagonal_positions, matrix, indices)
    missing_positions = findall(iszero, diagonal_positions)

    # Reconstruct only when a diagonal is absent, then locate the final positions.
    if !isempty(missing_positions)
        missing_diagonals = indices[missing_positions]
        rows, columns, values = findnz(matrix)
        prepared = sparse(
            vcat(rows, missing_diagonals),
            vcat(columns, missing_diagonals),
            vcat(values, zeros(eltype(matrix), length(missing_positions))),
            size(matrix)...,
        )
        copyto!(matrix, prepared)
        fill_dirichlet_diagonal_positions!(diagonal_positions, matrix, indices)
    end
    return diagonal_positions
end

"""
    fill_dirichlet_diagonal_positions!(
        positions,
        matrix::SparseArrays.AbstractSparseMatrixCSC,
        I_diri,
    )

Fill `positions` with existing CSC diagonal locations, in `I_diri` order.

Traverse each constrained column until its diagonal or a larger row is found.
Absent diagonals retain their initial zero sentinel. The matrix is unchanged;
return `nothing`.
"""
function fill_dirichlet_diagonal_positions!(
    positions,
    matrix::SparseArrays.AbstractSparseMatrixCSC,
    I_diri,
)
    rows = rowvals(matrix)
    for (compact_position, column) in enumerate(I_diri)
        for stored_position in nzrange(matrix, column)
            row = rows[stored_position]
            if row == column
                positions[compact_position] = stored_position
                break
            elseif row > column
                break
            end
        end
    end
end

"""
    dirichlet_mask(matrix, indices)

Build a separate global Dirichlet mask for matrix cancellation.
"""
function dirichlet_mask(matrix, indices)
    mask = similar(nonzeros(matrix), Bool, size(matrix, 1))
    fill!(mask, false)
    mask[indices] .= true
    return mask
end

"""
    zero_dirichlet_entries!(matrix::SparseArrays.AbstractSparseMatrixCSC, mask, cancel_columns::Bool, backend::BcubeBackendCPUSerial)

Zero stored CSC coefficients directly by column, preserving its sparsity structure.
"""
function zero_dirichlet_entries!(
    matrix::SparseArrays.AbstractSparseMatrixCSC,
    mask,
    cancel_columns::Bool,
    ::BcubeBackendCPUSerial,
)
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    for column in axes(matrix, 2), position in nzrange(matrix, column)
        if mask[rows[position]] || (cancel_columns && mask[column])
            values[position] = zero(eltype(matrix))
        end
    end
end

"""
    set_dirichlet_diagonal!(matrix::SparseArrays.AbstractSparseMatrixCSC, positions, value, backend::BcubeBackendCPUSerial)

Restore CSC diagonal values without inserting any entries.
"""
function set_dirichlet_diagonal!(
    matrix::SparseArrays.AbstractSparseMatrixCSC,
    positions,
    value,
    ::BcubeBackendCPUSerial,
)
    nonzeros(matrix)[positions] .= value
end

"""
    apply_dirichlet_to_matrix!(
        matrix::AbstractSparseMatrix,
        U,
        V,
        mesh::AbstractMesh;
        diag_value::Number = 1.0,
    )

Apply a "dirichlet" condition on the input matrix.

The columns corresponding to each Dirichlet dof are NOT canceled. If you want them to be canceled,
use `apply_homogeneous_dirichlet_to_matrix!`.

Mutate the square sparse matrix in place by zeroing its constrained rows and
setting their diagonal entries to `diag_value`.
"""
function apply_dirichlet_to_matrix!(
    matrix::AbstractSparseMatrix,
    U,
    V,
    mesh::AbstractMesh;
    diag_value::Number = 1.0,
)
    apply_dirichlet_to_matrix!((matrix,), U, V, mesh; diag_values = (diag_value,))
end

"""
    apply_dirichlet_to_matrix!(
        matrices::Tuple{Vararg{AbstractSparseMatrix, N}},
        U,
        V,
        mesh::AbstractMesh;
        diag_values::Tuple{Vararg{Number, N}} = ntuple(_ -> 1.0, N),
    ) where {N}

Same as above, but with a Tuple of matrix

Each square sparse matrix is mutated in place with the corresponding entry in
`diag_values`, in tuple order.
"""
function apply_dirichlet_to_matrix!(
    matrices::Tuple{Vararg{AbstractSparseMatrix, N}},
    U,
    V,
    mesh::AbstractMesh;
    diag_values::Tuple{Vararg{Number, N}} = ntuple(_ -> 1.0, N),
) where {N}
    apply_dirichlet_matrices!(matrices, U, V, mesh, diag_values, false)
end

"""
    apply_homogeneous_dirichlet_to_matrix!(
        matrix::AbstractSparseMatrix,
        U,
        V,
        mesh::AbstractMesh;
        diag_value::Number = 1.0,
    )

Apply a homogeneous "dirichlet" condition on the input matrix.

The rows and columns corresponding to each Dirichlet dof are canceled (with a 1 on the diag term by default).
If you don't want the cols to be canceled, use `apply_dirichlet_to_matrix!`.

Mutate the square sparse matrix in place and set each constrained diagonal entry
to `diag_value` after cancellation.
"""
function apply_homogeneous_dirichlet_to_matrix!(
    matrix::AbstractSparseMatrix,
    U,
    V,
    mesh::AbstractMesh;
    diag_value::Number = 1.0,
)
    apply_homogeneous_dirichlet_to_matrix!(
        (matrix,),
        U,
        V,
        mesh;
        diag_values = (diag_value,),
    )
end

"""
    apply_homogeneous_dirichlet_to_matrix!(
        matrices::Tuple{Vararg{AbstractSparseMatrix, N}},
        U,
        V,
        mesh::AbstractMesh;
        diag_values::Tuple{Vararg{Number, N}} = ntuple(_ -> 1.0, N),
    ) where {N}

Same as above, but with a Tuple of matrix.

Each square sparse matrix is mutated in place with the corresponding entry in
`diag_values`, in tuple order.
"""
function apply_homogeneous_dirichlet_to_matrix!(
    matrices::Tuple{Vararg{AbstractSparseMatrix, N}},
    U,
    V,
    mesh::AbstractMesh;
    diag_values::Tuple{Vararg{Number, N}} = ntuple(_ -> 1.0, N),
) where {N}
    apply_dirichlet_matrices!(matrices, U, V, mesh, diag_values, true)
end

"""
    apply_dirichlet_matrices!(matrices, U, V, mesh, diag_values, cancel_columns::Bool)

Orchestrate topology, storage preparation, cancellation and diagonal assignment for all matrices.
"""
function apply_dirichlet_matrices!(matrices, U, V, mesh, diag_values, cancel_columns::Bool)
    boundaries = prepare_dirichlet_boundaries(U, mesh)
    indices, _ = dirichlet_data(U, V, mesh, boundaries)
    apply_dirichlet_matrices!(
        matrices,
        indices,
        get_bcube_backend(mesh),
        diag_values,
        cancel_columns,
    )
end

"""
    apply_dirichlet_matrices!(matrices, I_diri, backend, diag_values, cancel_columns::Bool)

Apply CSC constraints using topology already prepared by the caller.

Each matrix keeps its own definitive diagonal positions. Cancellation and diagonal
restoration operate only on stored values, after all structural preparation.
"""
function apply_dirichlet_matrices!(
    matrices,
    I_diri,
    backend,
    diag_values,
    cancel_columns::Bool,
)
    @assert all(matrix -> size(matrix, 1) == size(matrix, 2), matrices) "Dirichlet matrices must be square"

    for (matrix, value) in zip(matrices, diag_values)
        # Diagonal positions belong to this matrix's final, prepared structure.
        positions = prepare_dirichlet_matrix!(matrix, I_diri, backend)
        mask = dirichlet_mask(matrix, I_diri)

        # Cancel stored entries first, then restore the prescribed diagonals.
        zero_dirichlet_entries!(matrix, mask, cancel_columns, backend)
        set_dirichlet_diagonal!(matrix, positions, value, backend)
    end
end

"""
    apply_homogeneous_dirichlet!(
        A::AbstractSparseMatrix{<:Number},
        b::AbstractVector{<:Number},
        U,
        V,
        mesh::AbstractMesh,
    )

Apply homogeneous Dirichlet condition on `A` and `b` assuming the linear system Ax=b.

Dirichlet values are applied on dofs lying on a Dirichlet boundary.

Mutate the square sparse matrix `A` and the vector `b` in place. Zero the constrained
rows and columns of `A`, set their diagonals to one, and zero the corresponding entries of `b`.
"""
function apply_homogeneous_dirichlet!(
    A::AbstractSparseMatrix{<:Number},
    b::AbstractVector{<:Number},
    U,
    V,
    mesh::AbstractMesh,
)
    boundaries = prepare_dirichlet_boundaries(U, mesh)
    I_diri, position_by_dof = dirichlet_data(U, V, mesh, boundaries)
    backend = get_bcube_backend(mesh)
    apply_dirichlet_matrices!((A,), I_diri, backend, (1.0,), true)
    set_homogeneous_dirichlet_values!(b, I_diri, position_by_dof, backend)
end

"""
    foreach_dirichlet_value!(arrays, callbacks, U::TrialFESpace, V::TestFESpace,
                            boundaries::Tuple, t)

Evaluate single-field boundary values at time `t` with an identity mapping.
Apply `callbacks` in place to `arrays`.

The outer tuple has one entry, as documented by [`prepare_dirichlet_boundaries`](@ref).
"""
function foreach_dirichlet_value!(
    arrays,
    callbacks,
    U::TrialFESpace,
    V::TestFESpace,
    boundaries::Tuple,
    t,
)
    foreach_dirichlet_value!(arrays, callbacks, U, V, 1:get_ndofs(U), only(boundaries), t)
end

"""
    foreach_dirichlet_value!(arrays, callbacks, U::AbstractMultiFESpace,
                            V::AbstractMultiFESpace, boundaries::Tuple, t)

Evaluate mixed fields using their domain tuples and original global mappings.

Version for MultiFESpace

Field and tag orders follow [`prepare_dirichlet_boundaries`](@ref). Materialization
remains shared with the single-field traversal.
"""
function foreach_dirichlet_value!(
    arrays,
    callbacks,
    U::AbstractMultiFESpace,
    V::AbstractMultiFESpace,
    boundaries::Tuple,
    t,
)
    for (trial, test, mapping, domains) in zip(U, V, get_mapping(U), boundaries)
        foreach_dirichlet_value!(arrays, callbacks, trial, test, mapping, domains, t)
    end
end

"""
    foreach_dirichlet_value!(arrays, callbacks, U, V, mapping, domains, t)

Evaluate each existing domain using the matching trial-space tag at time `t`.

`mapping` is the dof mapping (if SingleFESpace, mapping = Id; otherwise mapping = get_mapping(...))

# Warning
We assume same FESpace for U and V

# Dev notes
The original tag order is zipped with `domains`; no tags are stored or reconstructed
in the domain context. The explicit mapping and test space provide global dofs and
FE data.
"""
function foreach_dirichlet_value!(arrays, callbacks, U, V, mapping, domains, t)
    space = get_function_space(V)
    dhl = get_dhl(V)
    for (tag, domain) in zip(get_dirichlet_boundary_tags(U), domains)
        time_function = get_dirichlet_values(U, tag)(t)
        foreach_element(domain) do face, _, _
            apply_dirichlet_on_face!(
                arrays,
                callbacks,
                face,
                space,
                dhl,
                mapping,
                time_function,
            )
        end
    end
end

"""
    apply_dirichlet_on_face!(
        arrays::Tuple{Vararg{AbstractVecOrMat, N}},
        callbacks::NTuple{N, Function},
        fInfo::FaceInfo,
        fs_V::AbstractFunctionSpace,
        dhl_V::DofHandler,
        m::AbstractVector{<:Integer},
        f_t_lazy::AbstractLazy,
    ) where {N}

Apply dirichlet condition on a face given by `fInfo`.

`m` is the dof mapping to obtain global dof id (in case of MultiFESpace)
`f_t_lazy` is the AbstractLazy representing the Dirichlet value at time t
`callbacks` is the Tuple of functions of (array, idof_glo, icomp, values) that
must be applied to each array of `arrays`

# Dev notes
Materialize on the negative-side cell, then at the face dof coordinates of `fs_V`.
"""
function apply_dirichlet_on_face!(
    arrays::Tuple{Vararg{AbstractVecOrMat, N}},
    callbacks::NTuple{N, Function},
    fInfo::FaceInfo,
    fs_V::AbstractFunctionSpace,
    dhl_V::DofHandler,
    m::AbstractVector{<:Integer},
    f_t_lazy::AbstractLazy,
) where {N}
    # Alias
    sizeV = get_ncomponents(dhl_V)

    # Get cell info on the negative side of the face
    cInfo = get_cellinfo_n(fInfo)
    icell = cellindex(cInfo)
    ctype = celltype(cInfo)

    # Get face info
    kside = get_cell_side_n(fInfo)

    # Shape function space and reference coordinates on the face
    cshape = shape(ctype)
    ξcell = get_coords(fs_V, cshape) # ref coordinates of the FunctionSpace in the cell

    # local indices of dofs lying on the face (assuming scalar FE)
    idofs_loc = idof_by_face_with_bounds(fs_V, cshape)[kside]

    # Materialize the AbstractLazy on the cell info
    # This gives us a function that can be evaluated at CellPoint
    f_materialized = materialize(f_t_lazy, cInfo)

    # Loop over the dofs concerned by the Dirichlet condition
    for idof_loc in idofs_loc
        ξ = ξcell[idof_loc]

        # Create a CellPoint at the reference coordinate ξ
        cPoint = CellPoint(ξ, cInfo, ReferenceDomain())

        # Evaluate the materialized lazy at the cell point
        values = materialize(f_materialized, cPoint)

        # Loop over components
        for icomp in 1:sizeV
            # Absolute number of the dof for this component
            idof_glo = m[get_dof(dhl_V, icell, icomp, idof_loc)]

            # Apply condition on each array according to the corresponding callback
            for (array, callback!) in zip(arrays, callbacks)
                callback!(array, idof_glo, icomp, values)
            end
        end
    end
end
