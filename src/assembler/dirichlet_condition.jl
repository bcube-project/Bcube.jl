"""
    assemble_dirichlet_vector(
        U,
        V,
        mesh::AbstractMesh,
        t::Number = 0.0;
        dt_derivative_order::Int = 0,
    )
    assemble_dirichlet_sparse_vector(
        U,
        V,
        mesh::AbstractMesh,
        t::Number = 0.0;
        dt_derivative_order::Int = 0,
    )

Assemble a vector of zeros dofs except on boundary dofs where they take the Dirichlet values.

To obtain a sparse vector, use assemble_dirichlet_sparse_vector.
"""
function assemble_dirichlet_vector(args...; kwargs...)
    return Vector(assemble_dirichlet_sparse_vector(args...; kwargs...))
end

function assemble_dirichlet_sparse_vector(
    U,
    V,
    mesh::AbstractMesh,
    t::Number = 0.0;
    dt_derivative_order::Int = 0,
)
    d = spzeros(get_ndofs(U))
    apply_dirichlet_to_vector!(d, U, V, mesh, t; dt_derivative_order)
    return d
end

"""
Apply homogeneous Dirichlet condition on vector `b`

Dirichlet values are applied on dofs lying on a Dirichlet boundary.
"""
function apply_homogeneous_dirichlet_to_vector!(
    b::AbstractVector{<:Number},
    U,
    V,
    mesh::AbstractMesh,
)
    # Define callback to apply on `b`
    callback!(array, iglob, _, _) = array[iglob] = 0.0

    _apply_dirichlet!((b,), (callback!,), U, V, mesh)
end

"""
Apply a homogeneous "dirichlet" condition on the input matrix.

The rows and columns corresponding to each Dirichlet dof are canceled (with a 1 on the diag term).
If you don't want the cols to be canceled, use `apply_dirichlet_to_matrix!`.
"""
function apply_homogeneous_dirichlet_to_matrix!(
    matrix::AbstractMatrix,
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
Same as above, but with a Tuple of matrix
"""
function apply_homogeneous_dirichlet_to_matrix!(
    matrices::Tuple{Vararg{AbstractMatrix, N}},
    U,
    V,
    mesh::AbstractMesh;
    diag_values::Tuple{Vararg{Number, N}} = ntuple(i -> 1.0, N),
) where {N}
    callbacks = ntuple(
        i -> callback!(array, iglob, _, _) = begin
            array[iglob, :] .= 0.0
            array[:, iglob] .= 0.0
            array[iglob, iglob] = diag_values[i]
        end,
        N,
    )
    _apply_dirichlet!(matrices, callbacks, U, V, mesh)
end

"""
Apply homogeneous Dirichlet condition on `A` and `b` assuming the linear system Ax=b.

Dirichlet values are applied on dofs lying on a Dirichlet boundary.
"""
function apply_homogeneous_dirichlet!(
    A::AbstractMatrix{<:Number},
    b::AbstractVector{<:Number},
    U,
    V,
    mesh::AbstractMesh,
)
    # Define callback to apply on `A`
    callback_A!(array, iglob, _, _) = begin
        array[iglob, :] .= 0.0
        array[:, iglob] .= 0.0
        array[iglob, iglob] = 1.0
    end

    # Define callback to apply on `b`
    callback_b!(array, iglob, _, _) = array[iglob] = 0.0

    _apply_dirichlet!((A, b), (callback_A!, callback_b!), U, V, mesh)
end

"""
Apply Dirichlet condition on vector `b`.

Dirichlet values are applied on dofs lying on a Dirichlet boundary.

TODO : add `dt_derivative_order`
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
    # sizeU = get_size(U)

    # Define callback to apply on `b`
    callback!(array, iglob, icomp, values) = begin
        # if (values isa Number && sizeU > 1)
        #     # here we should consider that we have a condition of the type u ⋅ n
        #     error("Dirichlet condition with scalar product with normal not supported yet")
        # end
        array[iglob] = values[icomp]
    end

    _apply_dirichlet!((b,), (callback!,), U, V, mesh, t)
end

"""
Apply a "dirichlet" condition on the input matrix.

The columns corresponding to each Dirichlet dof are NOT canceled. If you want them to be canceled,
use `apply_homogeneous_dirichlet_to_matrix!`.
"""
function apply_dirichlet_to_matrix!(
    matrix::AbstractMatrix,
    U,
    V,
    mesh::AbstractMesh;
    diag_value::Number = 1.0,
)
    apply_dirichlet_to_matrix!((matrix,), U, V, mesh; diag_values = (diag_value,))
end

"""
Same as above, but with a Tuple of matrix

# Dev notes:
Use temporary transposed matrices for indexing efficiency
"""
function apply_dirichlet_to_matrix!(
    matrices::Tuple{Vararg{AbstractMatrix, N}},
    U,
    V,
    mesh::AbstractMesh;
    diag_values::Tuple{Vararg{Number, N}} = ntuple(i -> 1.0, N),
) where {N}
    # nullify given row (i.e column for the transposed matrix)
    # and set the diagonal value
    callbacks = ntuple(
        i ->
            callback!(array, iglob, _, _) = begin
                array[:, iglob] .= zero(diag_values[i])
                array[iglob, iglob] = diag_values[i]
            end,
        N,
    )
    _mat = map(copy ∘ transpose, matrices)
    _apply_dirichlet!(_mat, callbacks, U, V, mesh)
    for (m, _m) in zip(matrices, _mat)
        m .= transpose(_m)
    end
    nothing
end

function _apply_dirichlet!(
    arrays::Tuple{Vararg{AbstractVecOrMat, N}},
    callbacks::NTuple{N, Function},
    U::TrialFESpace{S, FE},
    V::TestFESpace{S, FE},
    mesh::AbstractMesh,
    t::Number = 0.0,
) where {N, S, FE}
    m = collect(1:get_ndofs(U))
    _apply_dirichlet!(arrays, callbacks, m, U, V, mesh, t)
end

"""
Version for MultiFESpace
"""
function _apply_dirichlet!(
    arrays::Tuple{Vararg{AbstractVecOrMat, P}},
    callbacks::NTuple{P, Function},
    U::AbstractMultiFESpace{N, Tu},
    V::AbstractMultiFESpace{N, Tv},
    mesh::AbstractMesh,
    t::Number = 0.0,
) where {P, N, Tu <: Tuple{Vararg{TrialFESpace}}, Tv <: Tuple{Vararg{TestFESpace}}}
    for (i, (_U, _V)) in enumerate(zip(U, V))
        _apply_dirichlet!(arrays, callbacks, get_mapping(U, i), _U, _V, mesh, t)
    end
end

"""
`m` is the dof mapping (if SingleFESpace, m = Id; otherwise m = get_mapping(...))

# Warning
We assume same FESpace for U and V

# Dev notes
* we could passe a view of the arrays instead of passing the mapping
"""
function _apply_dirichlet!(
    arrays::Tuple{Vararg{AbstractVecOrMat, N}},
    callbacks::NTuple{N, Function},
    m::Vector{Int},
    U::TrialFESpace{S, FE},
    V::TestFESpace{S, FE},
    mesh::AbstractMesh,
    t::Number,
) where {S, FE, N}
    # Alias
    _mesh = parent(mesh)
    # fs_U = get_function_space(U)
    fs_V = get_function_space(V)
    dhl_V = _get_dhl(V)

    # Loop over the boundaries
    for bndTag in get_dirichlet_boundary_tags(U)
        # Function to apply, giving the AbstractLazy for time t
        # f_t_lazy is an AbstractLazy (e.g., PhysicalFunction)
        f_t_lazy = get_dirichlet_values(U, bndTag)(t)

        # Build BoundaryFaceDomain for this boundary tag
        bndDomain = BoundaryFaceDomain(_mesh, String(bndTag))

        # Loop over the faces of the boundary using foreach_element
        foreach_element(bndDomain) do fInfo::FaceInfo, _, _
            _apply_dirichlet_on_face!(
                arrays,
                callbacks,
                fInfo,
                _mesh,
                fs_V,
                dhl_V,
                m,
                f_t_lazy,
            )
        end
    end
end

"""
Apply dirichlet condition on a face given by `fInfo`.

`m` is the dof mapping to obtain global dof id (in case of MultiFESpace)
`f_t_lazy` is the AbstractLazy representing the Dirichlet value at time t
`callbacks` is the Tuple of functions of (array, idof_glo, icomp, values) that
must be applied to each array of `arrays`

# Dev notes
* we could passe a view of the arrays instead of passing the mapping
"""
function _apply_dirichlet_on_face!(
    arrays::Tuple{Vararg{AbstractVecOrMat, N}},
    callbacks::NTuple{N, Function},
    fInfo::FaceInfo,
    mesh::Mesh,
    fs_V::AbstractFunctionSpace,
    dhl_V::DofHandler,
    m::Vector{Int},
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
    _f2n = get_nodes_index(fInfo)

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
