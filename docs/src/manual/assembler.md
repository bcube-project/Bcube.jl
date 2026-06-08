# Assemble (bi)linear forms

!!! todo ""
    This page is under construction. Checkout the [API reference](@ref dof) for more details.

## Apply dirichlet conditions

Once the linear system has been assembled (i.e., the matrix `A` and vector `b` have been computed from the bilinear and linear forms), boundary conditions must be applied before solving. This section describes the functions available for applying Dirichlet boundary conditions to the assembled matrices and vectors.

Dirichlet conditions constrain the solution on specific boundaries of the domain. The functions below allow you to modify the assembled system to enforce these constraints, either for homogeneous (zero) or non-homogeneous (non-zero) boundary values.

### Homogeneous Dirichlet conditions

For homogeneous Dirichlet conditions (where the boundary value is zero), use the following functions:

```julia
apply_homogeneous_dirichlet_to_vector!(b, U, V, mesh)
```

Apply homogeneous Dirichlet condition on vector `b`. Dirichlet values are applied on dofs lying on a Dirichlet boundary, setting them to zero.

```julia
apply_homogeneous_dirichlet_to_matrix!(matrix, U, V, mesh; diag_value=1.0)
```

Apply a homogeneous Dirichlet condition on the input matrix. The rows and columns corresponding to each Dirichlet dof are canceled (with the specified value on the diagonal term, default is 1.0). If you don't want the columns to be canceled, use `apply_dirichlet_to_matrix!` instead.

```julia
apply_homogeneous_dirichlet!(A, b, U, V, mesh)
```

Apply homogeneous Dirichlet condition on both `A` and `b` assuming the linear system Ax=b. This is a convenience function that combines the two previous operations.

### Non-homogeneous Dirichlet conditions

For non-homogeneous Dirichlet conditions (where the boundary value is non-zero), use:

```julia
apply_dirichlet_to_vector!(b, U, V, mesh, t=0.0; dt_derivative_order=0)
```

Apply Dirichlet condition on vector `b`. Dirichlet values are applied on dofs lying on a Dirichlet boundary. The values are obtained from the Dirichlet values defined in the trial space `U`.

```julia
apply_dirichlet_to_matrix!(matrix, U, V, mesh; diag_value=1.0)
```

Apply a Dirichlet condition on the input matrix. The columns corresponding to each Dirichlet dof are **not** canceled. If you want them to be canceled, use `apply_homogeneous_dirichlet_to_matrix!` instead.

### Assembling Dirichlet vectors

```julia
assemble_dirichlet_vector(U, V, mesh, t=0.0; dt_derivative_order=0)
assemble_dirichlet_sparse_vector(U, V, mesh, t=0.0; dt_derivative_order=0)
```

Assemble a vector of zeros dofs except on boundary dofs where they take the Dirichlet values. The sparse version returns a sparse vector, while the non-sparse version converts it to a dense vector.

### Arguments

- `U`: Trial finite element space (defines the Dirichlet boundary conditions)
- `V`: Test finite element space
- `mesh`: The computational mesh
- `t`: Time value for time-dependent problems (default: 0.0)
- `diag_value`: Value to set on the diagonal for Dirichlet dofs (default: 1.0)
- `dt_derivative_order`: Order of time derivative (currently only 0 is supported)

### Example

```julia
# Apply homogeneous Dirichlet conditions to a linear system
apply_homogeneous_dirichlet!(A, b, U, V, mesh)

# Or separately
apply_homogeneous_dirichlet_to_matrix!(A, U, V, mesh)
apply_homogeneous_dirichlet_to_vector!(b, U, V, mesh)

# For non-homogeneous conditions
apply_dirichlet_to_vector!(b, U, V, mesh, t)
apply_dirichlet_to_matrix!(A, U, V, mesh)
```
