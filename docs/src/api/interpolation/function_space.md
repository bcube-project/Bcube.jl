# Function spaces

# Generic
```@autodocs
Modules = [Bcube]
Pages   = ["function_space.jl"]
```

## Lagrange

For N > 1, the default version consists in "replicating" the shape functions.
If `shape_functions` returns the vector `[λ₁; λ₂; λ₃]`, and if the `FESpace` is of size `2`,
then this default behaviour consists in returning the matrix `[λ₁ 0; λ₂ 0; λ₃ 0; 0 λ₁; 0 λ₂; 0 λ₃]`.

### Square

#### Order 0
```math
\nabla \hat{\lambda}(\xi, \eta) =
\begin{pmatrix}
    0 \\ 0
\end{pmatrix}
```

### Triangle

#### Order 0
```math
\nabla \hat{\lambda}(\xi, \eta) =
\begin{pmatrix}
    0 \\ 0
\end{pmatrix}
```

#### Order 1
```math
\hat{\lambda}_1(\xi, \eta) = 1 - \xi - \eta \hspace{1cm}
\hat{\lambda}_2(\xi, \eta) = \xi                \hspace{1cm}
\hat{\lambda}_3(\xi, \eta) = \eta
```

```math
\begin{aligned}
    & \nabla \hat{\lambda}_1(\xi, \eta) =
        \begin{pmatrix}
            -1 \\ -1
        \end{pmatrix} \\
    & \nabla \hat{\lambda}_2(\xi, \eta) =
        \begin{pmatrix}
            1 \\ 0
        \end{pmatrix} \\
    & \nabla \hat{\lambda}_3(\xi, \eta) =
        \begin{pmatrix}
            0 \\ 1
        \end{pmatrix} \\
\end{aligned}
```

#### Order 2
```math
\begin{aligned}
    & \hat{\lambda}_1(\xi, \eta) = (1 - \xi - \eta)(1 - 2 \xi - 2 \eta) \
    & \hat{\lambda}_2(\xi, \eta) = \xi (2\xi - 1) \
    & \hat{\lambda}_3(\xi, \eta) = \eta (2\eta - 1) \
    & \hat{\lambda}_{12}(\xi, \eta) = 4 \xi (1 - \xi - \eta) \
    & \hat{\lambda}_{23}(\xi, \eta) = 4 \xi \eta \
    & \hat{\lambda}_{31}(\xi, \eta) = 4 \eta (1 - \xi - \eta)
\end{aligned}
```

```math
\begin{aligned}
    & \nabla \hat{\lambda}_1(\xi, \eta) =
        \begin{pmatrix}
            -3 + 4 (\xi + \eta) \\ -3 + 4 (\xi + \eta)
        \end{pmatrix} \\
    & \nabla \hat{\lambda}_2(\xi, \eta) =
        \begin{pmatrix}
            -1 + 4 \xi \\ 0
        \end{pmatrix} \\
    & \nabla \hat{\lambda}_3(\xi, \eta) =
        \begin{pmatrix}
            0 \\ -1 + 4 \eta
        \end{pmatrix} \\
    & \nabla \hat{\lambda}_{12}(\xi, \eta) =
        4 \begin{pmatrix}
            1 - 2 \xi - \eta \\ - \xi
        \end{pmatrix} \\
    & \nabla \hat{\lambda}_{23}(\xi, \eta) =
        4 \begin{pmatrix}
            \eta \\ \xi
        \end{pmatrix} \\
    & \nabla \hat{\lambda}_{31}(\xi, \eta) =
        4 \begin{pmatrix}
            - \eta \\ 1 - 2 \eta - \xi
        \end{pmatrix} \\
\end{aligned}
```

### Tetra

#### Order 1
```math
\hat{\lambda}_1(\xi, \eta, \zeta) = (1 - \xi - \eta - \zeta) \hspace{1cm}
\hat{\lambda}_2(\xi, \eta, \zeta) = \xi                        \hspace{1cm}
\hat{\lambda}_3(\xi, \eta, \zeta) = \eta                       \hspace{1cm}
\hat{\lambda}_5(\xi, \eta, \zeta) = \zeta                      \hspace{1cm}
```

### Prism
#### Order 1
```math
\begin{aligned}
    \hat{\lambda}_1(\xi, \eta, \zeta) = (1 - \xi - \eta)(1 - \zeta)/2 \hspace{1cm}
    \hat{\lambda}_2(\xi, \eta, \zeta) = \xi (1 - \zeta)/2          \hspace{1cm}
    \hat{\lambda}_3(\xi, \eta, \zeta) = \eta (1 - \zeta)/2  \hspace{1cm}
    \hat{\lambda}_5(\xi, \eta, \zeta) = (1 - \xi - \eta)(1 + \zeta)/2 \hspace{1cm}
    \hat{\lambda}_6(\xi, \eta, \zeta) = \xi (1 + \zeta)/2          \hspace{1cm}
    \hat{\lambda}_7(\xi, \eta, \zeta) = \eta (1 + \zeta)/2  \hspace{1cm}
\end{aligned}
```


## Taylor
The `Taylor` function space corresponds to a function space where functions are approximated by a Taylor series expansion of order ``n`` in each cell:
```math
    \forall x \in \Omega_i,~g(x) = g(x_0) + (x - x_0) g'(x_0) + o(x)
```
where ``x_0`` is the cell center.

Note that a Taylor-P0 is strictly equivalent to a 1st-order Finite Volume discretization (beware that "order" can have different meaning depending on whether one refers to the order of the function space basis or the order of the discretization method).

Recall that any function space implies that any function ``g`` is interpolated by ``g(x) = \sum g_i \lambda_i(x)`` where ``\lambda_i`` are the shape functions. For a Taylor expansion, the definition of ``\lambda_i`` is not unique. For instance for the Taylor expansion of order ``1`` on a 1D line above, we may be tempted to set ``\lambda_1(x) = 1`` and ``\lambda_2(x) = (x - x_0)``. If you do so, what are the corresponding shape functions in the reference element, the ``\hat{\lambda_i}``? We immediately recover ``\hat{\lambda_1}(\hat{x})  = 1``. For ``\hat{\lambda_2}``:
```math
    \hat{\lambda_2}(\hat{x}) = (\lambda \circ F)(\hat{x}) = (x \rightarrow x - x_0) \circ (\hat{x} \rightarrow \frac{x_r - x_l}{2} \hat{x} + \frac{x_r + x_l}{2}) = \frac{x_r - x_l}{2} \hat{x}
```
So if you set ``\lambda_2(x) = (x - x_0)`` then ``\hat{\lambda_2}`` depends on the element length (``\Delta x = x_r-x_l``), which is pointless. So ``\lambda_2`` must be proportional to the element length to obtain a universal definition for ``\hat{\lambda_2}``. For instance, we may choose ``\lambda_2(x) = (x - x_0) / \Delta x``, leading to ``\hat{\lambda_2}(\hat{x}) = \hat{x} / 2``. But we could have chosen an other element length multiple.

Don't forget that choosing ``\lambda_2(x) = (x - x_0) / \Delta x`` leads to ``g(x) = g(x_0) + \frac{x - x_0}{\Delta x} g'(x_0) Δx `` hence ``g_2 = g'(x_0) Δx`` in the interpolation.



```@autodocs
Modules = [Bcube]
Pages   = ["taylor.jl"]
```