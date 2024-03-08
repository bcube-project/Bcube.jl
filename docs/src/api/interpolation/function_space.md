# Function spaces

# Generic
```@autodocs
Modules = [Bcube]
Pages   = ["function_space.jl"]
```

## Lagrange
```@autodocs
Modules = [Bcube]
Pages   = ["lagrange.jl"]
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