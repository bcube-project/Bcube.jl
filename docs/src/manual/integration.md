# Integration

To compute an integral on a geometrical element, for instance a curved element, a variable substitution is used to compute the integral on the corresponding reference `Shape`. This variable substitution reads:

```math
\int_\Omega g(x) \mathrm{\,d} \Omega = \int_{\hat{\Omega}} |J(x)| \left(g \circ F \right)(\hat{x}) \mathrm{\,d} \hat{\Omega},
```

where we recall that $$F$$ is the reference to physical mapping and $$J$$ is the determinant of the jacobian matrix of this mapping. Depending on the shape and element order, this determinant is either hard-coded or computed with `ForwardDiff`.

Now, to compute the right side, i.e the integral on the reference shape, quadrature rules are applied to $\hat{g} = g \circ F$:

```math
\int_{\hat{\Omega}} \hat{g}(\hat{x}) \mathrm{\,d} \hat{\Omega} = \sum_{i =1}^{N_q} \omega_i \hat{g}(\hat{x}_i)
```

A specific procedure is applied to compute integrals on a face of a cell (i.e a surfacic integral on a face of a volumic element).
