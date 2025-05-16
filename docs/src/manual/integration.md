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

## Defining a (bi)linear form

When using the FEM or the DG method, (bi)linear forms have to be defined. These forms represent scalar products between functions, and involve an integration. For instance, the common bilinear form (corresponding to the mass matrix once assembled) is $$(u,v) \mapsto \int u\cdot v \, \mathrm{d}\Omega$$. In Bcube, this integral translates to:
```julia
mesh = (...)
Ω = CellDomain(mesh)
dΩ = Measure(Ω, 2)
a(u,v) = ∫(u⋅v)dΩ
```
You can see a new concept here : the `Measure` `dΩ`. A measure is built from a domain (cell of face) and a quadrature rule. Several quadrature rules are available, see [here](@ref quadratures).

!!! info
    We've done nothing else that defining a function `a` above. It is then intended to be used with the `assemble_bilinear` function (and FESpaces).

## Computing a raw integral

It is possible to compute an integral over a domain directly without assembling a (bi)linear form, for instance to compute the area of the cells, or the error between an analytical and a numerical solution. To do so, use the `Bcube.compute` function:
```julia
g = PhysicalFunction(x -> 2 * x[1])
result = Bcube.compute(∫(g)dΩ)
```
The obtained `result` is actually not the sum over the elements of `Ω`, but a `SparseVector` with the result of the integration element-wise. So if you integrate over a subset of the mesh cells, the result will be a sparse vector containing only non-zero values for those cells.

