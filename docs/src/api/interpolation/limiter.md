# Limiter

High-order discontinuous finite elements may produce spurious oscillations in the vicinity
of discontinuities. Slope limiters are a classical remedy: they locally reduce the order of
the scheme so that a discrete maximum principle (DMP) is satisfied. Bcube provides the
*linear scaling limiter* of Zhang & Shu, which scales — cell by cell — the fluctuation
around the cell mean so that the solution stays within the range of the neighboring cell
means.

The typical workflow is:

1. compute the cell mean of the field with `cell_mean`,
2. compute the limiter coefficients and the limited field with `linear_scaling_limiter`,
3. if the dofs are needed, project the limited field with `projection_l2!` (the limited
   field returned by `linear_scaling_limiter` is a lazy expression, not an `FEFunction`).

```julia
u_mean = cell_mean(u, dΩ)
lim_u, u_lim = linear_scaling_limiter(u, u_mean, Ω, (dΩ, dΓ))
u_limited = FEFunction(get_fespace(u))
projection_l2!(u_limited, u_lim, mesh)
```

!!! warning
    Nothing is automatic in the choice of the measures: the min/max of `u` are evaluated
    on the `targetMeasures` provided by the user, which should include both the cell
    measure and the interior face measure. In previous versions of the API the face
    measures were implicitly taken into account; omitting them now silently weakens the
    limiter.

## linear\_scaling\_limiter

```@docs
linear_scaling_limiter
```
