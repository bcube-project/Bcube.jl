# Limiter

High-order discontinuous finite elements may produce spurious oscillations in the vicinity
of discontinuities. Slope limiters are a classical remedy: they locally reduce the order of
the scheme so that a discrete maximum principle (DMP) is satisfied. Bcube provides the
*linear scaling limiter* of Zhang & Shu, which scales — cell by cell — the fluctuation
around the cell mean so that the solution stays within the range of the neighboring cell
means.

In each cell, the limited field is obtained by scaling the fluctuation around the cell
mean with a scalar coefficient `lim_u ∈ [0, 1]`:

    u_lim = u_mean + lim_u * (u - u_mean)

The cell mean is thus preserved by construction, and `lim_u = 0` corresponds to a
first-order (piecewise constant) solution.

The typical workflow is:

1. compute the cell mean of the field with [`cell_mean`](@ref),
2. compute the limiter coefficients and the limited field with [`linear_scaling_limiter`](@ref),
3. if the dofs are needed, project the limited field with [`projection_l2!`](@ref) (the
   limited field returned by `linear_scaling_limiter` is a lazy expression, not an
   `FEFunction`).

```julia
u_mean = cell_mean(u, dΩ)
lim_u, u_lim = linear_scaling_limiter(u, u_mean, dΩ)
u_limited = FEFunction(get_fespace(u))
projection_l2!(u_limited, u_lim, dΩ)
```

When the cell means are computed repeatedly — e.g. at each time step of an explicit
scheme — the underlying mass matrix can be assembled and factorized once with
[`build_cell_mean_cache`](@ref), and the means then obtained with `cell_mean(u, cache)`.

## cell\_mean

```@docs
cell_mean
```

```@docs
build_cell_mean_cache
```

## linear\_scaling\_limiter

```@docs
linear_scaling_limiter
```

```@docs
default_target_measures
```