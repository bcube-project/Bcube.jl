# Projection

The [`projection_l2!`](@ref) function sets the dof values of an `FEFunction` from the
L2-projection of a field onto its `FESpace` — typically to initialize a solution from an
analytical expression, or to obtain the dofs of a lazy expression such as the limited
field returned by [`linear_scaling_limiter`](@ref).

Note that [`cell_mean`](@ref) is a specialized L2-projection onto the piecewise-constant
space, used to compute the mean value of a field in each cell.

```@docs
projection_l2!
```