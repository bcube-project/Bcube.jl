```@meta
CurrentModule = Bcube
```

# Bcube.jl

Bcube is a Julia library providing tools for the spatial discretization of partial differential equation(s) (PDE). It offers a high-level API to discretize linear or non-linear problems on unstructured mesh using continuous or discontinuous finite elements (FEM - DG).

The main features are:

- high-level api : `a(u, v) = ∫(η * ∇(u) ⋅ ∇(v))dΩ`
- 1D, 2D, 3D unstructured mesh with high-order geometrical elements (gmsh format)
- Lagrange (continuous & discontinuous) and Taylor (discontinuous) finite elements (line, quad, tri, hexa, penta)
- arbitrary order for hypercube Lagrange elements

Commented tutorials as well as various examples can be found in the dedicated project [BcubeTutorials.jl](https://github.com/bcube-project/BcubeTutorials.jl).

## Installation

Bcube can be added to your Julia environment with this simple line :

```julia-repl
pkg> add https://github.com/bcube-project/Bcube.jl
```

## Alternatives

Numerous FEM-DG Julia packages are available, here is a non-exhaustive list;

- [Gridap.jl](https://github.com/gridap/Gridap.jl) (which has greatly influenced the development of Bcube)
- [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)
- [Trixi.jl](https://github.com/trixi-framework/Trixi.jl)

## Contribution

Any contribution(s) and/or remark(s) are welcome! Don't hesitate to open an issue to ask a question or signal a bug. PRs improving the code (new features, new elements, fixing bugs, ...) will be greatly appreciated.

## Gallery
| [Helmholtz equation](https://bcube-project.github.io/BcubeTutorials.jl/dev/tutorial/helmholtz) | [Phase field solidification](https://bcube-project.github.io/BcubeTutorials.jl/dev/tutorial/phase_field_supercooled) | [Linear transport equation](https://bcube-project.github.io/BcubeTutorials.jl/dev/tutorial/linear_transport) |
| :----------------: | :------------------------: | :-----------------------: |
| ![](https://bcube-project.github.io/BcubeTutorials.jl/dev/assets/helmholtz_x21_y21_vp6.png) | ![](https://github.com/bcube-project/BcubeTutorials.jl/blob/main/docs/src/assets/phase-field-supercooled-rectangle.gif?raw=true) | ![](https://github.com/bcube-project/BcubeTutorials.jl/blob/main/docs/src/assets/linear_transport.gif?raw=true) |


## Authors
Ghislain Blanchard, Lokman Bennani and Maxime Bouyges
