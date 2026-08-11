# [Mesh Data](@id meshdata)

The functions `MeshCellData`, `MeshFaceData` and `MeshPointData` allow describing data known respectively at cell centers, face centers, or mesh nodes. The "values" can be of any type: a vector of scalars (conductivity per cell), an array of functions, etc.

!!! tip
    The function `Bcube.convert_to_lagrange_P1(mesh, data)` converts a `MeshPointData` into a Lagrange `FEFunction` of degree 1.

```@autodocs
Modules = [Bcube]
Pages   = ["meshdata.jl"]
```
