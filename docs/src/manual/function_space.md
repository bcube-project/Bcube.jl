# Function and FE spaces

### `AbstractFunctionSpace`

In Bcube, a `FunctionSpace` is defined by a type (nodal Lagrange polynomials, modal Taylor expansion, etc) and a degree. For each implemented `FunctionSpace`, a list of shape functions is associated on a given `Shape`. For instance, one can get the shape functions associated to the Lagrange polynomials or order 3 on a `Square`. Note that for "tensor" elements such as `Line`, `Square` or `Cube`; the Lagrange polynomials are available at any order; being computed symbolically.

```julia
fs = FunctionSpae(:Lagrange, 2)
```

### `AbstractFESpace`

Then, an `FESpace` (more precisely `SingleFESpace`) is a function space associated to a numbering of the degrees of freedom. Note that the numbering may depend on the continuous or discontinuous feature of the space. Hence a `SingleFESpace` takes basically four input to be built : a `FunctionSpace`, the number of components of this space (scalar or vector), an indicator of the continuous/discontinuous characteristic, and the mesh. The dof numbering is built by combining the mesh numberings (nodes, cells, faces) and the function space. Note that the degree of the `FunctionSpace` can differ from the "degree" of the mesh elements : it is possible to build a `SingleFESpace` with P2 polynomials on a mesh only containing straight lines (defined by only two nodes, `Bar2_t`). 

For more clarity, the `SingleFESpace` is actually declined in two flavors : `TrialFESpace` and `TestFESpace`. These are the ones exposed to the API. The naming convention should be clear to the user familiar to the FE (and/or GD) method. Optionaly, a `TrialFESpace` can also contain the tags of the boundaries where Dirichlet condition(s) applies.

A `MultiFESpace` is simply a set of `TrialFESpace` (or a set of `TestFESpace`), eventually of different natures. Its befenit is that it allows to build a "global" numbering of all the dofs represented by this space. This is especially convenient to solve systems of equations.

```julia
fs = FunctionSpace(:Lagrange, 2)
U_fem_sca = TrialFEspace(fs, mesh) # scalar FEM FESpace
U_dg_sca = TrialFEspace(fs, mesh, :discontinuous) # scalar DG FESpace
U_dg_vec = TrialFEspace(fs, mesh, :discontinuous; size = 2) # 2-components DG FESpace

U = MultiFESpace(U_fem_sca, U_dg_sca, U_dg_vec)

V_fem_sca = TestFESpace(U_fem_sca)
V_dg_sca = TestFESpace(U_dg_sca)
V_dg_vec = TestFESpace(U_dg_vec)

V = MultiFESpace(V_fem_sca, V_dg_sca, V_dg_vec)
```

### `AbstractFEFunction`

With a `TrialFESpace`, one can build the representation of a function discretized on this space: a `FEFunction`. This structure stores a vector of values, one for each degree of freedom of the finite element space. To set or get the values of a `FEFunction`, the functions `set_dof_values!` and `get_dof_values` are available respectively. A `FEFunction` can be projected on another `FESpace`; or evaluated at some specific mesh location (a coordinates, all the nodes, all the mesh centers, etc).

```julia
U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
u = FEFunction(U)

dofs = get_dof_values(u) # get the values of the FEFunction
set_dof_values!(u, 2. * dofs) # assign the dof values
projection_l2!(u, PhysicalFunction(x -> norm(x)), mesh) # set the dof values using an L2 projection
```
