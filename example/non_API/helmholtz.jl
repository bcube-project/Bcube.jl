module Helmholtz #hide
println("Running Helmholtz example...") #hide
# # Helmholtz
# # Theory
# We consider the following Helmholtz equation, representing for instance the acoustic wave propagation with Neuman boundary condition(s):
# ```math
# \begin{cases}
#   \Delta u + \omega^2 u = 0 \\
#   \dfrac{\partial u}{\partial n} = 0 \textrm{  on  } \Gamma
# \end{cases}
# ```
#
# An analytic solution of this equation can be obtained: for a rectangular domain $$\Omega = [0,L_x] \times [0,L_y]$$,
# ```math
# u(x,y) = \cos \left( \frac{k_x \pi}{L_x} x \right) \cos \left( \frac{k_y \pi}{L_y} y \right) \mathrm{~~with~~} k_x,~k_y \in \mathbb{N}
# ```
# with $$\omega^2 = \pi^2 \left( \dfrac{k_x^2}{L_x^2} + \dfrac{k_y^2}{L_y^2} \right)$$
#
# Now, both the finite-element method and the discontinuous Galerkin method requires to write the weak form of the problem:
# ```math
# - \int_\Omega \nabla u \cdot \nabla v \mathrm{\,d}\Omega
# + \underbrace{\left[ (\nabla u \cdot n) v \right]_\Gamma}_{=0} + \omega^2 \int_\Omega u v \mathrm{\,d} \Omega = 0
# ```
# ```math
# \int_\Omega \nabla u \cdot \nabla v \mathrm{\,d} \Omega = \omega^2 \int_\Omega u v \mathrm{\,d} \Omega
# ```
# This equation is actually a generalized eigenvalue problem which can be writtin in matrix / linear operator form:
# ```math
# A u = \alpha B u
# ```
# where
# ```math
# A u = \int_\Omega \nabla u \cdot \nabla v \mathrm{\,d} \Omega,~~ B u = \int_\Omega u v \mathrm{\,d} \Omega,~~ \alpha = \omega^2
# ```

# # Easiest solution
# Load the necessary packages (Bcube is loaded only if not already loaded)
const dir = string(@__DIR__, "/../../") # Bcube directory
include(dir * "src/Bcube.jl")
using .Bcube
using LinearAlgebra
using WriteVTK
using Printf

# Mesh a rectangular domain with quads.
mesh = rectangle_mesh(21, 21)

# Next, create a scalar Finite Element Space. The Lagrange polynomial space is used here. The order is set to `1`.
fes = FESpace(FunctionSpace(:Lagrange, 1), :continuous)

# Create a dof handler that will handle the numbering.
u = CellVariable(:u, mesh, fes)

# Allocate the problem matrices
nd = ndofs(u)
A = zeros(Float64, (nd, nd))
B = zeros(Float64, (nd, nd))

# Now let's assemble the problem, filling the matrices `A` and `B`. Since the assembly is the
# same for this first example and for the other examples below, we define a function that we
# will be able to reuse. `Bcube` performs all integration in the reference element. However
# working in the reference element is harder than working in the local element. For simple non-curved
# element, such as quads, the local shape functions and their gradient can be obtained using the `shape_functions`
# and `grad_shape_functions`. Then, just use the `∫` to integrate any function in the local element.
function assemble!(mesh, u, A, B)
    ## Get connectivity and cell types
    c2n = connectivities_indices(mesh, :c2n)
    cellTypes = cells(mesh)

    ## Retrieve function space
    fs = function_space(u)

    ## Compute needed quadrature orders
    orderA = Val(2 * (get_order(fs) - 1) + 1)
    orderB = Val(2 * get_order(fs) + 1)

    ## Loop on cells
    for icell in 1:ncells(mesh)
        ## Alias for cell type
        ct = cellTypes[icell]

        ## Alias for nodes
        n = get_nodes(mesh, c2n[icell])

        ## Corresponding shape
        s = shape(ct)

        ## Get shape functions in reference element
        λ = shape_functions(fs, s)

        ## Get gradient (in the local element) of shape functions
        ∇λ = grad_shape_functions(fs, ct, n)

        ## Loop over cell dofs
        for i in 1:ndofs(fs, s)
            for j in 1:ndofs(fs, s)
                A[dof(u, icell, 1, i), dof(u, icell, 1, j)] +=
                    integrate_ref(ξ -> ∇λ(ξ)[i, :] ⋅ ∇λ(ξ)[j, :], n, ct, orderA)
                B[dof(u, icell, 1, i), dof(u, icell, 1, j)] +=
                    integrate_ref(ξ -> λ(ξ)[i] * λ(ξ)[j], n, ct, orderB)
            end
        end
    end
end

# Use this function to fill the matrices:
assemble!(mesh, u, A, B)

# Compute eigen-values and vectors
vp, vecp = eigen(A, B)

# Display the "first" five eigenvalues:
@show sqrt.(abs.(vp[3:8]))

# Now we can export the solution. Since we used continuous Lagrange elements, the computed information
# is on nodes but not necessarily organized with the same numbering as the mesh nodes. Hence we will
# use the `vars_on_nodes` interpolation function : given variable names and a vector of values
# for these variables, this function builds a <variable name => values on nodes> dictionnary. Here our
# variable `Val(:u)` has as many values as the number of eigenvalues so we need to call this function multiple
# time. We will restrict to the first 20 eigenvectors.
nvecs = min(20, nd)
values = zeros(nnodes(mesh), nvecs)
for i in 1:nvecs
    set_values!(u, vecp[:, i])
    values[:, i] = var_on_vertices(u)
end

# To write a VTK file, we need to build a dictionnary linking the variable name with its
# values and type
dict_vars = Dict(@sprintf("u_%02d", i) => (values[:, i], VTKPointData()) for i in 1:nvecs)
write_vtk(dir * "myout/helmholtz_bcube_mesh", 0, 0.0, mesh, dict_vars)

# And here is the eigenvector corresponding to the 6th eigenvalue:
# ```@raw html
# <img src="../assets/helmholtz_x21_y21_vp6.png" alt="drawing" width="500"/>
# ```

# ## Hybrid mesh with Gmsh
# For this example we still solve the Helmholtz equation but importing a mesh
# generated by Gmsh.
# First, import the mesh from the `.msh` file.
mesh = read_msh(dir * "input/mesh/domainSquare_hybrid_2.msh")

# The mesh looks like this :
# ```@raw html
# <img src="../assets/helmholtz_hybrid_mesh.png" alt="drawing" width="300"/>
# ```

# We can keep the same variable as before, no need to define a new one. We could
# treat this problem with discontinuous Lagrange elements, but this is treated in
# an other example (see linear transport example).
# As previously done, build the dof handler and allocate the matrices
u = CellVariable(:u, mesh, fes)
nd = ndofs(u)
A = zeros(Float64, (nd, nd))
B = zeros(Float64, (nd, nd))

# Now we can call the exact same method to assemble the problem
assemble!(mesh, u, A, B)

# Compute eigen-values and vectors and show the first five eigenvalues:
vp, vecp = eigen(A, B)
@show sqrt.(abs.(vp[3:8]))

# Now we can export the solution. Like before, we first need to "interpolate" the eigenvectors
# on the mesh.
nvecs = min(20, nd)
values = zeros(nnodes(mesh), nvecs)
for i in 1:nvecs
    set_values!(u, vecp[:, i])
    values[:, i] = var_on_vertices(u)
end
dict_vars = Dict(@sprintf("u_%02d", i) => (values[:, i], VTKPointData()) for i in 1:nvecs)
write_vtk(dir * "myout/helmholtz_gmsh_mesh", 0, 0.0, mesh, dict_vars)

# You can check that the result is identical to the previous one (except for the mesh):
# ```@raw html
# <img src="../assets/helmholtz_hybrid_vp5.png" alt="drawing" width="500"/>
# ```

# ## Mesh with boundary conditions
# In this part we want to solve the Helmholtz equation on a disk domain, using homogeneous Dirichlet boundary
# condition on the border.

# We build the mesh of a disk. The circle domain boundary is named "BORDER"
tmp_path = "tmp.msh"
gen_disk_mesh(tmp_path)
mesh = read_msh(tmp_path)
rm(tmp_path)

# We will need the face to cell, face to nodes and cell to nodes connectivities, we can retrieve them very easily
c2n = connectivities_indices(mesh, :c2n)
f2n = connectivities_indices(mesh, :f2n)
f2c = connectivities_indices(mesh, :f2c)

# As previously done, build the dof handler and allocate the matrices
u = CellVariable(:u, mesh, fes)
nd = ndofs(u)
A = zeros(Float64, (nd, nd))
B = zeros(Float64, (nd, nd))

# Call the exact same method to assemble the problem
assemble!(mesh, u, A, B)

# Now we can update the matrices to impose our Dirichlet boundary condition
## Get dofs lying on the boundary condition "BORDER":
bnd_dofs = boundary_dofs(u, "BORDER")

## Loop over these dofs to impose Dirichlet condition:
for iglob in bnd_dofs
    ## Clear matrice entries
    A[iglob, :] .= 0.0
    B[iglob, :] .= 0.0

    ## Set the condition `u = 0` explicitely
    A[iglob, iglob] = 1.0
end

# Compute eigen-values and vectors and show the first five eigenvalues:
vp, vecp = eigen(A, B)
@show sqrt.(abs.(vp[3:8]))

# Export the solution
nvecs = min(20, nd)
values = zeros(nnodes(mesh), nvecs)
for i in 1:nvecs
    set_values!(u, vecp[:, i])
    values[:, i] = var_on_vertices(u)
end
dict_vars = Dict(@sprintf("u_%02d", i) => (values[:, i], VTKPointData()) for i in 1:nvecs)
write_vtk(dir * "myout/helmholtz_disk_mesh", 0, 0.0, mesh, dict_vars)

end #hide
