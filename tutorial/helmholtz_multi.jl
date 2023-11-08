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
# # Code
# Load the necessary packages
const dir = string(@__DIR__, "/")
using Bcube
using LinearAlgebra
using WriteVTK
using Printf

# Mesh a 2D rectangular domain with quads.
# mesh = one_cell_mesh(:line)
mesh = line_mesh(4)
# mesh = rectangle_mesh(3, 3)
# mesh = rectangle_mesh(21, 21)

# Next, create a scalar variable named `:u`. The Lagrange polynomial space is used here. By default,
# a "continuous" function space is created (by opposition to a "discontinuous" one). The degree is set to `1`.
degree = 1
fs = FunctionSpace(:Lagrange, degree)
U1 = TrialFESpace(fs, mesh)
U2 = TrialFESpace(fs, mesh)
V1 = TestFESpace(U1)
V2 = TestFESpace(U2)

U = MultiFESpace(U1, U2; arrayOfStruct = false) # `false` only to facilitate debug
V = MultiFESpace(V1, V2; arrayOfStruct = false) # `false` only to facilitate debug

# Define measures for cell and interior face integrations
dΩ = Measure(CellDomain(mesh), 2 * degree + 1)

# compute volume residuals
a1((u1, u2), (v1, v2)) = ∫(∇(u1) ⋅ ∇(v1) + ∇(u2) ⋅ ∇(v2))dΩ
a2((u1, u2), (v1, v2)) = ∫(u1 ⋅ v1 + u2 ⋅ v2)dΩ

# build sparse matrices from integration result
A = assemble_bilinear(a1, U, V)
B = assemble_bilinear(a2, U, V)

display(A)
display(B)
# @show display(B)

# # Compute eigen-values and vectors : we convert to dense matrix to avoid importing additionnal packages,
# # but it is quite easy to solve it in a "sparse way".
# vp, vecp = eigen(Array(A), Array(B))

# # Display the "first" five eigenvalues:
# @show sqrt.(abs.(vp[3:8]))

# # Check results with expected ones #hide
# results = sqrt.(abs.(vp[3:8])) #hide
# ref_results = [3.144823462554393, 4.447451992013584, 6.309054755690625, 6.309054755690786, 7.049403274103087, 7.049403274103147] #hide
# @assert all(results .≈ ref_results) "Invalid results" #hide

# # Now we can export the solution at nodes of the mesh for several eigenvalues.
# # We will restrict to the first 20 eigenvectors.
# nd = length(get_values(ϕ))
# nvecs = min(20, nd)
# values = zeros(nnodes(mesh), nvecs)
# for i in 1:nvecs
#     set_values!(ϕ, vecp[:, i])
#     values[:, i] = var_on_nodes(ϕ)
# end

# # To write a VTK file, we need to build a dictionnary linking the variable name with its
# # values and type
# dict_vars = Dict(@sprintf("u_%02d", i) => (values[:, i], VTKPointData()) for i in 1:nvecs)
# write_vtk(dir * "../myout/helmholtz_rectangle_mesh", 0, 0.0, mesh, dict_vars)

# And here is the eigenvector corresponding to the 6th eigenvalue:
# ```@raw html
# <img src="../assets/helmholtz_x21_y21_vp6.png" alt="drawing" width="500"/>
# ```

end #hide
