module heat_equation_API #hide
println("Running heat equation API example...") #hide
# # Heat equation (FE)
# In this tutorial, the heat equation (first steady and then unsteady) is solved using finite-elements.
#
# # Theory
# This example shows how to solve the heat equation with eventually variable physical properties in steady and unsteady formulations:
# ```math
#   \rho C_p \partial_t u - \nabla . ( \lambda u) = f
# ```
# We shall assume that $$f, \, \rho, \, C_p, \, \lambda \, \in L^2(\Omega)$$. The weak form of the problem is given by: find $$ u \in \tilde{H}^1_0(\Omega)$$
# (there will be at least one Dirichlet boundary condition) such that:
# ```math
#   \forall v \in  \tilde{H}^1_0(\Omega), \, \, \, \underbrace{\int_\Omega \partial_t u . v dx}_{m(\partial_t u,v)} + \underbrace{\int_\Omega \nabla u . \nabla v dx}_{a(u,v)} = \underbrace{\int_\Omega f v dx}_{l(v)}
# ```
# To numerically solve this problem we seek an approximate solution using Lagrange $$P^1$$ or $$P^2$$ elements.
# Here we assume that the domain can be split into two domains having different material properties.

# # Steady case
# As usual, start by importing the necessary packages.
using Bcube
using LinearAlgebra
using WriteVTK

# First we define some physical and numerical constants
const htc = 100.0 # Heat transfer coefficient (bnd cdt)
const Tr = 268.0 # Recovery temperature (bnd cdt)
const phi = 100.0
const q = 1500.0
const λ = 100.0
const η = λ
const ρCp = 100.0 * 200.0
const degree = 2
const outputpath = joinpath(@__DIR__, "../myout/heat_equation/")

# Read 2D mesh
mesh_path = joinpath(@__DIR__, "../input/mesh/domainSquare_tri.msh")
mesh = read_msh(mesh_path)

# Build function space and associated Trial and Test FE spaces.
# We impose a Dirichlet condition with a temperature of 260K
# on boundary "West"
fs = FunctionSpace(:Lagrange, degree)
U = TrialFESpace(fs, mesh, Dict("West" => 260.0))
V = TestFESpace(U)

# Define measures for cell integration
dΩ = Measure(CellDomain(mesh), 2 * degree + 1)

# Define bilinear and linear forms
a(u, v) = ∫(η * ∇(u) ⋅ ∇(v))dΩ
l(v) = ∫(q * v)dΩ

# Create an affine FE system and solve it using the `AffineFESystem` structure.
# The package `LinearSolve` is used behind the scenes, so different solver may
# be used to invert the system (ex: `solve(...; alg = IterativeSolversJL_GMRES())`)
# The result is a FEFunction (`ϕ`).
# We can interpolate it on mesh centers : the result is named `Tcn`.
sys = AffineFESystem(a, l, U, V)
ϕ = solve(sys)
Tcn = var_on_centers(ϕ, mesh)

# Compute analytical solution for comparison. Apply the analytical solution
# on mesh centers
T_analytical = x -> 260.0 + (q / λ) * x[1] * (1.0 - 0.5 * x[1])
Tca = map(T_analytical, get_cell_centers(mesh))

# Write both the obtained FE solution and the analytical solution to a vtk file.
mkpath(outputpath)
dict_vars =
    Dict("Temperature" => (Tcn, VTKCellData()), "Temperature_a" => (Tca, VTKCellData()))
write_vtk(outputpath * "result_steady_heat_equation", 0, 0.0, mesh, dict_vars)

# Compute and display the error
@show norm(Tcn .- Tca, Inf) / norm(Tca, Inf)

# # Unsteady case
# The code for the unsteady case if of course very similar to the steady case, at least for the
# beginning. Start by defining two additional parameters:
totalTime = 100.0
Δt = 0.1

# Read a slightly different mesh
mesh_path = joinpath(@__DIR__, "../input/mesh/domainSquare_tri_2.msh")
mesh = read_msh(mesh_path)

# The rest is similar to the steady case
fs = FunctionSpace(:Lagrange, degree)
U = TrialFESpace(fs, mesh, Dict("West" => 260.0))
V = TestFESpace(U)
dΩ = Measure(CellDomain(mesh), 2 * degree + 1)

# Compute matrices associated to bilinear and linear forms, and assemble
a(u, v) = ∫(η * ∇(u) ⋅ ∇(v))dΩ
m(u, v) = ∫(ρCp * u ⋅ v)dΩ
l(v) = ∫(q * v)dΩ

A = assemble_bilinear(a, U, V)
M = assemble_bilinear(m, U, V)
L = assemble_linear(l, V)

# Compute a vector of dofs whose values are zeros everywhere
# except on dofs lying on a Dirichlet boundary, where they
# take the Dirichlet value
Ud = assemble_dirichlet_vector(U, V, mesh)

# Apply lift
L = L - A * Ud

# Apply homogeneous dirichlet condition
apply_homogeneous_dirichlet_to_vector!(L, U, V, mesh)
apply_dirichlet_to_matrix!((A, M), U, V, mesh)

# Form time iteration matrix
# (note that this is bad for performance since up to now,
# M and A are sparse matrices)
Miter = factorize(M + Δt * A)

# Init the solution with a constant temperature of 260K
ϕ = FEFunction(U, 260.0)

# Write initial solution to a file
mkpath(outputpath)
dict_vars = Dict("Temperature" => (var_on_centers(ϕ, mesh), VTKCellData()))
write_vtk(outputpath * "result_unsteady_heat_equation", 0, 0.0, mesh, dict_vars)

# Time loop
itime = 0
t = 0.0
while t <= totalTime
    global t, itime
    t += Δt
    itime = itime + 1
    @show t, itime

    ## Compute rhs
    rhs = Δt * L + M * (get_dof_values(ϕ) .- Ud)

    ## Invert system and apply inverse shift
    set_dof_values!(ϕ, Miter \ rhs .+ Ud)

    ## Write solution (every 10 iterations)
    if itime % 10 == 0
        dict_vars = Dict("Temperature" => (var_on_centers(ϕ, mesh), VTKCellData()))
        write_vtk(
            outputpath * "result_unsteady_heat_equation",
            itime,
            t,
            mesh,
            dict_vars;
            append = true,
        )
    end
end

end #hide
