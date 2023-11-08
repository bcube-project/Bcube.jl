module LinearTransport #hide
println("Running linear transport example...") #hide
# # Linear transport (DG)
# In this tutorial, we show how to solve a linear transport equation using a discontinuous-Galerkin
# framework with Bcube.
# # Theory
# In this example, we solve the following linear transport equation using discontinuous elements:
# ```math
# \frac{\partial \phi}{\partial t} + \nabla \cdot (c \phi) = 0
# ```
# where ``c`` is a constant velocity. Using an explicit time scheme, one obtains:
# ```math
# \phi^{n+1} = \phi^n - \Delta t \nabla \cdot (c \phi^n)
# ```
# The corresponding weak form of this equation is:
# ```math
# \int_\Omega \phi^{n+1} v \mathrm{\,d}\Omega = \int_\Omega \phi^n v \mathrm{\,d}\Omega + \Delta t \left[
# \int_\Omega c \phi^n \cdot \nabla v \mathrm{\,d}\Omega - \oint_\Gamma \left( c \phi \cdot n \right) v \mathrm{\,d}\Gamma
# \right]
# ```
# where ``\Gamma = \delta \Omega``. Adopting the discontinuous Galerkin framework, this equation is written in every mesh cell
# ``\Omega_i``. The cell boundary term involves discontinuous quantities and is replaced by a "numerical flux",
# leading to the expression:
# ```math
# \int_{\Omega_i} \phi^{n+1} v \mathrm{\,d}\Omega_i = \int_{\Omega_i} \phi^n v \mathrm{\,d}\Omega_i + \Delta t \left[
# \int_{\Omega_i} c \phi^n \cdot \nabla v \mathrm{\,d}\Omega_i - \oint_{\Gamma_i} F^*(\phi) v \mathrm{\,d} \Gamma_i
# \right]
# ```
# For this example, an upwind flux will be used for ``F^*``. Using a matrix formulation, the above equation can be written as:
# ```math
# \phi^{n+1} = \phi^n + M^{-1}(f_\Omega - f_\Gamma)
# ```
# where ``M^{-1}`` is the inverse of the mass matrix, ``f_\Omega`` the volumic flux term and ``f_\Gamma`` the surfacic flux term.
#
# # Commented code
# Start by importing the necessary packages:
# Load the necessary packages
using Bcube
using LinearAlgebra
using WriteVTK

# Before all, to ease to ease the solution VTK output we will write a
# structure to store the vtk filename and the number of iteration; and a function
# that exports the solution on demand. Note the use of `var_on_nodes_discontinuous`
# to export the solution on the mesh nodes, respecting the discontinuous feature of the
# solution.
mutable struct VtkHandler
    basename::Any
    ite::Any
    mesh::Any
    VtkHandler(basename, mesh) = new(basename, 0, mesh)
end

function append_vtk(vtk, u::Bcube.AbstractFEFunction, t)
    ## Values on center
    values = var_on_nodes_discontinuous(u, vtk.mesh)

    ## Write
    Bcube.write_vtk_discontinuous(
        vtk.basename,
        vtk.ite,
        t,
        vtk.mesh,
        Dict("u" => (values, VTKPointData())),
        1;
        append = vtk.ite > 0,
    )

    ## Update counter
    vtk.ite += 1
end

# First, we define some physical and numerical constant parameters
const degree = 0 # Function-space degree (Taylor(0) = first order Finite Volume)
const c = [1.0, 0.0] # Convection velocity (must be a vector)
const nite = 100 # Number of time iteration(s)
const CFL = 1 # 0.1 for degree 1
const nx = 41 # Number of nodes in the x-direction
const ny = 41 # Number of nodes in the y-direction
const lx = 2.0 # Domain width
const ly = 2.0 # Domain height
const Δt = CFL * min(lx / nx, ly / ny) / norm(c) # Time step

# Then generate the mesh of a rectangle using Gmsh and read it
tmp_path = "tmp.msh"
gen_rectangle_mesh(tmp_path, :quad; nx = nx, ny = ny, lx = lx, ly = ly, xc = 0.0, yc = 0.0)
mesh = read_msh(tmp_path)
rm(tmp_path)

# We can now init our `VtkHandler`
out_dir = joinpath(@__DIR__, "../myout")
isdir(out_dir) || mkdir(out_dir) #hide
vtk = VtkHandler(joinpath(out_dir, "linear_transport"), mesh)

# As seen in the previous tutorial, the definition of trial and test spaces needs a mesh and
# a function space. Here, we select Taylor space, and build discontinuous FE spaces with it.
# Then an FEFunction, that will represent our solution, is created.
fs = FunctionSpace(:Taylor, degree)
U = TrialFESpace(fs, mesh, :discontinuous)
V = TestFESpace(U)
u = FEFunction(U)

# Define measures for cell and interior face integrations
Γ = InteriorFaceDomain(mesh)
Γ_in = BoundaryFaceDomain(mesh, "West")
Γ_out = BoundaryFaceDomain(mesh, ("North", "East", "South"))

dΩ = Measure(CellDomain(mesh), 2 * degree + 1)
dΓ = Measure(Γ, 2 * degree + 1)
dΓ_in = Measure(Γ_in, 2 * degree + 1)
dΓ_out = Measure(Γ_out, 2 * degree + 1)

# We will also need the face normals associated to the different face domains.
# Note that this operation is lazy, `nΓ` is just an abstract representation on
# face normals of `Γ`.
nΓ = get_face_normals(Γ)
nΓ_in = get_face_normals(Γ_in)
nΓ_out = get_face_normals(Γ_out)

# Let's move on to the bilinear and linear forms. First, the two easiest ones:
m(u, v) = ∫(u ⋅ v)dΩ # Mass matrix
l_Ω(v) = ∫((c * u) ⋅ ∇(v))dΩ # Volumic convective term

# For the flux term, we first need to define a numerical flux. It is convenient to define it separately
# in a dedicated function. Here is the definition of simple upwind flux.
function upwind(ui, uj, nij)
    cij = c ⋅ nij
    if cij > zero(cij)
        flux = cij * ui
    else
        flux = cij * uj
    end
    flux
end
# We then define the "flux" as the composition of the upwind function and the needed entries: namely the
# solution on the negative side of the face, the solution on the positive face, and the face normal. The
# orientation negative/positive is arbitrary, the only convention is that the face normals are oriented from
# the negative side to the positive side.
flux = upwind ∘ (side⁻(u), side⁺(u), side⁻(nΓ))
l_Γ(v) = ∫(flux * jump(v))dΓ

# Finally, we define what to perform on the "two" boundaries : inlet / oulet.
# On the inlet, we directly impose the flux with a user defined function that depends on the time
# (the input is an oscillating wave).
# On the outlet, we keep our upwind flux but we impose the ghost cell value.
bc_in = t -> PhysicalFunction(x -> c .* cos(3 * x[2]) * sin(4 * t)) # flux
l_Γ_in(v, t) = ∫(side⁻(bc_in(t)) ⋅ side⁻(nΓ_in) * side⁻(v))dΓ_in
flux_out = upwind ∘ (side⁻(u), 0.0, side⁻(nΓ_out))
l_Γ_out(v) = ∫(flux_out * side⁻(v))dΓ_out

# Assemble the (constant) mass matrix. The returned matrix is a sparse matrix. To simplify the
# tutorial, we will directly compute the inverse mass matrix. But note that way more performant
# strategies should be employed to solve such a problem (since we don't need the inverse, only the
# matrix-vector product).
M = assemble_bilinear(m, U, V)
invM = inv(Matrix(M)) #WARNING : really expensive !!!

# Let's also create three vectors to avoid allocating them at each time step
nd = get_ndofs(V)
b_vol = zeros(nd)
b_fac = similar(b_vol)
rhs = similar(b_vol)

# The time loop is trivial : at each time step we compute the linear forms using
# the `assemble_` methods, we complete the rhs, perform an explicit step and write
# the solution.
t = 0.0
for i in 1:nite
    global t

    ## Reset pre-allocated vectors
    b_vol .= 0.0
    b_fac .= 0.0

    ## Compute linear forms
    assemble_linear!(b_vol, l_Ω, V)
    assemble_linear!(b_fac, l_Γ, V)
    assemble_linear!(b_fac, v -> l_Γ_in(v, t), V)
    assemble_linear!(b_fac, l_Γ_out, V)

    ## Assemble rhs
    rhs .= Δt .* invM * (b_vol - b_fac)

    ## Update solution
    u.dofValues .+= rhs

    ## Update time
    t += Δt

    ## Write to file
    append_vtk(vtk, u, t)
end

# And here is an animation of the result:
# ```@raw html
# <img src="../assets/linear_transport.gif" alt="drawing" width="700"/>
# ```

end #hide
