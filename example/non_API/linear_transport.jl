module LinearTransport #hide
println("Running linear transport example...") #hide
# # Linear transport
# ## Theory
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
# ## Solution with Bcube
# Start by importing the necessary packages:
# Load the necessary packages (Bcube is loaded only if not already loaded)
const dir = string(@__DIR__, "/../../") # Bcube dir
include(dir * "src/Bcube.jl")
using .Bcube
using LinearAlgebra
using WriteVTK
using StaticArrays
using BenchmarkTools

# Basically, we need three methods to solve the problem : one to compute the flux terms (looping on the mesh faces), one to assemble
# the problem (volumic + surfacic terms) and one to step forward in time. Before defining these three methods, let's define auxiliary
# ones that help improving the code readability. First we define our numerical flux function, the updwind flux:
"""
  Upwind flux. Face normal oriented from cell i to cell j.
  Here `c` is the constant convection velocity.
"""
function upwind(ϕᵢ, ϕⱼ, c, nᵢⱼ)
    vij = c ⋅ nᵢⱼ
    if vij > zero(vij)
        vij * ϕᵢ
    else
        vij * ϕⱼ
    end
end

# Then we define two methods to build the inverse of the mass matrix in each mesh cell. For this problem, the mass matrix
# are time-independant, so we will compute them only once.

"""
    Inverse of the mass matrix in a given cell.
"""
function inv_mass_matrix(λ, cnodes, ct, order)
    M = integrate_ref(ξ -> ⊗(λ(ξ)), cnodes, ct, order)
    return inv(M)
end

"""
    Build the inverse of mass matrix for all mesh cells
    @TODO : use projection.L2_projector
"""
function build_mass_matrix_inv(mesh, fs, params)
    ## Get cell -> node connectivity and cell types
    c2n = connectivities_indices(mesh, :c2n)
    cellTypes = cells(mesh)

    ## Alias for some params (optionnal)
    order = params.order

    ## Alias for quadrature orders
    orderM = Val(2 * order + 1)

    ## Allocate
    inv_matrix = [
        zeros(ntuple(i -> ndofs(fs, shape(cellTypes[icell])), 2)) for
        icell in 1:ncells(mesh)
    ]

    ## Assemble
    for icell in 1:ncells(mesh)

        ## Alias for cell type
        ct = cellTypes[icell]

        ## Alias for nodes
        n = get_nodes(mesh, c2n[icell])

        ## Get shape functions
        λ = shape_functions(fs, shape(ct))

        ## Compute inverse of mass matrix
        inv_matrix[icell] .= inv_mass_matrix(λ, n, ct, orderM)
    end
    return inv_matrix
end

# Now that we have these two methods in hand, we can write the main methods. Let's start with the one computing the flux on
# each mesh face. For inner faces, the flux is computed and used for both neighbor cells. For outter (=boundary) faces,
# a boundary-condition-dependant treatment is applied. We will see later how to define boundary conditions. Note the use
# of the `interpolate` function that builds the interpolating function ``\tilde{\phi} = \sum \phi_i \lambda_i`` using the values
# of ``\phi`` on the degree of freedom.

function compute_flux(mesh, ϕ, params, q, t)
    ## Allocate
    ## This is just the the demo -> for performance, this vector should be allocated
    ## outside of this function and given as an input
    f = zeros(ndofs(ϕ))

    ## Alias
    c = params.c

    ## Get cell -> node, face -> node and face -> cell connectivities
    c2n = connectivities_indices(mesh, :c2n)
    f2n = connectivities_indices(mesh, :f2n)
    f2c = connectivities_indices(mesh, :f2c)

    ## Cell and face types
    cellTypes = cells(mesh)
    faceTypes = faces(mesh)

    ## Function space
    fs = function_space(ϕ)

    ## Integration order
    order = Val(2 * params.order + 1) # it's a lucky guess since we don't really know the "flux order"

    ## Loop on all the inner faces
    for kface in inner_faces(mesh)
        ## Face nodes, type and shape
        ftype = faceTypes[kface]
        fnodes = get_nodes(mesh, f2n[kface])

        ## Neighbor cell i
        i = f2c[kface][1]
        xᵢ = get_nodes(mesh, c2n[i])
        ctᵢ = cellTypes[i]
        shapeᵢ = shape(ctᵢ)
        λᵢ = shape_functions(fs, shape(ctᵢ))
        ϕᵢ = interpolate(λᵢ, q[dof(ϕ, i)])
        sideᵢ = cell_side(ctᵢ, c2n[i], f2n[kface])
        fpᵢ = mapping_face(shapeᵢ, sideᵢ) # for adjacent cell 1, we assume that glob2loc = 1:nnodes(face)

        ## Neighbor cell j
        j = f2c[kface][2]
        xⱼ = get_nodes(mesh, c2n[j])
        ctⱼ = cellTypes[j]
        λⱼ = shape_functions(fs, shape(ctⱼ))
        ϕⱼ = interpolate(λⱼ, q[dof(ϕ, j)])
        shapeⱼ = shape(ctⱼ)
        sideⱼ = cell_side(ctⱼ, c2n[j], f2n[kface])
        # This part is a bit tricky : we want the face parametrization (face-ref -> cell-ref) on
        # side `j`. For this, we need to know the permutation between the vertices of `kface` and the
        # vertices of the `sideⱼ`-th face of cell `j`. However all the information we have for entities,
        # namely `fnodes` and `faces2nodes(ctⱼ, sideⱼ)` refer to the nodes, not the vertices. So we need
        # to retrieve the number of vertices of the face and then restrict the arrays to these vertices.
        # (by the way, we use that the vertices appears necessarily in first)
        # We could simplify the expressions below by introducing the notion of "vertex" in Entity, for
        # instance with `nvertices` and `faces2vertices`.
        nv = length(faces2nodes(shapeⱼ, sideⱼ)) # number of vertices of the face
        iglob_vertices_of_face_of_cell_j =
            [c2n[j][faces2nodes(ctⱼ, sideⱼ)[l]] for l in 1:nv]
        g2l = indexin(f2n[kface][1:nv], iglob_vertices_of_face_of_cell_j)
        fpⱼ = mapping_face(shapeⱼ, sideⱼ, g2l)

        ## Flux definition : upwind
        fluxn = (nᵢⱼ, ξ) -> upwind(ϕᵢ(fpᵢ(ξ)), ϕⱼ(fpⱼ(ξ)), c, nᵢⱼ) # Warning : 'flux' must contained the scalar product with n

        ## Append flux contribution of face `kface` to cell `i`, performing a surfacic integration
        g_ref = ξ -> fluxn(normal(xᵢ, ctᵢ, sideᵢ, ξ), ξ) * λᵢ(fpᵢ(ξ))
        f[dof(ϕ, i)] += integrate_ref(g_ref, fnodes, ftype, order)

        ## Append flux contribution of face `kface` to cell `j`, performing a surfacic integration
        g_ref = ξ -> fluxn(-normal(xⱼ, ctⱼ, sideⱼ, ξ), ξ) * λⱼ(fpⱼ(ξ))
        f[dof(ϕ, j)] -= integrate_ref(g_ref, fnodes, ftype, order)
    end

    ## Loop on all the boundary of type 'faces'
    for tag in keys(mesh.bc_faces)
        ## Loop over this boundary faces
        for kface in boundary_faces(mesh, tag)

            # Face nodes, type and shape
            ftype = faceTypes[kface]
            fnodes = get_nodes(mesh, f2n[kface])
            F = mapping(fnodes, ftype) # mapping face-ref coords to local coords

            ## Neighbor cell i
            i = f2c[kface][1]
            cnodes = get_nodes(mesh, c2n[i])
            ctype = cellTypes[i]
            s = shape(ctype)
            λ = shape_functions(fs, shape(ctype))
            ϕᵢ = interpolate(λ, q[dof(ϕ, i)])
            side = cell_side(ctype, c2n[i], f2n[kface])
            fp = mapping_face(s, side) # mapping face-ref -> cell-ref

            ## For a multi-variable problem, we should loop over the variables. Here we have only one variable
            ## Skip if no boundary condition on this boundary
            !haskey(params.cdts, tag) && continue

            ## Get associated boundary condition
            cdt = params.cdts[tag]

            ## Flux boundary condition
            if type(cdt) == :flux
                ## Append flux contribution of face `kface` to cell `i`
                g_ref = ξ -> normal(cnodes, ctype, side, ξ) ⋅ apply(cdt, F(ξ), t) * λ(fp(ξ))
                f[dof(ϕ, i)] += integrate_ref(g_ref, fnodes, ftype, order)

                ## Dirichlet boundary condition : we apply classic flux with imposed condition in ghost. We split in
                ## three steps to improve clarity, but this can be done in one line.
            elseif type(cdt) == :diri
                ϕ_bnd = ξ -> apply(cdt, F(ξ), t) # here ξ is in the face-ref-element, so F(ξ) is in the local element
                g_ref =
                    ξ ->
                        upwind(ϕᵢ(fp(ξ)), ϕ_bnd(ξ), c, normal(cnodes, ctype, side, ξ)) *
                        λ(fp(ξ)) # fp(ξ) is in the cell-ref-element
                f[dof(ϕ, i)] += integrate_ref(g_ref, fnodes, ftype, order)
            end
        end
    end

    ## Result : a vector of size `ndofs`
    return f
end

# Now let's write the assembling method which computes the volumic terms and assembles them with the surfacic flux terms.
# We call this method `explicit_step!` because it returns the ``\Delta \phi`` associated with the selected explicit time
# scheme.

function explicit_step!(mesh, ϕ, params, dq, q, t, cache)

    ## Get inverse of mass matrix from cache
    inv_mass_matrix, = cache

    ## Get cell -> node connectivity and cell types
    c2n = connectivities_indices(mesh, :c2n)
    cellTypes = cells(mesh)

    ## Alias for the function space
    fs = function_space(ϕ)

    ## Alias for some parameters
    c = params.c
    order = params.order

    ## Alias for quadrature orders
    orderF = Val(order + order - 1 + 1)

    ## Compute surfacic flux
    f = compute_flux(mesh, ϕ, params, q, t)

    ## Assemble
    for icell in 1:ncells(mesh)
        ## Allocate
        F = zeros(ndofs(ϕ, icell)) # Volumic flux

        ## Indices of all the variable dofs in this cell
        i_ϕ = dof(ϕ, icell)

        ## Alias for cell type
        ctype = cellTypes[icell]

        ## Alias for nodes
        cnodes = get_nodes(mesh, c2n[icell])

        ## Corresponding shape
        s = shape(ctype)

        ## Get shape functions in reference element
        λ = shape_functions(fs, s)

        ## Get gradient, in the local element, of shape functions
        ∇λ = grad_shape_functions(fs, ctype, cnodes)

        ## Create interpolation function
        ϕᵢ = interpolate(λ, q[i_ϕ])

        ## Loop over cell dofs for volumic term
        # bmxam : need an additionnal explanation here it should be ∇λ ⋅ (c ϕ) but
        # since ∇λ is a matrix, we compute every "dofs" at once
        F .= integrate_ref(ξ -> ∇λ(ξ) * (c .* ϕᵢ(ξ)), cnodes, ctype, orderF)

        ## Assemble
        dq[i_ϕ] .= inv_mass_matrix[icell] * (F .- f[i_ϕ])
    end
end

# We are all set to solve a linear transport equation. However, we will add two more things to ease the solution VTK output : a
# structure to store the vtk filename and the number of iteration:
mutable struct VtkHandler
    basename::Any
    ite::Any
    VtkHandler(basename) = new(basename, 0)
end

# ... and a method to interpolate the discontinuous solution on cell centers and to append it to the VTK file:
function append_vtk(vtk, ϕ, t)
    ## Write
    dict_vars = Dict("ϕ" => (var_on_centers(ϕ), VTKCellData()))
    write_vtk(vtk.basename, vtk.ite, t, mesh, dict_vars; append = vtk.ite > 0)

    ## Update counter
    vtk.ite += 1
end

# Now let's manipulate everything to solve a linear transport equation. First we define some constants, relative to
# the problem or the mesh :
const order = 0 # Function-space order (Taylor(0) = first order Finite Volume)
const nite = 100 # Number of time iteration(s)
const c = SA[1.0, 0.0] # Convection velocity (must be a vector)
const CFL = 1 # 0.1 for order 1
const nout = 100 # Number of time steps to save
const nx = 41 # Number of nodes in the x-direction
const ny = 41 # Number of nodes in the y-direction
const lx = 2.0 # Domain width
const ly = 2.0 # Domain height
Δt = CFL * min(lx / nx, ly / ny) / norm(c) # Time step

# Then generate the mesh of a rectangle using Gmsh and read it
tmp_path = "tmp.msh"
gen_rectangle_mesh(tmp_path, :quad; nx = nx, ny = ny, lx = lx, ly = ly, xc = 0.0, yc = 0.0)
mesh = read_msh(tmp_path)
rm(tmp_path)

# Create a `Variable` : we choose to use the `Taylor` function space and hence a discontinuous Galerkin framework.
fs = FunctionSpace(:Taylor, order)
fes = FESpace(fs, :discontinuous)

# Create the degree of freedom handler
ϕ = CellVariable(:ϕ, mesh, fes)

# Assign boundary conditions. On the west side of the rectangle, we impose an incomming wave : ``c \cos(\lambda_y y) \sin(\omega t)``.
# The wave oscillates in time and along the ``y`` direction. To impose this condition, we choose to directly set the numerical flux
# rather than a ghost value of ``\phi``. This choice is arbitrary and only serves this example. On all the other boundaries, we set
# ``\phi = 0`` using a Dirichlet boundary condition. Note that for a multivariable problem, the keys should be (tag, varname)
g(x, t) = c .* cos(3 * x[2]) * sin(4 * t) * 1.0 # flux
cdt_in = BoundaryCondition(:flux, g)
cdt_out = BoundaryCondition(:diri, (x, t) -> 0.0)
cdts = Dict(
    boundary_tag(mesh, "West") => cdt_in,
    boundary_tag(mesh, "East") => cdt_out,
    boundary_tag(mesh, "North") => cdt_out,
    boundary_tag(mesh, "South") => cdt_out,
)

# Then we create a `NamedTuple` to hold the simulation parameters.
params = (c = c, order = order, cdts = cdts)

# Let's allocate the unknown vector and set it to zero. Along with this vector, we also allocate the "increment" vector.
q = zeros(ndofs(ϕ))
dq = zeros(size(q))
set_values!(ϕ, q)

# Init vtk handler
vtk = VtkHandler(dir * "myout/linear_transport")

# Init time
t = 0.0

# Save initial solution
append_vtk(vtk, ϕ, t)

# Build the cache and store everything you want to compute only once (such as the mass matrice inverse...)
cache = (build_mass_matrix_inv(mesh, fs, params),)

# Let's loop to solve the equation.
for i in 1:nite
    ## Infos
    println("Iteration ", i)

    ## Step forward in time
    explicit_step!(mesh, ϕ, params, dq, q, t, cache)
    q .+= Δt * dq
    set_values!(ϕ, q)
    global t += Δt

    ## Write solution to file (respecting max. number of output)
    if (i % Int(max(floor(nite / nout), 1)) == 0)
        append_vtk(vtk, ϕ, t)
    end
end

@btime explicit_step!($mesh, $ϕ, $params, $dq, $q, $t, $cache)

# And here is an animation of the result:
# ```@raw html
# <img src="../assets/linear_transport.gif" alt="drawing" width="700"/>
# ```
end #hide
