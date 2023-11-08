module PhaseFieldSupercooled #hide
println("Running phase field supercooled equation example...") #hide

# # Phase field model - solidification of a liquid in supercooled state
# # Theory
# This case is taken from: Kobayashi, R. (1993). Modeling and numerical simulations of dendritic crystal growth. Physica D: Nonlinear Phenomena, 63(3-4), 410-423.
# In particular, the variables of the problem are denoted in the same way ($p$ for the phase indicator and $T$ for temperature).
# Consider a rectangular domain $$\Omega = [0, L_x] \times [0, L_y]$$ on which we wish to solve the following equations:
# ```math
#   \partial_t p = \epsilon^2 \Delta p + p (1-p)(p - \frac{1}{2} + m(T))
# ```
# ```math
#   \partial_t T = \Delta T + K \partial_t p
# ```
# where $m(T) = \frac{\alpha}{\pi} atan \left[ \gamma (T_e - T) \right]$.
# This set of equations represents the solidification of a liquid in a supercooled state. Here $T$ is a dimensionless temperature and $p$ is the solid volume fraction.
# Lagrange finite elements are used to discretize both equations. Time marching is performed with a forward Euler scheme for the first equation and a backward Euler scheme for the second one.
#
# To initiate the solidification process, a Dirichlet boundary condition ($p=1$,$T=1$) is applied at $x=0$ ("West" boundary).

const dir = string(@__DIR__, "/../../") # Bcube dir
include(dir * "src/Bcube.jl")
using .Bcube
using LinearAlgebra
using WriteVTK
using Printf
using SparseArrays

const l = 3.0
const nx = 100
const ny = 100
const eps = 0.01
const tau = 0.0003
const alp = 0.9
const gam = 10.0
const a = 0.01
const K = 1.6
const Te = 1.0

m = T -> (alp / pi) * atan(gam * (Te - T))

# Read mesh
#mesh = rectangle_mesh(nx, ny, xmin = 0, xmax = l, ymin = 0, ymax = l)
mesh = read_msh(dir * "input/mesh/domainPhaseField_tri.msh", 2)
# Function space
fs = FunctionSpace(:Lagrange, 1)

# Create a `Variable`
fes = FESpace(fs, :continuous)
u = CellVariable(:u, mesh, fes)

# Allocate the problem matrices and RHS
nd = ndofs(u)

function assembleRHS!(mesh, u, p, T)
    nd = ndofs(u)
    LP = zeros(Float64, (nd))

    ## Get connectivity and cell types
    c2n = connectivities_indices(mesh, :c2n)
    cellTypes = cells(mesh)

    ## Retrieve function space
    fs = function_space(u)

    ## Compute needed quadrature orders
    orderL = Val(get_order(fs) + 1)

    q = p .* (1.0 .- p) .* (p .- 0.5 .+ m.(T))

    ## Loop on cells
    for icell in 1:ncells(mesh)
        ## Alias for cell type
        ct = cellTypes[icell]

        i_ϕ = dof(u, icell)

        ## Alias for nodes
        n = get_nodes(mesh, c2n[icell])
        ## Corresponding shape
        s = shape(ct)

        ## Get shape functions in reference element
        λ = shape_functions(fs, s)

        ## Create interpolation function
        qsrc = interpolate(λ, q[i_ϕ])

        ## Loop over cell dofs to fill RHS L
        for i in 1:ndofs(fs, s)
            LP[dof(u, icell, 1, i)] += integrate_ref(ξ -> λ(ξ)[i] * qsrc(ξ), n, ct, orderL)
        end
    end

    return LP
end

function assemble!(mesh, u)
    ## Get connectivity and cell types
    c2n = connectivities_indices(mesh, :c2n)
    cellTypes = cells(mesh)

    ## Retrieve function space
    fs = function_space(u)

    ## Compute needed quadrature orders
    orderM = Val(2 * get_order(fs) + 1)
    orderA = Val(2 * (get_order(fs) - 1) + 1)

    ndm   = max_ndofs(u)
    nhint = ndm * ndm * ncells(mesh)
    Mval  = Float64[]
    rowM  = Int[]
    colM  = Int[]
    sizehint!(Mval, nhint)
    sizehint!(rowM, nhint)
    sizehint!(colM, nhint)

    Aval = Float64[]
    rowA = Int[]
    colA = Int[]
    sizehint!(Aval, nhint)
    sizehint!(rowA, nhint)
    sizehint!(colA, nhint)

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

        ## Loop over cell dofs to fill "stiffness" matrix A and RHS L
        for i in 1:ndofs(fs, s)
            for j in 1:ndofs(fs, s)
                push!(Aval, integrate_ref(ξ -> ∇λ(ξ)[i, :] ⋅ ∇λ(ξ)[j, :], n, ct, orderA))
                push!(rowA, dof(u, icell, 1, i))
                push!(colA, dof(u, icell, 1, j))

                push!(Mval, integrate_ref(ξ -> λ(ξ)[i] * λ(ξ)[j], n, ct, orderM))
                push!(rowM, dof(u, icell, 1, i))
                push!(colM, dof(u, icell, 1, j))
            end
        end
    end
    A = sparse(rowA, colA, Aval)
    M = sparse(rowM, colM, Mval)
    return M, A
end

# Function to get the indices of the dofs located on the boundary (this has meaning for Lagrange elements)
function generateBoundaryDofs!(mesh, u)

    ## Get cell -> node, face -> node and face -> cell connectivities
    c2n = connectivities_indices(mesh, :c2n)
    f2n = connectivities_indices(mesh, :f2n)
    f2c = connectivities_indices(mesh, :f2c)

    ## Cell and face types
    cellTypes = cells(mesh)

    ## Dictionary which will contain the list of dofs on a given boundary
    bnd_dofs = Dict{String, Vector{Int}}()

    ## Loop on all the boundary of type 'faces'
    for tag in keys(mesh.bc_faces)
        dof_glob = Vector{Int}()
        ## Loop over this boundarie's faces
        for kface in boundary_faces(mesh, tag)
            ## Neighbor cell
            icell = f2c[kface][1]
            ctype = cellTypes[icell]
            s = shape(ctype)
            side = cell_side(ctype, c2n[icell], f2n[kface])

            ## "Outer" dofs on face `side` (i.e faces vertices)
            ivertices = faces2nodes(s)[side] # get local index of vertices on face 'iside'
            for ivertex in ivertices
                idofs_loc = idof_by_vertex(fs, s)[ivertex]
                for idof_loc in idofs_loc
                    push!(dof_glob, dof(u, icell, 1, idof_loc))
                end
            end

            ## Add dofs located on the boundary egde
            for idof_loc in idof_by_edge(fs, s)[side]
                push!(dof_glob, dof(u, icell, 1, idof_loc))
            end
        end
        bnd_dofs[boundary_names(mesh, tag)] = unique(dof_glob)
    end
    return bnd_dofs
end

# Use this function to fill the matrices:
M, A = assemble!(mesh, u)

time = 0.0
dt = 0.0001
totalTime = 1.0

T0 = zeros(Float64, nd)
T1 = zeros(Float64, nd)

p0 = zeros(Float64, nd)
p1 = zeros(Float64, nd)

qp = zeros(Float64, nd)

Matp = (M + (dt / tau) * eps * eps * A)
MatT = (M + dt * A)

# generate boundary dofs:
bnd_dofs = generateBoundaryDofs!(mesh, u)

# here a Dirichlet boundary condition is applied on "West". p and T are imposed to 1 (solid).
for idof in bnd_dofs["West"]
    Matp[idof, :] .= 0.0
    Matp[idof, idof] = 1.0
    MatT[idof, :] .= 0.0
    MatT[idof, idof] = 1.0
    p0[idof] = 1.0
    T0[idof] = 1.0
end

# write initial condition in vtk format
set_values!(u, T0)
valuesT = var_on_nodes(u)
set_values!(u, p0)
valuesp = var_on_nodes(u)

dict_vars =
    Dict("Temperature" => (valuesT, VTKPointData()), "phi" => (valuesp, VTKPointData()))
write_vtk(dir * "myout/result_phaseField", 0, 0.0, mesh, dict_vars)

itime = 0
while time <= totalTime
    global time = time + dt
    global itime = itime + 1
    @show time, itime

    #qp[:] .= p0[:]*(1.0 .- p0[:])*(p0[:] .- 0.5 .+ m(T0)[:])

    LP = assembleRHS!(mesh, u, p0, T0)
    RHSP = (M * p0 + (dt / tau) * LP)
    for idof in bnd_dofs["West"]
        RHSP[idof] = 1.0
    end

    p1 = Matp \ RHSP

    RHST = (M * T0 + K * M * (p1 - p0))
    for idof in bnd_dofs["West"]
        RHST[idof] = 1.0
    end
    T1 = MatT \ RHST

    p0 .= p1
    T0 .= T1

    if itime % 100 == 0
        # write solution in vtk format
        set_values!(u, T1)
        valuesT = var_on_nodes(u) # the result is a dictionnary, but we are only interested in the values
        set_values!(u, p1)
        valuesp = var_on_nodes(u)

        dict_vars = Dict(
            "Temperature" => (valuesT, VTKPointData()),
            "phi" => (valuesp, VTKPointData()),
        )
        write_vtk(dir * "myout/result_phaseField", itime, time, mesh, dict_vars)
    end
end

end #hide
