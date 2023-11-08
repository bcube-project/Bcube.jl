module Poisson #hide
println("Running Poisson example...") #hide

# # Poisson
# # Theory
# Consider a square domain $$\Omega$$ on which we wish to solve Poisson's equation:
# ```math
#   -\Delta u = f
# ```
# This equation represents for instance steady state heat conduction with a given source term $$f$$ and a thermal conductivity of $$1 \, W.K^{-1}.m^{-1}$$.
# For the problem to be well-posed, boundary conditions also have to be specified.
# The "West" boundary will be noted $$\Gamma_w$$ and the "East" boundary will be noted $$\Gamma_e$$.
# We shall consider here two cases:
#
# (1) Homogeneous Dirichlet condition ($$u=0$$)  on $$\Gamma_w$$ and  homogeneous Neumann ($$\nabla u.n = 0$$) on the rest of $$\partial \Omega$$.
#
# (2) Homogeneous Dirichlet condition ($$u=0$$)  on $$\Gamma_w$$ and non-homogeneous Dirichlet on $$\Gamma_e$$
#
#
# We shall assume that $$f \in L^2(\Omega)$$. The weak form of the problem is given by: find $$ u \in \tilde{H}^1_0(\Omega)$$
# such that:
# ```math
#   \forall v \in  \tilde{H}^1_0(\Omega), \, \, \, \underbrace{\int_\Omega \nabla u . \nabla v dx}_{a(u,v)} = \underbrace{\int_\Omega f v dx}_{l(v)}
# ```
# To numerically solve this problem we seek an approximate solution using Lagrange $$P^1$$ or $$P^2$$ elements.

const dir = string(@__DIR__, "/../../") # Bcube dir
include(dir * "src/Bcube.jl")
using .Bcube
using LinearAlgebra
using WriteVTK
using Printf

# Read mesh
mesh = read_msh(dir * "input/mesh/domainSquare_tri.msh", 2)

# Function space
fs = FunctionSpace(:Lagrange, 2)

# Create a `Variable`
fes = FESpace(fs, :continuous)
u = CellVariable(:u, mesh, fes)

# Allocate the problem matrices and RHS
nd = ndofs(u)
A = zeros(Float64, (nd, nd))
L = zeros(Float64, nd)

function assemble!(mesh, u, A, L)
    ## Get connectivity and cell types
    c2n = connectivities_indices(mesh, :c2n)
    cellTypes = cells(mesh)

    ## Retrieve function space
    fs = function_space(u)

    ## Compute needed quadrature orders
    orderA = Val(2 * (get_order(fs) - 1) + 1)
    orderL = Val(get_order(fs) + 1)

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
                A[dof(u, icell, 1, i), dof(u, icell, 1, j)] +=
                    integrate_ref(ξ -> ∇λ(ξ)[i, :] ⋅ ∇λ(ξ)[j, :], n, ct, orderA)
            end
            L[dof(u, icell, 1, i)] += integrate_ref(ξ -> λ(ξ)[i], n, ct, orderL)
        end
    end
end

# Function to get the indices of the dofs located on the boundary (this has meaning for Lagrange elements)
function generateBoundaryDofs!(mesh, u)

    ## Get cell -> node, face -> node and face -> cell connectivities
    c2n = connectivities_indices(mesh, :c2n)
    f2n = connectivities_indices(mesh, :f2n)
    f2c = connectivities_indices(mesh, :f2c)

    ## Cell and face types
    cellTypes = cells(mesh)
    faceTypes = faces(mesh)

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
assemble!(mesh, u, A, L)

# generate boundary dofs:
bnd_dofs = generateBoundaryDofs!(mesh, u)
@show bnd_dofs["West"]
# here a homogeneous Dirichlet boundary condition is applied on "West". The value of u is imposed to 0.
for idof in bnd_dofs["West"]
    A[idof, :] .= 0.0
    A[:, idof] .= 0.0
    A[idof, idof] = 1.0
    L[idof] = 0.0
end

# solve the linear system
sol = A \ L

# write solution in vtk format
set_values!(u, sol)
values = var_on_nodes(u)

dict_vars = Dict(@sprintf("sol") => (values, VTKPointData()))
write_vtk(dir * "myout/result_poisson_homogeneousDirichlet", 0, 0.0, mesh, dict_vars)

# here we add a non-homogeneous Dirichlet boundary condition on "East" and solve the problem again. The value of u is imposed to 1.
ud = zeros(Float64, nd)
ud[bnd_dofs["East"]] .= 1.0
L = L - A * ud
for idof in bnd_dofs["East"]
    A[idof, :] .= 0.0
    A[:, idof] .= 0.0
    A[idof, idof] = 1.0
    L[idof] = 1.0
end

sol = A \ L

set_values!(u, sol)
values .= var_on_nodes(u) # the result is a dictionnary, but we are only interested in the values

dict_vars = Dict(@sprintf("sol") => (values, VTKPointData()))
write_vtk(dir * "myout/result_poisson_nonHomogeneousDirichlet", 0, 0.0, mesh, dict_vars)

end #hide
