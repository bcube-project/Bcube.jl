module TransportHypersurface #hide
println("Running transport on hypersurface example...") #hide

# Load the necessary packages
const dir = string(@__DIR__, "/../../") # Bcube dir
include(dir * "src/Bcube.jl")
using .Bcube
using LinearAlgebra
using WriteVTK

# Matrix-vector multiplication when matrix is dimension 1 and vector also...
Base.:*(a::Array{Float64, 1}, b::Array{Float64, 1}) = a .* b

# Upwind flux, nij is the local element normal
# This function is wrong : we should apply a "rotation" of the velocity in cell-j to
# bring it back in cell-i plane. But we assume here that the normal vector are coplanar.
function upwind(ϕᵢ, ϕⱼ, c, nᵢⱼ)
    vij = c ⋅ nᵢⱼ
    if vij > zero(vij)
        vij * ϕᵢ
    else
        vij * ϕⱼ
    end
end

function inv_mass_matrix(λ, cnodes, ct, order)
    M = integrate_ref(ξ -> ⊗(λ(ξ)), cnodes, ct, order)
    return inv(M)
end

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
        ctype = cellTypes[icell]

        ## Alias for nodes
        cnodes = get_nodes(mesh, c2n[icell])

        ## Get shape functions in the reference element
        λ = shape_functions(fs, shape(ctype))

        ## Compute inverse of mass matrix
        inv_matrix[icell] .= inv_mass_matrix(λ, cnodes, ctype, orderM)
    end
    return inv_matrix
end

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
        F = mapping(fnodes, ftype) # Face mapping : face-ref coords -> local coords

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
        shapeⱼ = shape(ctⱼ)
        λⱼ = shape_functions(fs, shape(ctⱼ))
        ϕⱼ = interpolate(λⱼ, q[dof(ϕ, j)])
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

        ## Flux definition from `i` to `j` : using upwind
        fluxn = (nᵢⱼ, ξ) -> upwind(ϕᵢ(fpᵢ(ξ)), ϕⱼ(fpⱼ(ξ)), c(F(ξ)), nᵢⱼ) # Note the use of F

        ## Append flux contribution of face `kface` to cell `i`, performing a surfacic integration
        # ξ is the face-ref-element
        g_ref = ξ -> fluxn(normal(xᵢ, ctᵢ, sideᵢ, ξ), ξ) * λᵢ(fpᵢ(ξ))
        f[dof(ϕ, i)] += integrate_ref(g_ref, fnodes, ftype, order)

        ## Append flux contribution of face `kface` to cell `j`
        g_ref = ξ -> fluxn(-normal(xⱼ, ctⱼ, sideⱼ, ξ), ξ) * λⱼ(fpⱼ(ξ))
        f[dof(ϕ, j)] -= integrate_ref(g_ref, fnodes, ftype, order)
    end

    ## Loop on all the boundary of type 'faces'
    for tag in keys(mesh.bc_faces)
        ## Loop over this boundary faces
        for kface in boundary_faces(mesh, tag)

            ## Face nodes, type and shape
            ftype = faceTypes[kface]
            fnodes = get_nodes(mesh, f2n[kface])
            F = mapping(fnodes, ftype) # mapping face-ref coords -> local coords

            ## Neighbor cell i
            i = f2c[kface][1]
            cnodes = get_nodes(mesh, c2n[i])
            ct = cellTypes[i]
            s = shape(ct)
            λ = shape_functions(fs, shape(ct))
            ϕᵢ = interpolate(λ, q[dof(ϕ, i)])
            side = cell_side(ct, c2n[i], f2n[kface])
            fp = mapping_face(s, side)

            ## For a multi-variable problem, we should loop over the variables. Here we have only one variable
            ## Skip if no boundary condition on this boundary
            !haskey(params.cdts, tag) && continue

            ## Get associated boundary condition
            cdt = params.cdts[tag]

            ## Flux boundary condition
            if type(cdt) == :flux
                ## Append flux contribution of face `kface` to cell `i`
                g_ref = ξ -> normal(cnodes, ct, side, ξ) ⋅ apply(cdt, F(ξ), t) * λ(fp(ξ))
                f[dof(ϕ, i)] += integrate_ref(g_ref, fnodes, ftype, order)

                ## Dirichlet boundary condition : we apply classic flux with imposed condition in ghost. We split in
                ## three steps to improve clarity, but this can be done in one line.
            elseif type(cdt) == :diri
                ϕ_bnd = ξ -> apply(cdt, F(ξ), t)
                fluxn = (n, ξ) -> upwind(ϕᵢ(fp(ξ)), ϕ_bnd(ξ), c(F(ξ)), n)
                g_ref = ξ -> fluxn(normal(cnodes, ct, side, ξ), ξ) * λ(fp(ξ))
                f[dof(ϕ, i)] += integrate_ref(g_ref, fnodes, ftype, order)
            end
        end
    end

    ## Result : a vector of size `ndofs`
    return f
end

function explicit_step!(mesh, ϕ, params, dq, q, t)

    ## Get inverse of mass matrix from cache
    inv_mass_matrix = build_mass_matrix_inv(mesh, function_space(ϕ), params)

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
        FV = zeros(ndofs(ϕ, icell)) # Volumic flux

        ## Indices of all the variable dofs in this cell
        i_ϕ = dof(ϕ, icell)

        ## Alias for cell type
        ctype = cellTypes[icell]

        ## Alias for nodes
        cnodes = get_nodes(mesh, c2n[icell])

        ## Get reference shape functions
        λ = shape_functions(fs, shape(ctype))

        ## Compute shape functions gradient
        ∇λ = grad_shape_functions(fs, ctype, cnodes)

        ## Create interpolation function
        ϕᵢ = interpolate(λ, q[i_ϕ])

        ## Loop over cell dofs for volumic term
        FV .= integrate_ref(
            ξ -> ∇λ(ξ) * (c(mapping(cnodes, ctype, ξ)) .* ϕᵢ(ξ)),
            cnodes,
            ctype,
            orderF,
        )

        ## Assemble
        dq[i_ϕ] .= inv_mass_matrix[icell] * (FV .- f[i_ϕ])
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
function append_vtk(vtk, mesh, ϕ, q, t)
    ## Values on center
    set_values!(ϕ, q)
    values = var_on_centers(ϕ)

    ## Write
    dict_vars = Dict("ϕ" => (values, VTKCellData()))
    write_vtk(vtk.basename, vtk.ite, t, mesh, dict_vars; append = vtk.ite > 0)

    ## Update counter
    vtk.ite += 1
end

function solve!(mesh, ϕ, params, dq, q)
    # Alias
    nite = params.nite
    nout = params.nout
    Δt = params.Δt

    # Init time
    t = 0.0

    # Solution at t = 0
    vtk = VtkHandler(params.name)
    append_vtk(vtk, mesh, ϕ, q, t)

    # Let's loop to solve the equation.
    for i in 1:nite
        ## Infos
        println("Iteration ", i)

        ## Step forward in time
        explicit_step!(mesh, ϕ, params, dq, q, t)
        q .+= Δt * dq
        t += Δt

        ## Write solution to file (respecting max. number of output)
        if (i % Int(max(floor(nite / nout), 1)) == 0)
            append_vtk(vtk, mesh, ϕ, q, t)
        end
    end
end

# Define horizontal line mesh
nx = 10
xmin = 1.0
xmax = 4.0
mesh = line_mesh(nx; xmin, xmax)

# Augment space dimension, rotate and translate
A(x) = [x[1], 0]
R(θ) = [
    cos(θ) -sin(θ)
    sin(θ) cos(θ)
]
T(u, t) = u + t

# Rotage and translate mesh
θ = π / 4
t⃗ = [1.0, 2.0]
f(x) = T(R(θ) * A(x), t⃗)
mesh = transform(mesh, f)

# Function space and var
fes = FESpace(FunctionSpace(:Taylor, 0), :discontinuous)
ϕ = CellVariable(:phi, mesh, fes)

# Analytic velocity
c(x) = 1.0 * [cos(θ), sin(θ)]

# Boundary condition
#cdt_in = BoundaryCondition(:flux, (x, t) -> 0. * c(x))
cdt_in = BoundaryCondition(:diri, (x, t) -> 0.0)
cdt_out = BoundaryCondition(:diri, (x, t) -> 0.0)
cdts = Dict(boundary_tag(mesh, "LEFT") => cdt_in, boundary_tag(mesh, "RIGHT") => cdt_out)

# Time step
Δt = (xmax - xmin) / (nx - 1)

# Then we create a `NamedTuple` to hold the simulation parameters
params = (
    name = dir * "myout/transport_on_inclined_line",
    nout = nx,
    nite = nx,
    Δt = Δt,
    order = 2,
    c = c,
    cdts = cdts,
)

# Let's allocate the unknown vector and set it to zero. Along with this vector, we also allocate the "increment" vector.
q = zeros(ndofs(ϕ))
dq = zeros(size(q))

# Initial solution
q[1] = 1.0

# Solve
solve!(mesh, ϕ, params, dq, q)
@show q

# Define circle mesh
nx = 10
radius = 1.0
mesh = circle_mesh(nx; radius = radius, order = 2)

# Function space and var
fes = FESpace(FunctionSpace(:Taylor, 0), :discontinuous)
ϕ = CellVariable(:ϕ, mesh, fes)

# Analytic velocity
c(x) = 1.0 * [-x[2], x[1]] / radius

# Time step
Δt = 2 * π * radius / nx

# Then we create a `NamedTuple` to hold the simulation parameters
params = (
    name = dir * "myout/transport_on_circle",
    nout = nx,
    nite = nx,
    Δt = Δt,
    order = 2,
    c = c,
    cdts = Dict(),
)

# Let's allocate the unknown vector and set it to zero. Along with this vector, we also allocate the "increment" vector.
q = zeros(ndofs(ϕ))
dq = zeros(size(q))

# Initial solution
q[1] = 1.0

# Solve
solve!(mesh, ϕ, params, dq, q)
@show q

# Settings
r = 1.0
nite = 75
nout = 100

# Read mesh of a sphere
mesh = read_msh(dir * "input/mesh/sphere_2.msh")

# Function space and var
fes = FESpace(FunctionSpace(:Taylor, 0), :discontinuous)
ϕ = CellVariable(:ϕ, mesh, fes)

# Analytic velocity
# x = r sin(\theta)cos(\phi)
# y = r sin(\theta)sin(\phi)
# z = r cos(\theta)
# u_r = sin(\theta)cos(\phi) ex + sin(\theta)sin(\phi) ey + cos(\theta) ez
# u_\theta = cos(\theta)cos(\phi) ex + cos(\theta)sin(\phi) ey - sin(\theta) ez
# u_\phi = -sin(\phi) ex + cos(\phi) ey
# We want the velocity to be u_\theta
# sin(theta) = x / (r cos(phi))
# cos(theta) = z / r
φ = 0.0
c(x) = 1.0 * [x[3] / r * cos(φ), x[3] / r * sin(φ), -x[1] / (r * cos(φ))]

# Time step
Δt = 0.04

# Then we create a `NamedTuple` to hold the simulation parameters
params = (
    name = dir * "myout/transport_on_sphere",
    nout = nout,
    nite = nite,
    Δt = Δt,
    order = 2,
    c = c,
    cdts = Dict(),
)

# Let's allocate the unknown vector and set it to zero. Along with this vector, we also allocate the "increment" vector.
q = zeros(ndofs(ϕ))
dq = zeros(size(q))

# Initial solution
q[1] = 1.0

# Solve
solve!(mesh, ϕ, params, dq, q)

end #hide
