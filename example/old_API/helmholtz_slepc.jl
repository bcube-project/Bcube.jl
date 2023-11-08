module Helmholtz #hide
println("Running Helmholtz example...") #hide
# Remark : this example should be removed from this repository and moved to BcubeParallel
#
# # Helmholtz with SLEPc
# # Theory
# We consider the following Helmholtz equation, representing for instance the acoustic wave propagation with Neuman boundary condition(s):
# ```math
# \begin{cases}
#   \Delta u + \omega^2 u = 0 \\
#   \dfrac{\partial u}{\partial n} = 0 \textrm{  on  } \Gamma
# \end{cases}
# ```

# # Easiest solution
# Load the necessary packages (Bcube is loaded only if not already loaded)
const dir = string(@__DIR__, "/")
include(dir * "../src/Bcube.jl")
using .Bcube
using LinearAlgebra
using WriteVTK
using Printf
using SparseArrays
using MPI
using PetscWrap
using SlepcWrap
include(dir * "private/stability/util.jl")

# Settings
const with_nozzle = false
const write_eigenvectors = true
const write_mat = false
const out_dir = dir * "../myout/"

# Init Slepc to init MPI comm (to be improved, should be able to start Slepc from existing comm...)
SlepcInitialize("-eps_nev 50 -st_pc_factor_shift_type NONZERO -st_type sinvert -eps_view")
#SlepcInitialize("-eps_type lapack -eps_view")

# Get MPI infos
#MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm) + 1 # '+1' to start at 1
nprocs = MPI.Comm_size(comm)
isRoot = (rank == 1)

# Mesh
const mesh_path = out_dir * "mesh.msh"

#- Generate a 1D mesh : (un)comment
#spaceDim = 1; topoDim = 1
#isRoot && gen_line_mesh(mesh_path; nx = 4, npartitions = nprocs)

#- Generate a 2D mesh : (un)comment
#spaceDim = 2; topoDim = 2
#isRoot && gen_rectangle_mesh(mesh_path, :tri; nx = 2, ny = 2, lx = 1., ly = 1., xc = 0.5, yc = 0.5, npartitions = nprocs)

#- Generate a 3D mesh : (un)comment
spaceDim = 3
topoDim = 3
isRoot && gen_cylinder_mesh(mesh_path, 10.0, 30; npartitions = nprocs)

# Read mesh and partitions
MPI.Barrier(comm) # all procs must wait for the mesh to be built (otherwise they start reading a wrong mesh)
mesh = read_msh(mesh_path)
cell2part = read_partitions(mesh_path, topoDim)

# Next, create a scalar variable named `:u`. The Lagrange polynomial space is used here. By default,
# a "continuous" function space is created (by opposition to a "discontinuous" one). The order is set to `1`.
const degree = 1
fs = FunctionSpace(:Lagrange, degree)
fes = FESpace(fs, :continuous; size = 1) #  size=1 for scalar variable
ϕ = CellVariable(:ϕ, mesh, fes, ComplexF64)
system = System(ϕ)
nd = get_ndofs(system)
@show ndofs(ϕ)

# Partition
#part2dof, part2cell, dof2loc = partition(system, ncells(mesh), cell2part)
part2dof, part2cell, part2minmax, dof2loc, loc2dof, dof2glob, glob2dof =
    partition(system, ncells(mesh), cell2part)
my_dofs = part2dof[rank]
my_cells = part2cell[rank]
i_min, i_max = part2minmax[rank, :]
nd_loc = length(my_dofs)

# Create a `TestFunction`
λ = get_trial_function(ϕ)

# Define measures for cell and interior face integrations
dΩ = Measure(CellDomain(mesh, my_cells), 2) # no quadrature higher than 2 for Penta6...

# compute integrals
println("Computing integrals...")
_A = ∫(∇(λ) * transpose(∇(λ)))dΩ
_B = ∫(λ * transpose(λ))dΩ

# build julia sparse matrices from integration result
println("Assembling...")
# NOTE : we could directly fill Slepc matrices, but manipulating SparseArrays
# is more convenient for boundary conditions (if any)
function fun(IJV, i, j, v)
    (i_min <= dof2glob[i] <= i_max) && push!.(IJV, (dof2glob[i], dof2glob[j], v))
end # sparse with global numbering

IJV = (Int[], Int[], Float64[])
sizehint!.(IJV, nd_loc)
assemble((i, j, v) -> fun(IJV, i, j, v), _A, ((ϕ, ϕ),), system)
As = sparse(IJV...)

IJV = (Int[], Int[], Float64[])
sizehint!.(IJV, nd_loc)
assemble((i, j, v) -> fun(IJV, i, j, v), _B, ((ϕ, ϕ),), system)
Bs = sparse(IJV...)

# Boundary conditions
if with_nozzle
    println("Nozzle not implemented yet")
    MPI.Barrier(comm)
    SlepcFinalize()
end

# Convert to Slepc matrices
println("Converting to Petsc matrices...")
A = julia_sparse_to_petsc(As, nd, nd_loc, i_min)
B = julia_sparse_to_petsc(Bs, nd, nd_loc, i_min)

# Assemble PETSc mat
PetscWrap.assemble!(A, MAT_FINAL_ASSEMBLY)
PetscWrap.assemble!(B, MAT_FINAL_ASSEMBLY)

# Print mat to file for debug
if write_mat
    println("Writing matrices to file for debug...")
    mat2file(A, out_dir * "A_$nprocs.txt")
    mat2file(B, out_dir * "B_$nprocs.txt")
end

# Now we set up the eigenvalue solver
println("Creating EPS...")
eps = create_eps(A, B; auto_setup = true)

# Then we solve
println("Solving...")
solve!(eps)

# Retrieve eigenvalues
println("Number of converged eigenvalues : " * string(neigs(eps)))
i_eigs = 1:min(50, neigs(eps))
vp = get_eig(eps, i_eigs)

# Display the "first" eigenvalues:
@show sqrt.(abs.(vp[i_eigs]))

# Write result to ascii
const casename = "helmholtz_slepc_$(nprocs)"
println("Writing results to files")
println("Writing eigenvalues to '$(casename)'...")
eigenvalues2file(
    eps,
    out_dir * casename * "_vp.csv";
    two_cols = true,
    write_index = true,
    write_header = true,
    comment = "",
)
if write_eigenvectors
    println("Writing eigenvectors...")
    eigenvectors2file(eps, out_dir * casename * "_vecp")
    A_start, _ = get_range(A) # needed for eigenvectors2vtk
end

# Free memory
println("Destroying Petsc/Slepc objects")
destroy!.((A, B, eps))

# Convert to VTK
if write_eigenvectors
    println("Converting to VTK...")
    proc2start = MPI.Gather(A_start, 0, comm) # gather starting row of all procs

    # Only root proc converts everything to VTK
    if (isRoot)
        # Read real and imag parts of eigenvectors
        #TODO: (could use PetscViewerAsciiOpen...)
        vecs_r = readPetscAscii(out_dir * casename * "_vecp_r.dat")
        vecs_i = readPetscAscii(out_dir * casename * "_vecp_i.dat")
        println("Vectors successfully read!")

        # Now we need to reorder this vectors. We mainly "reverse" what is done in `julia_sparse_to_petsc`
        #vecs .= vecs[dof2glob,:] # reorder elements -> I don't understand why this is working : just luck?
        vecs = zeros(ComplexF64, size(vecs_r))
        for irank in 1:nprocs
            i_min = part2minmax[irank, 1]
            p2s = proc2start[irank] # first row handled by proc `irank`
            nd_rank = length(part2dof[irank]) # number of dofs handled by proc `irank`
            ind_rank = p2s:(p2s + nd_rank - 1) # row indices of `vecs_*` handled by proc `irank`
            ind_glob = ind_rank .- p2s .+ i_min
            vecs[glob2dof[ind_glob], :] = vecs_r[ind_rank, :] .+ vecs_i[ind_rank, :] .* im
        end

        # Finally convert to VTK
        eigenvectors2VTK(system, vecs, out_dir * casename * "_vecp", i_eigs)
        println("done writing " * out_dir * casename * "_vecp")
    end
end

# The end (the Barrier helps debugging)
println("processor $(rank)/$(nprocs) reached end of script")
MPI.Barrier(comm)
SlepcFinalize()

end #hide
