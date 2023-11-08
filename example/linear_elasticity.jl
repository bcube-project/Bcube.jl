module linear_elasticity #hide
println("Running linear elasticity API example...") #hide

# # Linear elasticity

const dir = string(@__DIR__, "/") # bcube/example dir
using Bcube
using LinearAlgebra
using WriteVTK
using StaticArrays

# function space (here we shall use Lagrange P1 elements) and quadrature degree.
const fspace = :Lagrange
const degree = 1 # FunctionSpace degree
const degquad = 2 * degree + 1

# Input and output paths
const outputpath = dir * "../myout/elasticity/"
const meshpath = dir * "../input/mesh/domainElast_tri.msh"

# Time stepping scheme params
const α = 0.05
const γ = 0.5 + α
const β = 0.25 * (1.0 + α)^2

# Material parameters (Young's modulus, Poisson coefficient and deduced Lamé coefficients)
const ρ = 2500.0
const E = 200.0e9
const ν = 0.3
const λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))
const μ = E / (2.0 * (1.0 + ν))

# Strain tensor and stress tensor (Hooke's law)
ϵ(u) = 0.5 * (∇(u) + transpose(∇(u)))
σ(u) = λ * tr(ϵ(u)) * I + 2 * μ * ϵ(u)

π(u, v) = σ(u) ⊡ ϵ(v) # with the chosen contraction convention ϵ should be transposed, but as it is symmetric the expression remains correct

# materialize for identity operator
Bcube.materialize(A::LinearAlgebra.UniformScaling, B) = A

# Function that runs the steady case:
function run_steady()
    # read mesh, the second argument specifies the spatial dimension
    mesh = read_msh(meshpath, 2)

    fs = FunctionSpace(fspace, degree)
    U_vec = TrialFESpace(
        fs,
        mesh,
        Dict("West" => SA[0.0, 0.0], "East" => SA[1.0, 0.0]);
        size = 2,
    )
    V_vec = TestFESpace(U_vec)

    # Define measures for cell
    dΩ = Measure(CellDomain(mesh), degquad)

    # no volume force term
    f = PhysicalFunction(x -> SA[0.0, 0.0])

    # definition of bilinear and linear forms
    a(u, v) = ∫(π(u, v))dΩ
    l(v) = ∫(f ⋅ v)dΩ

    # solve using AffineFESystem
    sys = Bcube.AffineFESystem(a, l, U_vec, V_vec)
    ϕ = Bcube.solve(sys)

    Un = var_on_vertices(ϕ, mesh)
    # Write the obtained FE solution
    dict_vars = Dict("Displacement" => (transpose(Un), VTKPointData()))
    mkpath(outputpath)
    write_vtk(outputpath * "result_elasticity", itime, t, mesh, dict_vars; append = false)
end

# Function that performs a time step using a Newmark α-HHT scheme
# The scheme updates the acceleration G, the velocity V and the displacement U using the following formulas:
#
# M G +(1-α)A U + αA U0 = (1-α) L + α L0 = L (because here L is time independent)
# V = V0 + (1-γ) Δt G0 + γ Δt G
# U = U0 + Δt V0 + (0.5-β)*Δt^2 G0 + β Δt^2 G
#
# G is then computed by solving the linear system obtained by inserting the expressions for U and V in the equation for G.
function Newmark_α_HHT(dt, L, A, Mat, U0, V0, G0)
    L1 = L - α * A * U0
    L2 = -(1.0 - α) * (A * U0 + dt * A * V0 + (0.5 - β) * dt * dt * A * G0)
    RHS = L1 .+ L2

    G = Mat \ RHS
    V = V0 + (1.0 - γ) * dt * G0 + γ * dt * G
    U = U0 + dt * V0 + (0.5 - β) * dt * dt * G0 + β * dt * dt * G

    return U, V, G
end

# Function that runs the unsteady case:
function run_unsteady()
    # read mesh, the second argument specifies the spatial dimension
    mesh = read_msh(meshpath, 2)

    fs = FunctionSpace(fspace, degree)
    U_vec = TrialFESpace(fs, mesh, Dict("West" => SA[0.0, 0.0]); size = 2)
    V_vec = TestFESpace(U_vec)

    # Define measures for cell
    dΩ = Measure(CellDomain(mesh), degquad)
    Γ = BoundaryFaceDomain(mesh, ("East",))
    dΓ = Measure(Γ, degquad)

    # surface force to be applied on East boundary
    f = PhysicalFunction(x -> SA[100000.0, 1000.0])

    # Definition of bilinear and linear forms
    a(u, v) = ∫(π(u, v))dΩ
    m(u, v) = ∫(ρ * u ⋅ v)dΩ
    l(v) = ∫(side⁻(f) ⋅ side⁻(v))dΓ

    # Assemble matrices and vector
    M = assemble_bilinear(m, U_vec, V_vec)
    A = assemble_bilinear(a, U_vec, V_vec)
    L = assemble_linear(l, V_vec)

    # Apply homogeneous dirichlet on A and b
    Bcube.apply_homogeneous_dirichlet_to_vector!(L, U_vec, V_vec, mesh)
    Bcube.apply_dirichlet_to_matrix!((A, M), U_vec, V_vec, mesh)

    # Initialize solution
    ϕ = FEFunction(U_vec, 0.0)
    U0 = zeros(Bcube.get_ndofs(U_vec))
    V0 = zeros(Bcube.get_ndofs(U_vec))
    G0 = zeros(Bcube.get_ndofs(U_vec))

    # Write initial solution
    Un = var_on_vertices(ϕ, mesh)
    # Write the obtained FE solution
    dict_vars = Dict("Displacement" => (transpose(Un), VTKPointData()))
    mkpath(outputpath)
    write_vtk(outputpath * "result_elasticity", 0, 0.0, mesh, dict_vars; append = false)

    # Time loop
    totalTime = 1.0e-3
    Δt = 1.0e-6
    itime = 0
    t = 0.0

    # Matrix for time stepping
    Mat = factorize(M + (1.0 - α) * (β * Δt * Δt * A))

    while t <= totalTime
        t += Δt
        itime = itime + 1
        @show t, itime

        # solve time step
        U, V, G = Newmark_α_HHT(Δt, L, A, Mat, U0, V0, G0)

        # Update solution
        U0 .= U
        V0 .= V
        G0 .= G

        set_dof_values!(ϕ, U)

        # Write solution
        if itime % 10 == 0
            Un = var_on_vertices(ϕ, mesh)
            # Write the obtained FE solution
            dict_vars = Dict("Displacement" => (transpose(Un), VTKPointData()))
            write_vtk(
                outputpath * "result_elasticity",
                itime,
                t,
                mesh,
                dict_vars;
                append = true,
            )
            # In order to use the warp function in paraview (solid is deformed using the displacement field)
            # the calculator filter has to be used with the following formula to reconstruct a 3D displacement field
            # with 0 z-component: Displacement_X*iHat+Displacement_Y*jHat+0.0*kHat
        end
    end
end

#run_steady()
run_unsteady()

end #hide
