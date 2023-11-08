module PhaseFieldSupercooled #hide
println("Running phase field supercooled equation example...") #hide

# # Phase field model - solidification of a liquid in supercooled state
# # Theory
# This case is taken from: Kobayashi, R. (1993). Modeling and numerical simulations of dendritic crystal growth. Physica D: Nonlinear Phenomena, 63(3-4), 410-423.
# In particular, the variables of the problem are denoted in the same way ($p$ for the phase indicator and $T$ for temperature).
# Consider a rectangular domain $$\Omega = [0, L_x] \times [0, L_y]$$ on which we wish to solve the following equations:
# ```math
#   \tau \partial_t p = \epsilon^2 \Delta p + p (1-p)(p - \frac{1}{2} + m(T))
# ```
# ```math
#   \partial_t T = \Delta T + K \partial_t p
# ```
# where $m(T) = \frac{\alpha}{\pi} atan \left[ \gamma (T_e - T) \right]$.
# This set of equations represents the solidification of a liquid in a supercooled state. Here $T$ is a dimensionless temperature and $p$ is the solid volume fraction.
# Lagrange finite elements are used to discretize both equations. Time marching is performed with a forward Euler scheme for the first equation and a backward Euler scheme for the second one.
#
# To initiate the solidification process, a Dirichlet boundary condition ($p=1$,$T=1$) is applied at $x=0$ ("West" boundary).
using Bcube
using LinearAlgebra
using Random
using WriteVTK
# using NLsolve
#using Profile
using Symbolics
using DifferentialEquations
using SparseDiffTools
#using InteractiveUtils
#using Cthulhu

Random.seed!(1234) # to obtain reproductible results

const dir = string(@__DIR__, "/../") # Bcube dir
const lx = 5.0
const ly = 3.0
const nx = 30
const ny = 30
const ε = 0.01 # original value 0.01
const τ = 0.0003
const α = 0.9
const γ = 10.0
const K = 1.6
const Te = 1.0
const β = 0.0 # noise amplitude, original value : 0.01
const totalTime = 1.0 # original value : 1
const nout = 50 # Number of iterations to skip before writing file

g(T) = (α / π) * atan(γ * (Te - T))

# Read mesh
# mesh = rectangle_mesh(nx, ny, xmin=0, xmax=lx, ymin=0, ymax=ly; bnd_names=("North", "South", "East", "West"))
# const mesh_path = dir * "myout/tmp.msh"
const mesh_path = dir * "input/mesh/domainPhaseField_tri.msh"
# gen_rectangle_mesh(mesh_path, :tri; transfinite=true, nx=nx, ny=ny, lx=lx, ly=ly)
const mesh = read_msh(mesh_path, 2)
# const mesh = line_mesh(10; names=("West", "East"))

# Noise function : random between [-1/2,1/2]
const χ = Bcube.MeshCellData(rand(ncells(mesh)) .- 0.5)

"""
Version with explicit - implicit time integration
We don't use a MultiFESpace here, only one FESpace -> the two variables can't be of different degrees
"""
function run_imex_1space()
    degree = 1

    # Function spaces and FE Spaces
    fs = FunctionSpace(:Lagrange, degree)
    U = TrialFESpace(fs, mesh, Dict("West" => (x, t) -> 1.0))
    V = TestFESpace(U)

    # FE functions
    ϕ = FEFunction(U)
    T = FEFunction(U)

    # Define measures for cell integration
    dΩ = Measure(CellDomain(mesh), 2 * degree + 1)

    # Bilinear and linear forms
    a(u, v) = ∫(∇(u) ⋅ ∇(v))dΩ
    m(u, v) = ∫(u ⋅ v)dΩ
    l(v) = ∫(v * ϕ * (1.0 - ϕ) * (ϕ - 0.5 + g(T) + β * χ))dΩ

    # Assemble matrices
    A = assemble_bilinear(a, U, V)
    M = assemble_bilinear(m, U, V)

    # Numerical parameters
    Δt = 0.0001

    # Create iterative matrices
    C_ϕ = M + Δt / τ * ε^2 * A
    C_T = M + Δt * A

    # Set Dirichlet conditions
    # For this example, we don't use a lifting method to impose the Dirichlet.
    d = Bcube.assemble_dirichlet_vector(U, V, mesh)
    Bcube.apply_dirichlet_to_matrix!((C_ϕ, C_T), U, V, mesh)

    # Factorize for performance
    C_ϕ = factorize(C_ϕ)
    C_T = factorize(C_T)

    # Init solution
    set_dof_values!(ϕ, d)
    set_dof_values!(T, d)

    dict_vars = Dict(
        "Temperature" => (var_on_vertices(T, mesh), VTKPointData()),
        "Phi" => (var_on_vertices(ϕ, mesh), VTKPointData()),
    )
    write_vtk(dir * "myout/result_phaseField_imex_1space", 0, 0.0, mesh, dict_vars)

    # Preallocate
    L = zero(d)
    rhs = zero(d)
    ϕ_new = zero(d)

    # Time loop
    t = 0.0
    itime = 0
    while t <= totalTime
        t += Δt
        itime += 1
        @show t, itime

        # Integrate equation on ϕ
        L .= 0.0 # reset L
        assemble_linear!(L, l, V)
        rhs .= M * get_dof_values(ϕ) .+ Δt / τ .* L
        Bcube.apply_dirichlet_to_vector!(rhs, U, V, mesh)
        ϕ_new .= C_ϕ \ rhs

        # Integrate equation on T
        rhs .= M * (get_dof_values(T) .+ K .* (ϕ_new .- get_dof_values(ϕ)))
        Bcube.apply_dirichlet_to_vector!(rhs, U, V, mesh)

        # Update solution
        set_dof_values!(ϕ, ϕ_new)
        set_dof_values!(T, C_T \ rhs)

        # write solution in vtk format
        if itime % nout == 0
            dict_vars = Dict(
                "Temperature" => (var_on_vertices(T, mesh), VTKPointData()),
                "Phi" => (var_on_vertices(ϕ, mesh), VTKPointData()),
            )
            write_vtk(
                dir * "myout/result_phaseField_imex_1space",
                itime,
                t,
                mesh,
                dict_vars;
                append = true,
            )
        end
    end

    println("End of imex 1 space")
end

"""
Version with explicit - implicit time integration
We don't use a MultiFESpace here, but two distincts FESpaces
"""
function run_imex_2spaces()
    degree_ϕ = 1
    degree_T = 1

    # Function spaces and FE Spaces
    fs_ϕ = FunctionSpace(:Lagrange, degree_ϕ)
    fs_T = FunctionSpace(:Lagrange, degree_T)
    U_ϕ = TrialFESpace(fs_ϕ, mesh, Dict("West" => (x, t) -> 1.0))
    U_T = TrialFESpace(fs_T, mesh, Dict("West" => (x, t) -> 1.0))
    V_ϕ = TestFESpace(U_ϕ)
    V_T = TestFESpace(U_T)

    # FE functions
    ϕ = FEFunction(U_ϕ)
    T = FEFunction(U_T)

    # Define measures for cell integration
    dΩ = Measure(CellDomain(mesh), 2 * max(degree_ϕ, degree_T) + 1)

    # Bilinear and linear forms
    a(u, v) = ∫(∇(u) ⋅ ∇(v))dΩ
    m(u, v) = ∫(u ⋅ v)dΩ
    l(v) = ∫(v * ϕ * (1 - ϕ) * (ϕ - 0.5 + g(T) + β * χ))dΩ

    # Assemble matrices
    A_ϕ = assemble_bilinear(a, U_ϕ, V_ϕ)
    A_T = assemble_bilinear(a, U_T, V_T)
    M_ϕ = assemble_bilinear(m, U_ϕ, V_ϕ)
    M_T = assemble_bilinear(m, U_T, V_T)

    # Numerical parameters
    Δt = 0.0001

    # Create iterative matrices
    C_ϕ = M_ϕ + Δt / τ * ε^2 * A_ϕ
    C_T = M_T + Δt * A_T

    # Set Dirichlet conditions
    ϕ_Γ = Bcube.assemble_dirichlet_vector(U_ϕ, V_ϕ, mesh)
    T_Γ = Bcube.assemble_dirichlet_vector(U_T, V_T, mesh)
    Bcube.apply_dirichlet_to_matrix!(C_ϕ, U_ϕ, V_ϕ, mesh)
    Bcube.apply_dirichlet_to_matrix!(C_T, U_T, V_T, mesh)

    # Factorize for performance
    C_ϕ = factorize(C_ϕ)
    C_T = factorize(C_T)

    # Init solution
    set_dof_values!(ϕ, ϕ_Γ)
    set_dof_values!(T, T_Γ)

    dict_vars = Dict(
        "Temperature" => (var_on_vertices(T, mesh), VTKPointData()),
        "Phi" => (var_on_vertices(ϕ, mesh), VTKPointData()),
    )
    write_vtk(dir * "myout/result_phaseField_imex_2spaces", 0, 0.0, mesh, dict_vars)

    # Preallocate
    L_ϕ = zero(ϕ_Γ)
    ϕ_new = zero(ϕ_Γ)
    rhs_ϕ = zero(ϕ_Γ)
    rhs_T = zero(T_Γ)

    # Time loop
    t = 0.0
    itime = 0
    while t <= totalTime
        t += Δt
        itime += 1
        @show t, itime

        # Integrate equation on ϕ
        L_ϕ .= 0.0 # reset
        assemble_linear!(L_ϕ, l, V_ϕ)
        rhs_ϕ .= M_ϕ * get_dof_values(ϕ) + Δt / τ * L_ϕ
        Bcube.apply_dirichlet_to_vector!(rhs_ϕ, U_ϕ, V_ϕ, mesh)
        ϕ_new .= C_ϕ \ rhs_ϕ

        # Integrate equation on T
        rhs_T .= M_T * (get_dof_values(T) .+ K .* (ϕ_new .- get_dof_values(ϕ)))
        apply_dirichlet_to_vector!(rhs_T, U_T, V_T, mesh)

        # Update solution
        set_dof_values!(ϕ, ϕ_new)
        set_dof_values!(T, C_T \ rhs_T)

        # write solution in vtk format
        if itime % nout == 0
            dict_vars = Dict(
                "Temperature" => (var_on_vertices(T, mesh), VTKPointData()),
                "Phi" => (var_on_vertices(ϕ, mesh), VTKPointData()),
            )
            write_vtk(
                dir * "myout/result_phaseField_imex_2spaces",
                itime,
                t,
                mesh,
                dict_vars;
                append = true,
            )
        end
    end

    println("End of imex 2 spaces")
end

"""
Full implicit resolution. This is just an experiment to test DifferentialEquations. The example fails after
a few iterations.
"""
function run_full_implicit()
    filepath = dir * "myout/result_phaseField_implicit"

    # Numerical parameters
    Δt = 0.0001
    degree_ϕ = 1
    degree_T = 1

    # Function spaces and FE Spaces
    fs_ϕ = FunctionSpace(:Lagrange, degree_ϕ)
    fs_T = FunctionSpace(:Lagrange, degree_T)
    U_ϕ = TrialFESpace(fs_ϕ, mesh, Dict("West" => (x, t) -> 1.0))
    U_T = TrialFESpace(fs_T, mesh, Dict("West" => (x, t) -> 1.0))
    V_ϕ = TestFESpace(U_ϕ)
    V_T = TestFESpace(U_T)

    # Multi-FE spaces
    U = MultiFESpace(U_ϕ, U_T)
    V = MultiFESpace(V_ϕ, V_T)

    # FE functions
    q = FEFunction(U)
    ϕ, T = get_fe_functions(q)

    # Define measures for cell integration
    dΩ = Measure(CellDomain(mesh), 2 * max(degree_ϕ, degree_T) + 1)

    # Bilinear and linear forms
    a((u_ϕ, u_T), (v_ϕ, v_T)) = ∫(ε^2 * ∇(u_ϕ) ⋅ ∇(v_ϕ) + ∇(u_T) ⋅ ∇(v_T))dΩ
    m((u_ϕ, u_T), (v_ϕ, v_T)) = ∫(τ * u_ϕ ⋅ v_ϕ + u_T ⋅ v_T - K * u_ϕ ⋅ v_T)dΩ
    l((ϕ, T), (v_ϕ, v_T)) = ∫(v_ϕ * ϕ * (1 - ϕ) * (ϕ - 0.5 + g(T) + β * χ))dΩ # need (ϕ, T) for DiffEqns

    # Assemble matrices
    A = assemble_bilinear(a, U, V)
    M = assemble_bilinear(m, U, V)

    # Set Dirichlet conditions on vectors and matrix
    Ud = Bcube.assemble_dirichlet_vector(U, V, mesh)
    Bcube.apply_homogeneous_dirichlet_to_matrix!(M, U, V, mesh)
    Bcube.apply_dirichlet_to_matrix!(A, U, V, mesh) # no "homogeneous" because no lifting

    # Init solution, and write to file
    set_dof_values!(q, Ud)
    dict_vars = Dict(
        "Temperature" => (var_on_vertices(T, mesh), VTKPointData()),
        "Phi" => (var_on_vertices(ϕ, mesh), VTKPointData()),
    )
    write_vtk(filepath, 0, 0.0, mesh, dict_vars)

    # Init solution
    Q0 = zeros(Bcube.get_ndofs(U))

    # Function to cancel via iterative solver
    p = (U = U, counter = [1])
    function f!(dQ, Q, p, t)
        # Unpack
        U = p.U

        # Update q (necessary to compute l(v))
        ϕ, T = FEFunction(U, Q)

        # Compute rhs
        L = zero(Q)
        _l = ((v_ϕ, v_T),) -> l((ϕ, T), (v_ϕ, v_T))
        assemble_linear!(L, _l, V)
        Bcube.apply_dirichlet_to_vector!(L, U, V, mesh)
        dQ .= L .- A * Q
    end

    # (Optional) compute jacobian function
    println("computing jacobian cache...")
    _f! = (y, x) -> f!(y, x, p, 0.0)
    output = zeros(Bcube.get_ndofs(U))
    input = zeros(Bcube.get_ndofs(U))
    sparsity_pattern = Symbolics.jacobian_sparsity(_f!, output, input)
    println("sparsity pattern computed !")
    jac = Float64.(sparsity_pattern)
    display(jac)
    colors = matrix_colors(jac)
    println("coloring done!")
    @show maximum(colors)
    # jac_cache = ForwardColorJacCache(_f!, input, nothing, dx=similar(input), colorvec=colors, sparsity=sparsity_pattern)
    # j! = (J, u, p, t) -> forwarddiff_color_jacobian!(J, (y, x) -> f!(y, x, p, t), u, jac_cache=jac_cache)

    # Use ModelingToolkit and/or DifferentialEquations to solve the problem
    println("Setting DiffEq...")
    tspan = (0.0, totalTime)
    # odeFunction = ODEFunction(f!; mass_matrix=M)
    odeFunction = ODEFunction(
        f!;
        mass_matrix = M,
        jac_prototype = sparsity_pattern,
        colorvec = colors,
    )
    prob = ODEProblem(odeFunction, Q0, tspan, p)
    always_true(args...) = true

    isoutofdomain(Q, p, t) = begin
        U = p.U
        ϕ, T = FEFunction(U, Q)
        cdt1 = any(v -> v < -0.0001, get_dof_values(ϕ))
        cdt2 = any(v -> v > 1.0001, get_dof_values(ϕ))
        cdt3 = any(Q .< -0.0001)

        # return cdt1 && cdt2
        return cdt3
    end

    cb_vtk = DiscreteCallback(
        always_true,
        integrator -> begin
            # Alias
            t = integrator.t
            counter = integrator.p.counter

            Δt = t - integrator.tprev
            println("i = $(counter[1]), t = $t, Δt = $(Δt)")

            if (counter[1] % nout == 0)
                set_dof_values!(q, integrator.u)
                dict_vars = Dict(
                    "Temperature" => (var_on_vertices(T, mesh), VTKPointData()),
                    "Phi" => (var_on_vertices(ϕ, mesh), VTKPointData()),
                )
                println("saving to vtk...")
                write_vtk(filepath, counter[1], t, mesh, dict_vars; append = true)
            end

            counter .+= 1
        end;
        save_positions = (false, false),
    )

    timestepper = ImplicitEuler(; nlsolve = NLNewton(; max_iter = 20))
    println("solving...")
    sol = solve(
        prob,
        timestepper;
        progress = false,
        callback = CallbackSet(cb_vtk),
        save_everystep = false,
        save_start = false,
        save_end = true,
        isoutofdomain = isoutofdomain,
    )
    # Write final result
    set_dof_values!(q, sol.u[end])
    dict_vars = Dict(
        "Temperature" => (var_on_vertices(T, mesh), VTKPointData()),
        "Phi" => (var_on_vertices(ϕ, mesh), VTKPointData()),
    )
    write_vtk(filepath, 1, totalTime, mesh, dict_vars; append = true)

    println("End of full implicit")
end

# run_imex_1space()
# run_imex_2spaces()
run_full_implicit()

end
