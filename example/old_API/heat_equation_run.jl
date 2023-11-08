module heat_equation_API #hide
println("Running heat equation API example...") #hide
# # Heat equation
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

const dir = string(@__DIR__, "/../") # Bcube dir
using Bcube
using LinearAlgebra
using WriteVTK
using Printf
using SparseArrays
using DelimitedFiles
using BenchmarkTools
using Profile
using StaticArrays
using Cthulhu

# Function that defines the bilinear form a:
function f1(u, v, η)
    λ, = v
    return η .* ∇(λ) * transpose(∇(λ))
end

# Function that defines the bilinear form m:
function f2(u, v, params)
    λ, = v
    return params.ρCp * λ * transpose(λ)
end

# Function that defines the linear form l:
function f3(u, v, q)
    λ, = v
    return q * λ
end

# Function that defines the bilinear par of a Fourier-Robin condition:
function f4(u, v, params)
    λ, = v
    return params.htc * λ * transpose(λ)
end

# Function that defines the linear par of a Fourier-Robin condition:
function f5(u, v, params)
    λ, = v
    return (params.htc * params.Tr + params.phi) * λ
end

function f_Dirichlet(ud, v, params)
    λ, = v
    φ, = ud
    return params.η * ∇(φ) * transpose(∇(λ))
end

function T_analytical_twoLayer(x, lam1, lam2, T0, T1, L)
    if x < 0.5 * L
        T = 2.0 * (lam2 / (lam1 + lam2)) * ((T1 - T0) / L) * x + T0
    else
        T = 2.0 * (lam1 / (lam1 + lam2)) * ((T1 - T0) / L) * (x - L) + T1
    end
    return T
end

# convective boundary condition
const htc = 100.0
const Tr = 268.0
const phi = 100.0

const degree = 2
const outputpath = string(@__DIR__, "/../myout/heat_equation/")

# Function that runs the steady single layer case:
function run_steady()
    println("Running steady single layer case")

    # Read mesh
    mesh = read_msh(dir * "input/mesh/domainSquare_tri.msh", 2)

    # Build function space and associated Trial and Test FE spaces.
    # We impose a Dirichlet condition with a temperature of 260K
    # on boundary "West"
    fs = FunctionSpace(:Lagrange, degree)
    U = TrialFESpace(fs, mesh, Dict("West" => 260.0))
    V = TestFESpace(U)

    # Define measures for cell integration
    dΩ = Measure(CellDomain(mesh), 2 * degree + 1)

    # Physical parameters
    q = 1500.0
    λ = 100.0
    η = 100.0

    # Define bilinear and linear forms
    a(u, v) = ∫(η * ∇(u) ⋅ ∇(v))dΩ
    l(v) = ∫(q * v)dΩ

    # Create an affine FE system and solve it. The result is a FEFunction.
    # We can interpolate it on mesh centers
    sys = Bcube.AffineFESystem(a, l, U, V)
    ϕ = Bcube.solve(sys)
    Tcn = var_on_centers(ϕ, mesh)

    # Compute analytical solution for comparison. Apply the analytical solution
    # on mesh centers
    T_analytical = x -> 260.0 + (q / λ) * x[1] * (1.0 - 0.5 * x[1])
    Tca = map(T_analytical, Bcube.get_cell_centers(mesh))

    # Write both the obtained FE solution and the analytical solution.
    mkpath(outputpath)
    dict_vars =
        Dict("Temperature" => (Tcn, VTKCellData()), "Temperature_a" => (Tca, VTKCellData()))
    write_vtk(outputpath * "result_steady_heat_equation", 0, 0.0, mesh, dict_vars)

    # Compute and display the error
    @show norm(Tcn .- Tca, Inf) / norm(Tca, Inf)
end

# Function that runs the unsteady single layer case:
function run_unsteady()
    println("Running unsteady single layer case")

    # Read mesh
    mesh = read_msh(dir * "input/mesh/domainSquare_tri_2.msh", 2)

    # Build function space and associated Trial and Test FE spaces.
    # We impose a Dirichlet condition with a temperature of 260K
    # on boundary "West"
    fs = FunctionSpace(:Lagrange, degree)
    U = TrialFESpace(fs, mesh, Dict("West" => 260.0))
    V = TestFESpace(U)

    # Define measures for cell integration
    dΩ = Measure(CellDomain(mesh), 2 * degree + 1)

    # Physical parameters
    q = 1500.0
    λ = 150.0
    ρCp = 100.0 * 200.0
    η = λ
    totalTime = 100.0

    # Numerical parameters
    Δt = 0.1

    # Compute matrices associated to bilinear and linear forms
    a(u, v) = ∫(η * ∇(u) ⋅ ∇(v))dΩ
    m(u, v) = ∫(ρCp * u ⋅ v)dΩ
    l(v) = ∫(q * v)dΩ

    # Assemble
    A = assemble_bilinear(a, U, V)
    M = assemble_bilinear(m, U, V)
    L = assemble_linear(l, V)

    # Compute a vector of dofs whose values are zeros everywhere
    # except on dofs lying on a Dirichlet boundary, where they
    # take the Dirichlet value
    Ud = Bcube.assemble_dirichlet_vector(U, V, mesh)

    # Apply lift
    L = L - A * Ud

    # Apply homogeneous dirichlet condition
    Bcube.apply_homogeneous_dirichlet_to_vector!(L, U, V, mesh)
    Bcube.apply_dirichlet_to_matrix!((A, M), U, V, mesh)

    # Form time iteration matrix
    # (note that this is bad for performance since up to now,
    # M and A are sparse matrices)
    Miter = factorize(M + Δt * A)

    # Init solution
    ϕ = FEFunction(U, 260.0)

    # Write initial solution
    mkpath(outputpath)
    dict_vars = Dict("Temperature" => (var_on_centers(ϕ, mesh), VTKCellData()))
    write_vtk(outputpath * "result_unsteady_heat_equation", 0, 0.0, mesh, dict_vars)

    # Time loop
    itime = 0
    t = 0.0
    while t <= totalTime
        t += Δt
        itime = itime + 1
        @show t, itime

        # Compute rhs
        rhs = Δt * L + M * (get_dof_values(ϕ) .- Ud)

        # Invert system and apply inverse shift
        set_dof_values!(ϕ, Miter \ rhs .+ Ud)

        # Write solution
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
end

# Function that runs the steady two layer case:
function run_steady_twoLayer()
    println("Running steady two layer case")
    # Read mesh
    with_cell_names = true
    mesh, el_names, el_names_inv, el_cells, glo2loc_cell_indices =
        read_msh_with_cell_names(dir * "input/mesh/domainTwoLayer_tri.msh", 2)

    fs = FunctionSpace(:Lagrange, degree)
    U = TrialFESpace(fs, mesh)
    V = TestFESpace(U)

    ϕ = CellVariable(:ϕ, mesh, fes)
    # φ = CellVariable(:φ, mesh, fes)

    # Create a `TestFunction`
    # λ = TestFunction(mesh, fes)

    # u, v = ((ϕ,), (λ,))
    # ud = (φ,)

    # Define measures for cell and interior face integrations
    dΩ = Measure(CellDomain(mesh), 2 * degree + 1)

    nd = get_ndofs(U)

    T0 = 260.0
    bnd_dofs0 = boundary_dofs(ϕ, "West")
    # here a Dirichlet boundary condition is applied on "West". p and T are imposed to 1 (solid).
    Ud = zeros(Float64, (nd))
    for idof in bnd_dofs0
        Ud[idof] = T0
    end

    T1 = 300.0
    bnd_dofs1 = boundary_dofs(ϕ, "East")
    # here a Dirichlet boundary condition is applied on "West". p and T are imposed to 1 (solid).
    for idof in bnd_dofs1
        Ud[idof] = T1
    end

    # set_values!(φ, Ud) # mbouyges -> I believe this is a mistake in the original tutorial

    #Adense = zeros(Float64, (nd,nd))
    #Mdense = zeros(Float64, (nd,nd))
    L = zeros(Float64, (nd))

    lamda1 = 150.0
    lamda2 = 10.0

    mat_1 = el_cells[el_names_inv["Domain_1"]]
    mat_2 = el_cells[el_names_inv["Domain_2"]]

    rho = zeros(Float64, (ncells(mesh)))
    cp = zeros(Float64, (ncells(mesh)))
    lamda = zeros(Float64, (ncells(mesh)))
    rhoCp = zeros(Float64, (ncells(mesh)))

    for i in mat_1
        rho[glo2loc_cell_indices[i]] = 2000.0
        cp[glo2loc_cell_indices[i]] = 1000.0
        lamda[glo2loc_cell_indices[i]] = lamda1
        rhoCp[glo2loc_cell_indices[i]] =
            rho[glo2loc_cell_indices[i]] * cp[glo2loc_cell_indices[i]]
    end
    for i in mat_2
        rho[glo2loc_cell_indices[i]] = 2500.0
        cp[glo2loc_cell_indices[i]] = 900.0
        lamda[glo2loc_cell_indices[i]] = lamda2
        rhoCp[glo2loc_cell_indices[i]] =
            rho[glo2loc_cell_indices[i]] * cp[glo2loc_cell_indices[i]]
    end

    qtmp = zeros(Float64, (ncells(mesh)))

    q = CellData(qtmp)
    ρCp = CellData(rhoCp)
    η = CellData(lamda)

    params = (q = q, ρCp = ρCp, η = η)
    #params = (q = qtmp, ρCp= rhoCp, η=lamda, htc=htc, Tr=Tr, phi=phi)

    # compute matrices associated to bilinear and linear forms
    a(u, v) = ∫(params.η * ∇(u) ⋅ ∇(v))dΩ
    A = assemble_bilinear(a, U, V)
    # _A = ∫(f1(u, v, params),)dΩ
    #_Ad = ∫( f_Dirichlet(ud, v, params) )dΩ
    l(v) = ∫(params.q * v)dΩ
    L = assemble_linear(l, V)
    # _L = ∫(f3(u, v, params),)dΩ

    # A = sparse(_A, ϕ)

    # for (ic, val) in result(_L)
    #     idof = get_dof(ϕ, ic)
    #     ndof = length(idof)

    #     for i in 1:ndof
    #         L[idof[i]] += val[i]
    #     end
    # end

    nodes = get_nodes(mesh)
    L = L - A * Ud

    # here a Dirichlet boundary condition is applied on "West". p and T are imposed to 1 (solid).
    for idof in bnd_dofs0
        A[idof, :] .= 0.0
        A[:, idof] .= 0.0
        A[idof, idof] = 1.0
        L[idof] = Ud[idof]
    end

    for idof in bnd_dofs1
        A[idof, :] .= 0.0
        A[:, idof] .= 0.0
        A[idof, idof] = 1.0
        L[idof] = Ud[idof]
    end

    b = A \ L
    set_values!(ϕ, b)

    Ta =
        CellVariable(:Ta, mesh, FESpace(FunctionSpace(:Lagrange, 1), :continuous; size = 1))
    T_analytical = X -> T_analytical_twoLayer(X[1], lamda1, lamda2, T0, T1, 0.2)
    set_values!(Ta, T_analytical; degquad = 5)

    Tcn = var_on_centers(ϕ)
    Tca = var_on_centers(Ta)
    #dict_vars = Dict(@sprintf("Temperature") => (get_values(ϕ), VTKPointData()))
    dict_vars =
        Dict("Temperature" => (Tcn, VTKCellData()), "Temperature_a" => (Tca, VTKCellData()))
    write_vtk(dir * "myout/result_steady_heat_equation_two_layer", 0, 0.0, mesh, dict_vars)

    if degree == 1
        @show norm(get_values(ϕ) .- get_values(Ta), Inf) / norm(get_values(Ta), Inf)
    else
        @show norm(Tca .- Tcn, Inf) / norm(Tca, Inf)
    end
end

# Function that runs the unsteady two layer case:
function run_unsteady_twoLayer()
    println("Running unsteady two layer case")
    # Read mesh
    with_cell_names = true
    mesh, el_names, el_names_inv, el_cells, glo2loc_cell_indices =
        read_msh_with_cell_names(dir * "input/mesh/domainTwoLayer_quad.msh", 2)

    fs = FunctionSpace(:Lagrange, degree)
    fes = FESpace(fs, :continuous; size = 1) #  size=1 for scalar variable
    ϕ = CellVariable(:ϕ, mesh, fes)
    φ = CellVariable(:φ, mesh, fes)

    # Create a `TestFunction`
    λ = TestFunction(mesh, fes)

    u, v = ((ϕ,), (λ,))
    ud = (φ,)

    # Define measures for cell and interior face integrations
    dΩ = Measure(CellDomain(mesh), 2 * degree + 1)

    nd = length(get_values(ϕ))

    T0 = 0.0
    bnd_dofs0 = boundary_dofs(ϕ, "West")
    # here a Dirichlet boundary condition is applied on "West". p and T are imposed to 1 (solid).
    Ud = zeros(Float64, (nd))
    for idof in bnd_dofs0
        Ud[idof] = T0
    end

    T1 = 1.0
    bnd_dofs1 = boundary_dofs(ϕ, "East")
    # here a Dirichlet boundary condition is applied on "West". p and T are imposed to 1 (solid).
    for idof in bnd_dofs1
        Ud[idof] = T1
    end

    set_values!(φ, Ud)

    #Adense = zeros(Float64, (nd,nd))
    #Mdense = zeros(Float64, (nd,nd))
    L = zeros(Float64, (nd))

    lamda1 = 1.0
    lamda2 = 1.0

    mat_1 = el_cells[el_names_inv["Domain_1"]]
    mat_2 = el_cells[el_names_inv["Domain_2"]]

    qtmp = zeros(Float64, (ncells(mesh)))
    rho = zeros(Float64, (ncells(mesh)))
    cp = zeros(Float64, (ncells(mesh)))
    lamda = zeros(Float64, (ncells(mesh)))
    rhoCp = zeros(Float64, (ncells(mesh)))

    for i in mat_1
        rho[glo2loc_cell_indices[i]] = 100000.0
        cp[glo2loc_cell_indices[i]] = 1.0
        lamda[glo2loc_cell_indices[i]] = lamda1
        rhoCp[glo2loc_cell_indices[i]] =
            rho[glo2loc_cell_indices[i]] * cp[glo2loc_cell_indices[i]]
    end
    for i in mat_2
        rho[glo2loc_cell_indices[i]] = 100000.0
        cp[glo2loc_cell_indices[i]] = 1.0
        lamda[glo2loc_cell_indices[i]] = lamda2
        rhoCp[glo2loc_cell_indices[i]] =
            rho[glo2loc_cell_indices[i]] * cp[glo2loc_cell_indices[i]]
    end

    q = CellData(qtmp)
    ρCp = CellData(rhoCp)
    η = CellData(lamda)

    params = (q = q, ρCp = ρCp, η = η)

    # compute matrices associated to bilinear and linear forms
    _A = ∫(f1(u, v, params))dΩ
    _M = ∫(f2(u, v, params))dΩ
    #_Ad = ∫( f_Dirichlet(ud, v, params) )dΩ
    _L = ∫(f3(u, v, params))dΩ

    A = sparse(_A, ϕ)
    M = sparse(_M, ϕ)

    for (ic, val) in result(_L)
        idof = get_dof(ϕ, ic)
        ndof = length(idof)

        for i in 1:ndof
            L[idof[i]] += val[i]
        end
    end

    L = L - A * Ud

    # here a Dirichlet boundary condition is applied on "West". p and T are imposed to 1 (solid).
    for idof in bnd_dofs0
        A[idof, :] .= 0.0
        A[:, idof] .= 0.0
        A[idof, idof] = 1.0
        M[idof, :] .= 0.0
        M[:, idof] .= 0.0
        M[idof, idof] = 1.0
        L[idof] = Ud[idof]
    end
    for idof in bnd_dofs1
        A[idof, :] .= 0.0
        A[:, idof] .= 0.0
        A[idof, idof] = 1.0
        M[idof, :] .= 0.0
        M[:, idof] .= 0.0
        M[idof, idof] = 1.0
        L[idof] = Ud[idof]
    end

    #vp, vecp = eigen(Array(A), Array(M))
    #display(vp)

    time = 0.0
    dt = 0.1
    totalTime = 100.0

    Miter = (M + 0.5 * dt * A)

    U0 = 260.0 * ones(Float64, (nd))
    U1 = 260.0 * ones(Float64, (nd))

    U0 = 0.0 * ones(Float64, (nd))
    U1 = 0.0 * ones(Float64, (nd))

    for idof in bnd_dofs0
        U0[idof] = Ud[idof]
        U1[idof] = Ud[idof]
    end
    for idof in bnd_dofs1
        U0[idof] = Ud[idof]
        U1[idof] = Ud[idof]
    end

    # set_values!(ϕ, x -> 260.0) Not implemented for continuous elements
    set_values!(ϕ, U1)
    # here a Dirichlet boundary condition is applied on "West". p and T are imposed to 1 (solid).
    #for idof in bnd_dofs["REAR"]
    #    Miter[idof,:] .= 0.0
    #    Miter[idof,idof]  = 1.0
    #    U0[idof] = 300.0
    #end

    #dict_vars = Dict(@sprintf("Temperature") => (get_values(ϕ), VTKPointData()))
    mkpath(outputpath)
    dict_vars = Dict("Temperature" => (var_on_centers(ϕ), VTKCellData()))
    write_vtk(outputpath * "result_unsteady_heat_equation", 0, 0.0, mesh, dict_vars)

    itime = 0
    while time <= totalTime
        time = time + dt
        itime = itime + 1
        @show time, itime
        RHS = dt * L + (M - 0.5 * dt * A) * U0
        #@show extrema(U0)
        U1 = Miter \ RHS
        #@show extrema(U1)
        U0[:] .= U1[:]

        #writedlm("Miter.txt", Miter)
        #writedlm("RHS.txt", RHS)

        #@show extrema(U1)
        set_values!(ϕ, U1)

        if itime % 100 == 0
            #dict_vars = Dict(@sprintf("Temperature") => (get_values(ϕ), VTKPointData()))
            dict_vars = Dict("Temperature" => (var_on_centers(ϕ), VTKCellData()))
            write_vtk(
                outputpath * "result_unsteady_heat_equation",
                itime,
                time,
                mesh,
                dict_vars,
            )
        end
    end

    #Usteady = A\L
    #@show extrema(Usteady)

end

run_steady()
run_unsteady()

# run_steady_twoLayer()
# run_unsteady_twoLayer()

end #hide
