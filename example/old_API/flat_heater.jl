module flat_heater #hide
println("Running flat heater API example...") #hide

const dir = string(@__DIR__, "/../") # Bcube dir
include(dir * "src/Bcube.jl")
using .Bcube
using LinearAlgebra
using WriteVTK
using Printf
using SparseArrays

function f1(u, v, params)
    λ, = v
    return params.η * ∇(λ) * transpose(∇(λ))
end

function f2(u, v, params)
    λ, = v
    return params.ρCp * λ * transpose(λ)
end

function f3(u, v, params)
    λ, = v
    return params.q * λ
end

function f4(u, v, params)
    λ, = v
    (params.htc * λ * transpose(λ),)
end

function f5(u, v, params)
    λ, = v
    (params.htc * params.Tr * λ,)
end

# convective boundary condition
const htc = 10000.0
const Tr  = 260.0
const phi = 0.0

# heat source
const l1_h = 60.0e-3
const l2_h = 300.0e-3
const e_h = 0.2e-3

const qtot = 50.0
const qheat = qtot / (l1_h * l2_h * e_h)
@show qheat

const degree = 1

function run()

    # Read mesh
    mesh, el_names, el_names_inv, el_cells, glo2loc_cell_indices =
        read_msh_with_cell_names(dir * "input/mesh/flat_heater.msh", 2)

    fs = FunctionSpace(:Lagrange, degree)
    fes = FESpace(fs, :continuous; size = 1) #  size=1 for scalar variable
    ϕ = CellVariable(:ϕ, mesh, fes)
    # Create a `TestFunction`
    λ = TestFunction(mesh, fes)

    u, v = ((ϕ,), (λ,))

    # Define measures for cell and interior face integrations
    dΩ = Measure(CellDomain(mesh), 2 * degree + 1)

    nd = ndofs(ϕ)

    #Adense = zeros(Float64, (nd,nd))
    #Mdense = zeros(Float64, (nd,nd))
    L = zeros(Float64, (nd))

    qtmp = zeros(Float64, (ncells(mesh)))
    heater = vcat(el_cells[el_names_inv["HEATER"]])
    for i in heater
        qtmp[glo2loc_cell_indices[i]] = qheat
    end
    volTag = zeros(Int64, (ncells(mesh)))
    for k in el_names
        elements = el_cells[el_names_inv[k[2]]]
        for i in elements
            volTag[glo2loc_cell_indices[i]] = k[1]
        end
    end

    mat_1 = el_cells[el_names_inv["MAT_1"]]
    mat_2 = el_cells[el_names_inv["MAT_2"]]

    rho = zeros(Float64, (ncells(mesh)))
    cp = zeros(Float64, (ncells(mesh)))
    lamda = zeros(Float64, (ncells(mesh)))
    rhoCp = zeros(Float64, (ncells(mesh)))
    for i in heater
        rho[glo2loc_cell_indices[i]] = 1500.0
        cp[glo2loc_cell_indices[i]] = 900.0
        lamda[glo2loc_cell_indices[i]] = 120.0
        rhoCp[glo2loc_cell_indices[i]] =
            rho[glo2loc_cell_indices[i]] * cp[glo2loc_cell_indices[i]]
    end
    for i in mat_1
        rho[glo2loc_cell_indices[i]] = 2000.0
        cp[glo2loc_cell_indices[i]] = 1000.0
        lamda[glo2loc_cell_indices[i]] = 120.0
        rhoCp[glo2loc_cell_indices[i]] =
            rho[glo2loc_cell_indices[i]] * cp[glo2loc_cell_indices[i]]
    end
    for i in mat_2
        rho[glo2loc_cell_indices[i]] = 2500.0
        cp[glo2loc_cell_indices[i]] = 900.0
        lamda[glo2loc_cell_indices[i]] = 10.0
        rhoCp[glo2loc_cell_indices[i]] =
            rho[glo2loc_cell_indices[i]] * cp[glo2loc_cell_indices[i]]
    end

    #set_values!(q, qtmp)
    #set_values!(ρ, rho)
    #set_values!(Cp, cp)
    #set_values!(η, lamda)

    q = CellData(qtmp)
    ρCp = CellData(rhoCp)
    η = CellData(lamda)

    params = (q = q, ρCp = ρCp, η = η, htc = htc, Tr = Tr)
    #params = (q = 0.0, ρCp= 1.0e6, η=160.0, htc=300.0, Tr=280.0)

    ndm   = max_ndofs(ϕ)
    nhint = ndm * ndm * ncells(mesh)
    Aval  = Float64[]
    rowA  = Int[]
    colA  = Int[]
    sizehint!(Aval, nhint)
    sizehint!(rowA, nhint)
    sizehint!(colA, nhint)

    dict_vars = Dict(
        "rhoCp" => (get_values(ρCp), VTKCellData()),
        "lam" => (get_values(η), VTKCellData()),
        "qheat" => (get_values(q), VTKCellData()),
    )
    write_vtk(dir * "myout/params_flat_heater", 0, 0.0, mesh, dict_vars)

    Γ_front  = BoundaryFaceDomain(mesh, ("FRONT",))
    dΓ_front = Measure(Γ_front, 2 * degree + 1)

    _AFR = ∫(f4(u, v, params))dΓ_front
    _LFR = ∫(f5(u, v, params))dΓ_front

    # compute matrices associated to bilinear and linear forms
    _A = ∫(f1(u, v, params))dΩ
    _M = ∫(f2(u, v, params))dΩ
    _L = ∫(f3(u, v, params))dΩ

    for (ic, val) in result(_L)
        idof = get_dof(ϕ, ic)
        ndof = length(idof)

        for i in 1:ndof
            L[idof[i]] += val[i]
        end
    end

    for FR_res in _AFR.result
        ic = FR_res[1][1]

        idof = get_dof(ϕ, ic)
        ndof = length(idof)

        #@show ic, idof
        for i in 1:length(idof)
            for j in 1:length(idof)
                push!(Aval, FR_res[3][1][i, j])
                push!(rowA, idof[i])
                push!(colA, idof[j])
            end
        end
    end

    for LFR_res in _LFR.result
        ic = LFR_res[1][1]

        idof = get_dof(ϕ, ic)
        ndof = length(idof)

        for i in 1:length(idof)
            L[idof[i]] += LFR_res[3][1][i]
        end
    end

    AFR = sparse(rowA, colA, Aval, nd, nd)
    #M = sparse(rowM,colM,Mval, nd, nd)
    #A = Adense
    #M = Mdense

    A = sparse(_A, ϕ) + AFR
    M = sparse(_M, ϕ)

    time = 0.0
    dt = 0.1
    totalTime = 10.0

    Miter = (M + dt * A)

    U0 = 260.0 * ones(Float64, nd)
    U1 = 260.0 * ones(Float64, nd)

    # set_values!(ϕ, x -> 260.0) Not implemented for continuous elements
    set_values!(ϕ, U1)
    # here a Dirichlet boundary condition is applied on "West". p and T are imposed to 1 (solid).
    #for idof in bnd_dofs["REAR"]
    #    Miter[idof,:] .= 0.0
    #    Miter[idof,idof]  = 1.0
    #    U0[idof] = 300.0
    #end

    #dict_vars = Dict(@sprintf("Temperature") => (get_values(ϕ), VTKPointData()))
    dict_vars = Dict("Temperature" => (var_on_centers(ϕ), VTKCellData()))
    write_vtk(dir * "myout/result_flat_heater", 0, 0.0, mesh, dict_vars)

    itime = 0
    while time <= totalTime
        time = time + dt
        itime = itime + 1
        @show time, itime
        RHS = dt * L + M * U0
        #for idof in bnd_dofs["REAR"]
        #    RHS[idof] = 300.0
        #end
        U1 = Miter \ RHS
        U0[:] .= U1[:]

        @show extrema(U1)

        set_values!(ϕ, U1)

        if itime % 10 == 0
            #dict_vars = Dict(@sprintf("Temperature") => (get_values(ϕ), VTKPointData()))
            dict_vars = Dict("Temperature" => (var_on_centers(ϕ), VTKCellData()))
            write_vtk(dir * "myout/result_flat_heater", itime, time, mesh, dict_vars)
        end
    end

    Usteady = A \ L
    set_values!(ϕ, Usteady)
    dict_vars = Dict("Temperature" => (var_on_centers(ϕ), VTKCellData()))
    write_vtk(dir * "myout/result_flat_heater", itime + 1, time + 10.0, mesh, dict_vars)

    @show extrema(Usteady)
end

run()

end #hide
