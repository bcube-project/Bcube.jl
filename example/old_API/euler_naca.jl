module EulerNaca #hide
println("Running euler_naca example...") #hide
# # Solve Euler equation around a NACA0012 airfoil

include(string(@__DIR__, "/../src/Bcube.jl"))
using .Bcube
using LinearAlgebra
using WriteVTK
using StaticArrays
using BenchmarkTools
using Roots
using SparseArrays
using SparseDiffTools
using Profile
#using Symbolics
using InteractiveUtils
using WriteVTK

const dir = string(@__DIR__, "/")

function compute_residual(w, params, cache)
    # destructuring
    u, v = w

    limiter_projection && apply_limitation!(w, params, cache)

    # alias on measures
    dΓ = params.dΓ
    dΩ = params.dΩ
    dΓ_wall = params.dΓ_wall
    dΓ_farfield = params.dΓ_farfield

    # init a tuple of new CellVariables from `u` to store all residual contributions
    du = zeros.(u)

    # compute volume residuals
    du_Ω = ∫(flux_Ω(u, v))dΩ

    # Store volume residuals in `du`
    du += du_Ω

    # face normals for each face domain (lazy, no computation at this step)
    n_Γ = FaceNormals(dΓ)
    n_Γ_wall = FaceNormals(dΓ_wall)
    n_Γ_farfield = FaceNormals(dΓ_farfield)

    # flux residuals from interior faces for all variables
    du_Γ = ∫(flux_Γ(u, v, n_Γ))dΓ

    # flux residuals from bc faces for all variables
    du_Γ_wall = ∫(flux_Γ_wall(u, v, n_Γ_wall))dΓ_wall
    du_Γ_farfield = ∫(flux_Γ_farfield(u, v, n_Γ_farfield))dΓ_farfield

    # accumulate face-flux residuals to cell residuals for all variables
    # (this step will be improved when a better API will be available)
    du = ((du - du_Γ) - du_Γ_wall) - du_Γ_farfield

    return du
end

"""
    flux_Ω(u,v)

Compute volume residual using the lazy-operators approach
"""
flux_Ω(u, v) = _flux_Ω ∘ cellvar(u, v)
cellvar(u, v) = (u, ∇.(v))
function _flux_Ω(args)
    u, ∇v = args
    ρ, ρu, ρE = u
    ∇λ_ρ, ∇λ_ρu, ∇λ_ρE = ∇v
    γ = stateInit.γ

    ncomp = length(ρu)
    Id = SMatrix{ncomp, ncomp}(1.0I)

    flux_ρ = ∇λ_ρ * ρu

    vel = ρu ./ ρ
    ρuu = ρu * transpose(vel)
    p = (γ - 1) * (ρE - tr(ρuu) / 2)
    flux_ρu = ∇λ_ρu ⊡ (ρuu .+ p .* Id)

    flux_ρE = ∇λ_ρE * ((ρE + p) .* vel)

    return (flux_ρ, flux_ρu, flux_ρE)
end

"""
    flux_Γ(u,v,n,params)

Flux at the interface is defined by a composition of two functions:
* facevar(u,v,n) defines the input states which are needed for
  the riemann flux using operator notations
* flux_roe(w) defines the Riemann flux (as usual)
"""
flux_Γ(u, v, n) = flux_roe ∘ facevar(u, v, n)
facevar(u, v, n) = (side⁻(u), side⁺(u), side⁻(v), n)

"""
    flux_roe((w1, w2, λ1, n12))
"""
function flux_roe(args)
    w1, w2, λ1, n12 = args
    γ = stateInit.γ
    nx = n12[1]
    ny = n12[2]
    ρ1, (ρu1, ρv1), ρE1 = w1
    ρ2, (ρu2, ρv2), ρE2 = w2
    λ_ρ1, λ_ρu1, λ_ρE1 = λ1

    # Closure
    u1 = ρu1 / ρ1
    v1 = ρv1 / ρ1
    u2 = ρu2 / ρ2
    v2 = ρv2 / ρ2
    p1 = pressure(ρ1, SA[ρu1, ρv1], ρE1, γ)
    p2 = pressure(ρ2, SA[ρu2, ρv2], ρE2, γ)

    H2 = (γ / (γ - 1)) * p2 / ρ2 + (u2 * u2 + v2 * v2) / 2.0
    H1 = (γ / (γ - 1)) * p1 / ρ1 + (u1 * u1 + v1 * v1) / 2.0

    R = √(ρ1 / ρ2)
    invR1 = 1.0 / (R + 1)
    uAv = (R * u1 + u2) * invR1
    vAv = (R * v1 + v2) * invR1
    Hav = (R * H1 + H2) * invR1
    cAv = √(abs((γ - 1) * (Hav - (uAv * uAv + vAv * vAv) / 2.0)))
    ecAv = (uAv * uAv + vAv * vAv) / 2.0

    λ1 = nx * uAv + ny * vAv
    λ3 = λ1 + cAv
    λ4 = λ1 - cAv

    d1 = ρ1 - ρ2
    d2 = ρ1 * u1 - ρ2 * u2
    d3 = ρ1 * v1 - ρ2 * v2
    d4 = ρE1 - ρE2

    # computation of the centered part of the flux
    flu11 = nx * ρ2 * u2 + ny * ρ2 * v2
    flu21 = nx * p2 + flu11 * u2
    flu31 = ny * p2 + flu11 * v2
    flu41 = H2 * flu11

    # Temp variables
    rc1 = (γ - 1) / cAv
    rc2 = (γ - 1) / cAv / cAv
    uq41 = ecAv / cAv + cAv / (γ - 1)
    uq42 = nx * uAv + ny * vAv

    fdc1 = max(λ1, 0.0) * (d1 + rc2 * (-ecAv * d1 + uAv * d2 + vAv * d3 - d4))
    fdc2 = max(λ1, 0.0) * ((nx * vAv - ny * uAv) * d1 + ny * d2 - nx * d3)
    fdc3 =
        max(λ3, 0.0) * (
            (-uq42 * d1 + nx * d2 + ny * d3) / 2.0 +
            rc1 * (ecAv * d1 - uAv * d2 - vAv * d3 + d4) / 2.0
        )
    fdc4 =
        max(λ4, 0.0) * (
            (uq42 * d1 - nx * d2 - ny * d3) / 2.0 +
            rc1 * (ecAv * d1 - uAv * d2 - vAv * d3 + d4) / 2.0
        )

    duv1 = fdc1 + (fdc3 + fdc4) / cAv
    duv2 = uAv * fdc1 + ny * fdc2 + (uAv / cAv + nx) * fdc3 + (uAv / cAv - nx) * fdc4
    duv3 = vAv * fdc1 - nx * fdc2 + (vAv / cAv + ny) * fdc3 + (vAv / cAv - ny) * fdc4
    duv4 =
        ecAv * fdc1 +
        (ny * uAv - nx * vAv) * fdc2 +
        (uq41 + uq42) * fdc3 +
        (uq41 - uq42) * fdc4

    return (
        λ_ρ1 * (flu11 + duv1),
        λ_ρu1 * (SA[flu21 + duv2, flu31 + duv3]),
        λ_ρE1 * (flu41 + duv4),
    )
end

"""
    flux_Γ_farfield(u, v, n)
"""
facevar_farfield(u, v, n) = (side⁻(u), side⁻(v), n)
# Note:
# Currently operator composition works on tuples which contain 'CellVariable's only.
# Then, the farfield state 'u_inf', which is used as the rigth state of the Roe flux at the farfield boundary condition,
# is directly given the function 'flux_roe'. Ultimately, we would like to be able to write something like:
# flux_Γ_farfield(u,v,n) = flux_roe ∘ facevar_farfield(u,stateBcFarfield.u_inf, v,n)
function flux_Γ_farfield(u, v, n)
    (w -> flux_roe((w[1], stateBcFarfield.u_inf, w[2], w[3]))) ∘ facevar_farfield(u, v, n)
end

"""
    flux_Γ_wall(u, v, n)
"""
facevar_wall(u, v, n) = (side⁻(u), side⁻(v), n)
flux_Γ_wall(u, v, n) = _flux_Γ_wall ∘ facevar_wall(u, v, n)

function _flux_Γ_wall((w1, λ1, n12))
    γ = stateInit.γ
    ρ1, (ρu1, ρv1), ρE1 = w1
    λ_ρ1, λ_ρu1, λ_ρE1 = λ1

    p1 = pressure(ρ1, SA[ρu1, ρv1], ρE1, γ)

    flu11 = 0.0
    flu21 = p1 * n12
    flu41 = 0.0

    return (λ_ρ1 * (flu11), λ_ρu1 * (flu21), λ_ρE1 * (flu41))
end

# Now let's write the assembling method which computes the volumic terms and assembles them with the surfacic flux terms.
# We call this method `explicit_step!` because it returns the ``\Delta \phi`` associated with the selected explicit time
# scheme.
function explicit_step(w, params, cache, Δt, t)
    (ρ, ρu, ρE), λ = w

    _dρ, _dρu, _dρE, = compute_residual(w, params, cache)

    for x in (_dρ, _dρu, _dρE)
        lmul!(Δt, x)
    end

    ρ_new  = get_values(ρ) .+ cache.invMass_ρ * get_values(_dρ)
    ρu_new = get_values(ρu) .+ cache.invMass_ρu * get_values(_dρu)
    ρE_new = get_values(ρE) .+ cache.invMass_ρE * get_values(_dρE)

    return ρ_new, ρu_new, ρE_new
end

function mass_matrix(u, v)
    u_ρ, u_ρu, u_ρE = get_trial_function.(u)
    v_ρ, v_ρu, v_ρE = v
    (u_ρ * transpose(v_ρ), u_ρu * transpose(v_ρu), u_ρE * transpose(v_ρE))
end

function compute_mass_matrix(w, params)
    u, v = w
    dΩ = params.dΩ
    ∫(mass_matrix(u, v))dΩ
end

function implicit_step(w, params, cache, Δt, t, jac_cache)
    u, v = w
    dofs = cache.dofs
    system = cache.system
    dofs2 = similar(dofs)
    pack!(dofs, u, system)

    f! = (out, in) -> rhs!(out, in, w, params, cache)
    if isa(jac_cache, Nothing)
        println("Computing jac_cache...")
        jac_cache = ForwardColorJacCache(
            f!,
            dofs,
            nothing;
            dx = similar(dofs),
            colorvec = get_jac_colors(system),
            sparsity = get_jac_sparsity(system),
        )
        @show maximum(get_jac_colors(system))
        debug_implicit && write_sparsity(system, "sparsity")
    end

    J = get_jac_sparsity(system)
    forwarddiff_color_jacobian!(J, f!, dofs, jac_cache)

    f!(dofs2, dofs)
    _M = compute_mass_matrix(w, params)
    M = sparse(_M, u, get_mapping(system))

    Δtmin = minimum(get_values(Δt))
    A = M ./ Δtmin - J
    dofs .= A \ dofs2
    Δu = zeros.(u)
    unpack!(Δu, dofs, system)

    return coef_relax_implicit .* get_values.(Δu) .+ get_values.(u), jac_cache
end

function rhs!(out, in, w, params, cache)
    u, v = w
    u2 = zeros.(u, eltype(in))
    unpack!(u2, in, cache.system)
    u3 = compute_residual((u2, v), params, cache)
    pack!(out, u3, cache.system)
    return nothing
end

function sparse2vtk(
    a::AbstractSparseMatrix,
    name::String = string(@__DIR__, "/../myout/sparse"),
)
    vtk_write_array(name, Array(a), "my_property_name")
end

function RK3_SSP(f::Function, w)
    u, v = w

    _u₁ = f((u, v))
    u₁ = zeros.(u)
    map(set_values!, u₁, _u₁)

    _u₂ = f((u₁, v))
    u₂ = zeros.(u)
    map(
        (a, b, c) -> set_values!(a, (3.0 / 4) .* get_values(b) .+ (1.0 / 4) .* (c)),
        u₂,
        u,
        _u₂,
    )

    _u₃ = f((u₂, v))
    u₃ = map((a, b) -> (1.0 / 3) .* get_values(a) .+ (2.0 / 3) .* (b), u, _u₃)

    return u₃
end

mutable struct VtkHandler
    basename::String
    basename_residual::String
    ite::Int
    VtkHandler(basename) = new(basename, basename * "_residual", 0)
end

"""
    Write solution (at cell centers) to vtk
    Wrapper for `write_vtk`
"""
function append_vtk(vtk, mesh, vars, t, params; res = nothing)
    # Values on center
    # Mean cell values
    name2val_mean = (; zip(get_name.(vars), mean_values.(vars, degquad))...)
    p_mean =
        pressure.(
            name2val_mean[:ρ],
            name2val_mean[:ρu],
            name2val_mean[:ρE],
            params.stateInit.γ,
        )

    vtk_degree = maximum(x -> get_order(function_space(x)), vars)
    vtk_degree = max(1, mesh_degree, vtk_degree)
    name2val_dg = (; zip(get_name.(vars), var_on_nodes_discontinuous.(vars, vtk_degree))...)

    dict_vars_dg = Dict(
        "rho" => (name2val_dg[:ρ], VTKPointData()),
        "rhou" => (name2val_dg[:ρu], VTKPointData()),
        "rhoE" => (name2val_dg[:ρE], VTKPointData()),
        "rho_mean" => (name2val_mean[:ρ], VTKCellData()),
        "rhou_mean" => (name2val_mean[:ρu], VTKCellData()),
        "rhoE_mean" => (name2val_mean[:ρE], VTKCellData()),
    )
    Bcube.write_vtk_discontinuous(
        vtk.basename * "_DG",
        vtk.ite,
        t,
        mesh,
        dict_vars_dg,
        vtk_degree;
        append = vtk.ite > 0,
    )

    #residual:
    if !isa(res, Nothing)
        vtkfile = vtk_grid(vtk.basename_residual, Float64.(res.iter), [0.0, 1.0])
        for (k, valₖ) in enumerate(res.val)
            vtkfile["res_" * string(k), VTKPointData()] = [valₖ valₖ]
        end
        vtk_save(vtkfile)
    end

    # Update counter
    vtk.ite += 1

    return nothing
end

function init!(u, initstate)
    AoA  = initstate.AoA
    Minf = initstate.M_inf
    Pinf = initstate.P_inf
    Tinf = initstate.T_inf
    r    = initstate.r_gas
    γ    = initstate.γ

    ρinf = Pinf / r / Tinf
    ainf = √(γ * r * Tinf)
    Vinf = Minf * ainf
    ρVxinf = ρinf * Vinf * cos(AoA)
    ρVyinf = ρinf * Vinf * sin(AoA)
    ρEinf = Pinf / (γ - 1) + 0.5 * ρinf * Vinf^2

    ρ, ρu, ρE = u
    set_values!(ρ, x -> ρinf; degquad = degquad)
    set_values!(ρu, (x -> ρVxinf, x -> ρVyinf); degquad = degquad)
    set_values!(ρE, x -> ρEinf; degquad = degquad)
end

function run(stateInit, stateBcFarfield, degree)
    @show degree, degquad

    if debug_implicit
        tmp_path = "tmp.msh"
        gen_rectangle_mesh(
            tmp_path,
            :quad;
            nx = 5,
            ny = 5,
            lx = 1.0,
            ly = 1.0,
            xc = 0.0,
            yc = 0.0,
        )
        mesh = read_msh(tmp_path)
    else
        mesh = read_msh(dir * "../input/mesh/naca0012_o" * string(mesh_degree) * ".msh")
    end

    dimcar = compute_dimcar(mesh)

    # Create a `CellVariable`
    fs = FunctionSpace(fspace, degree)
    fes1 = FESpace(fs, :discontinuous; size = 1) #  size=1 for scalar variable
    fes2 = FESpace(fs, :discontinuous; size = 2) # DG, vectoriel
    ρ = CellVariable(:ρ, mesh, fes1)
    ρu = CellVariable(:ρu, mesh, fes2)
    ρE = CellVariable(:ρE, mesh, fes1)
    λ₁ = TestFunction(mesh, fes1)
    λ₂ = TestFunction(mesh, fes2)
    u, v = ((ρ, ρu, ρE), (λ₁, λ₂, λ₁))

    # select an initial configurations:
    init!(u, stateInit)

    DMPrelax = DMPcurv₀ .* dimcar .^ 2

    # Then we create a `NamedTuple` to hold the simulation parameters.
    params = (
        degree = degree,
        degquad = degquad,
        stateInit = stateInit,
        stateBcFarfield = stateBcFarfield,
        DMPrelax = DMPrelax,
    )

    # Define measures for cell and interior face integrations
    dΩ = Measure(CellDomain(mesh), degquad)
    dΓ = Measure(InteriorFaceDomain(mesh), degquad)

    # Declare periodic boundary conditions and
    # create associated domains and measures
    if debug_implicit
        Γ_wall = BoundaryFaceDomain(mesh, ("South", "North"))
        Γ_farfield = BoundaryFaceDomain(mesh, ("East", "West"))
    else
        Γ_wall = BoundaryFaceDomain(mesh, ("NACA",))
        Γ_farfield = BoundaryFaceDomain(mesh, ("FARFIELD",))
    end
    dΓ_wall = Measure(Γ_wall, degquad)
    dΓ_farfield = Measure(Γ_farfield, degquad)

    params = (params..., dΓ = dΓ, dΩ = dΩ, dΓ_wall = dΓ_wall, dΓ_farfield = dΓ_farfield)

    # create CellData to store limiter values
    limρ = CellData(ones(ncells(mesh)))
    limAll = CellData(ones(ncells(mesh)))
    params = (params..., limρ = limρ, limAll = limAll)

    # Init vtk handler
    mkpath(outputpath)
    vtk = VtkHandler(
        outputpath * "euler_naca_mdeg" * string(mesh_degree) * "_deg" * string(degree),
    )

    # Init time
    time = 0.0

    # Save initial solution
    append_vtk(vtk, mesh, u, time, params)

    # Build the cache and store everything you want to compute only once (such as the mass matrice inverse...)
    invMass_ρ  = InvMassMatrix(ρ, Val(degquad))
    invMass_ρu = InvMassMatrix(ρu, Val(degquad))
    invMass_ρE = InvMassMatrix(ρE, Val(degquad))
    cache      = (invMass_ρ = invMass_ρ, invMass_ρu = invMass_ρu, invMass_ρE = invMass_ρE)

    system = System(u; cacheJacobian = true)
    dofs = zeros(eltype(get_values(ρ)), get_ndofs(system))
    cache = (cache..., dofs = dofs, system = system)
    jac_cache = nothing

    Δt = CellData(Δt₀ .* ones(typeof(Δt₀), ncells(mesh)))

    res = (iter = Int[], val = map(x -> typeof(norm(get_values(x)))[], u))

    # Let's loop to solve the equation.
    for i in 1:nite_max
        _CFL = fun_CFL(i)

        compute_timestep!(Δt, u, mesh, dimcar, degquad, _CFL, params)
        !(localTimeStep) && set_values!(Δt, minimum(Δt))

        ## Infos
        if (i % Int(max(floor(nite_max / (nout * 10)), 1)) == 0)
            println("---")
            println("Iteration ", i)
            @show minimum(Δt), _CFL
        end

        ## Step forward in time
        if timeScheme == :ForwardEuler
            u_new = explicit_step((u, v), params, cache, Δt, time)
        elseif timeScheme == :BackwardEuler
            u_new, jac_cache = implicit_step((u, v), params, cache, Δt, time, jac_cache)
        elseif timeScheme == :RK3
            stepper = w -> explicit_step(w, params, cache, Δt, time)
            u_new = RK3_SSP(stepper, (u, v))
        else
            println("Unknown time scheme: ", timeScheme)
        end

        res_ = norm.(get_values.(u) .- u_new)
        push!(res.iter, i)
        for (k, valₖ) in enumerate(res.val)
            length(valₖ) > 0 ? valₖ₀ = valₖ[1] : valₖ₀ = 1.0
            push!(valₖ, res_[k] / valₖ₀)
        end
        println("Residual = ", [res.val[k][end] for k in 1:length(res.val)])
        map(set_values!, u, u_new)

        time += minimum(get_values(Δt))

        ## Write solution to file (respecting max. number of output)
        if (i % Int(max(floor(nite_max / nout), 1)) == 0)
            println("writing vtk...")
            append_vtk(vtk, mesh, u, time, params; res = res)
        end
    end

    append_vtk(vtk, mesh, u, time, params; res = res)

    @show degmass, degquad
    println("Benchmarking 'explicit_step':")
    @btime explicit_step(($u, $v), $params, $cache, $Δt, $time)
    if timeScheme == :BackwardEuler
        println("Benchmarking 'implicit_step':")
        @btime implicit_step(($u, $v), $params, $cache, $Δt, $time, $jac_cache)
    end
    println("ndofs total = ", sum(length.(get_values.(u))))

    if timeScheme == :BackwardEuler
        Profile.clear()
        Profile.clear_malloc_data()
        @profile begin
            for i in 1:5
                implicit_step((u, v), params, cache, Δt, time, jac_cache)
            end
        end
    end
    # And here is an animation of the result:
    # ```@raw html
    # <img src="../assets/linear_transport_heavidise.gif" alt="drawing" width="700"/>
    # ```
end

function compute_timestep!(Δt, u, mesh, dimcar, degquad, CFL, params)
    λ = Bcube._minmax_cells(euler_maxeigval(u, params.stateInit.γ), mesh, Val(degquad))
    λmax = getindex.(λ, 2)
    set_values!(Δt, CFL .* dimcar ./ λmax)
end

euler_maxeigval(u, γ) = (x -> _euler_maxeigval(x, γ)) ∘ (u,)
function _euler_maxeigval(args, γ)
    u, = args
    ρ, ρu, ρE = u
    vel = ρu ./ ρ
    ρuu = ρu * transpose(vel)
    p = (γ - 1) * (ρE - tr(ρuu) / 2)
    sqrt(tr(ρuu) / ρ) + sqrt(γ * p / ρ)
end

compute_Pᵢ(P, γ, M) = P * (1 + 0.5 * (γ - 1) * M^2)^(γ / (γ - 1))
compute_Tᵢ(T, γ, M) = T * (1 + 0.5 * (γ - 1) * M^2)

function compute_dimcar(mesh)
    fs = FunctionSpace(:Taylor, 0)
    fes = FESpace(fs, :discontinuous; size = 1)
    a = CellVariable(:a, mesh, fes)
    set_values!(a, x -> 1.0)

    # Define measures for cell and interior face integrations
    dΩ = Measure(CellDomain(mesh), degquad)
    dΓ = Measure(InteriorFaceDomain(mesh), degquad)
    dΓ_bc = Measure(BoundaryFaceDomain(mesh, Tuple(values(boundary_names(mesh)))), degquad)

    int_Ω    = ∫(a * SA[1.0])dΩ
    int_Γ    = ∫(a * SA[1.0])dΓ
    int_Γ_bc = ∫(a * SA[1.0])dΓ_bc
    vol      = zeros(a) + int_Ω
    surf     = (zeros(a) + int_Γ) + int_Γ_bc
    dimcar1  = get_values(vol) ./ get_values(surf)
    return dimcar1
end

function apply_limitation!(w, params, cache)
    u, v = w
    ρ, ρu, ρE = u
    mesh = ρ.mesh

    ρ_mean, ρu_mean, ρE_mean = CellData.(mean_values.((ρ, ρu, ρE), Val(params.degquad)))

    _limρ, _ρ_proj = linear_scaling_limiter(
        ρ,
        params.degquad;
        bounds = (ρmin₀, ρmax₀),
        DMPrelax = params.DMPrelax,
        invMass = cache.invMass_ρ,
    )
    ρ_proj = zeros(ρ)
    set_values!(ρ_proj, _ρ_proj)
    op_t = limiter_param_p ∘ (ρ_proj, ρu, ρE, ρ_mean, ρu_mean, ρE_mean)
    t = Bcube._minmax_cells(op_t, mesh, Val(params.degquad))
    tmin = CellData(getindex.(t, 1))

    #@show extrema(get_values(_limρ)), extrema(get_values(tmin))
    if eltype(_limρ) == eltype(params.limρ)
        set_values!(params.limρ, get_values(_limρ))
        set_values!(params.limAll, get_values(tmin))
    end

    set_values!(ρ, Bcube._projection(ρ_proj, tmin, ρ_mean, cache.invMass_ρ, degquad))
    set_values!(ρu, Bcube._projection(ρu, tmin, ρu_mean, cache.invMass_ρu, degquad))
    set_values!(ρE, Bcube._projection(ρE, tmin, ρE_mean, cache.invMass_ρE, degquad))
    nothing
end

function pressure(ρ, ρu, ρE, γ)
    vel = ρu ./ ρ
    ρuu = ρu * transpose(vel)
    p = (γ - 1) * (ρE - tr(ρuu) / 2)
    return p
end

function limiter_param_p((ρ̂, ρu, ρE, ρ_mean, ρu_mean, ρE_mean))
    γ = stateInit.γ

    p = pressure(ρ̂, ρu, ρE, γ)
    if p ≥ pmin₀
        t = 1.0
    else
        @show p, ρ̂, ρu, ρE
        @show ρ_mean, ρu_mean, ρE_mean
        @show pressure(ρ_mean, ρu_mean, ρE_mean, γ)
        #@show p, ρ̂, ρu, ρE, ρ_mean, ρu_mean, ρE_mean
        fₜ =
            t ->
                pressure(
                    t * ρ̂ + (1 - t) * ρ_mean,
                    t * ρu + (1 - t) * ρu_mean,
                    t * ρE + (1 - t) * ρE_mean,
                    γ,
                ) - pmin₀
        bounds = (0.0, 1.0)
        t = find_zero(fₜ, bounds, Bisection())
    end
    return t
end

function bc_state_farfield(AoA, M, P, T, r, γ)
    a = √(γ * r * T)
    vn = M * a
    ρ = P / r / T
    ρu = SA[ρ * vn * cos(AoA), ρ * vn * sin(AoA)]
    ρE = P / (γ - 1) + 0.5 * ρ * vn^2
    return (ρ, ρu, ρE)
end

const debug_implicit = false
const degreemax = 2# Function-space order (Taylor(0) = first order Finite Volume)
const mesh_degree = 2
const fspace = :Lagrange
const limiter_projection = false
const ρmin₀ = 1.0e-10
const ρmax₀ = 1.0e+10
const pmin₀ = 1.0e-10
const pmax₀ = 1.0e+10
const DMPcurv₀ = 10.0
const localTimeStep = true

const stateInit = (
    AoA = deg2rad(4.0),
    M_inf = 0.3,
    P_inf = 88888.0,
    T_inf = 263.0,
    r_gas = 287.0,
    γ = 1.4,
)
const nite_max = 300 #300000 # Number of time iteration(s)
const CFL = 100 * 0.1 * 1.0 / (2 * degreemax + 1)
const timeScheme = :BackwardEuler # :BackwardEuler, :ForwardEuler, :RK3
const CFLratio = 1.5
const CFLmin = 1
const CFLmax = 3000
const iter_ramp_min_CFL = 1
const iter_ramp_max_CFL = 200
const coef_relax_implicit = 0.5
const nout = 100 # Number of time steps to save
const mass_matrix_in_solve = false
const Δt₀ = 1.e-7
const outputpath = string(@__DIR__, "/../myout/euler_naca/")

function fun_CFL(i)
    k = (i - iter_ramp_min_CFL) / (iter_ramp_max_CFL - iter_ramp_min_CFL)
    k = max(0.0, min(1.0, k))
    CFL = (1 - k) * CFLmin + k * CFLmax
    return CFL
end

const degquad = 4 # Int(ceil((degmass+3)/2))

const stateBcFarfield = (
    AoA = stateInit.AoA,
    M_inf = stateInit.M_inf,
    Pᵢ_inf = compute_Pᵢ(stateInit.P_inf, stateInit.γ, stateInit.M_inf),
    Tᵢ_inf = compute_Tᵢ(stateInit.T_inf, stateInit.γ, stateInit.M_inf),
    u_inf = bc_state_farfield(
        stateInit.AoA,
        stateInit.M_inf,
        stateInit.P_inf,
        stateInit.T_inf,
        stateInit.r_gas,
        stateInit.γ,
    ),
    r_gas = stateInit.r_gas,
    γ = stateInit.γ,
)

run(stateInit, stateBcFarfield, degreemax)

end #hide
