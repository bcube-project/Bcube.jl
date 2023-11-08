module EulerNacaSteady #hide
println("Running euler_naca_steady example...") #hide
# # Solve Euler equation around a NACA0012 airfoil

using Bcube
using LinearAlgebra
using WriteVTK
using StaticArrays
using BenchmarkTools
using Roots
using SparseArrays
using Profile
using InteractiveUtils
using WriteVTK
using DifferentialEquations
using Symbolics
using SparseDiffTools

const dir = string(@__DIR__, "/")

function compute_residual(qdof, Q, V, params)
    q = (FEFunction(Q, qdof)...,)

    # alias on measures
    dΓ = params.dΓ
    dΩ = params.dΩ
    dΓ_wall = params.dΓ_wall
    dΓ_farfield = params.dΓ_farfield

    # Allocate rhs vectors
    b_vol = zero(qdof)
    b_fac = zero(qdof)

    # compute volume residuals
    l_vol(v) = ∫(flux_Ω(q, v))dΩ
    assemble_linear!(b_vol, l_vol, V)

    # face normals for each face domain (lazy, no computation at this step)
    nΓ = get_face_normals(dΓ)
    nΓ_wall = get_face_normals(dΓ_wall)
    nΓ_farfield = get_face_normals(dΓ_farfield)

    # flux residuals from interior faces for all variables
    l_Γ(v) = ∫(flux_Γ(q, v, nΓ))dΓ
    assemble_linear!(b_fac, l_Γ, V)

    # flux residuals from bc faces for all variables
    l_Γ_wall(v) = ∫(flux_Γ_wall(q, v, nΓ_wall))dΓ_wall
    l_Γ_farfield(v) = ∫(flux_Γ_farfield(q, v, nΓ_farfield))dΓ_farfield
    assemble_linear!(b_fac, l_Γ_wall, V)
    assemble_linear!(b_fac, l_Γ_farfield, V)
    dQ = b_vol .- b_fac

    return dQ
end

"""
    flux_Ω(q, v)

Compute volume residual using the lazy-operators approach
"""
flux_Ω(q, v) = _flux_Ω ∘ (q, map(∇, v))

function _flux_Ω(q, ∇v)
    ρ, ρu, ρE = q
    ∇λ_ρ, ∇λ_ρu, ∇λ_ρE = ∇v
    γ = stateInit.γ

    vel = ρu ./ ρ
    ρuu = ρu * transpose(vel)
    p = pressure(ρ, ρu, ρE, γ)

    flux_ρ  = ρu
    flux_ρu = ρuu + p * I
    flux_ρE = (ρE + p) .* vel

    return return ∇λ_ρ ⋅ flux_ρ + ∇λ_ρu ⊡ flux_ρu + ∇λ_ρE ⋅ flux_ρE
end

"""
    flux_Γ(q, v, n)

Flux at the interface is defined by a composition of two functions:
* the input states at face sides which are needed for the riemann flux
* `flux_roe` defines the Riemann flux (as usual)
"""
flux_Γ(q, v, n) = flux_roe ∘ (side⁻(q), side⁺(q), jump(v), side⁻(n))

"""
    flux_roe(q⁻, q⁺, δv, n)
"""
function flux_roe(q⁻, q⁺, δv, n)
    γ = stateInit.γ
    nx, ny = n
    ρ1, (ρu1, ρv1), ρE1 = q⁻
    ρ2, (ρu2, ρv2), ρE2 = q⁺
    δλ_ρ1, δλ_ρu1, δλ_ρE1 = δv

    ρ1 = max(eps(ρ1), ρ1)
    ρ2 = max(eps(ρ2), ρ2)

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
    flux_ρ  = nx * ρ2 * u2 + ny * ρ2 * v2
    flux_ρu = nx * p2 + flux_ρ * u2
    flux_ρv = ny * p2 + flux_ρ * v2
    flux_ρE = H2 * flux_ρ

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

    flux_ρ  += duv1
    flux_ρu += duv2
    flux_ρv += duv3
    flux_ρE += duv4

    return (δλ_ρ1 ⋅ flux_ρ + δλ_ρu1 ⋅ SA[flux_ρu, flux_ρv] + δλ_ρE1 ⋅ flux_ρE)
end

"""
    flux_Γ_farfield(q, v, n)

Compute `Roe` flux on boundary face by imposing
`stateBcFarfield.u_in` on `side_p`
"""
flux_Γ_farfield(q, v, n) = flux_roe ∘ (side⁻(q), stateBcFarfield.u_inf, side⁻(v), side⁻(n))

"""
    flux_Γ_wall(q, v, n)
"""
flux_Γ_wall(q, v, n) = _flux_Γ_wall ∘ (side⁻(q), side⁻(v), side⁻(n))

function _flux_Γ_wall(q⁻, v⁻, n)
    γ = stateInit.γ
    ρ1, ρu1, ρE1 = q⁻
    λ_ρ1, λ_ρu1, λ_ρE1 = v⁻

    p1 = pressure(ρ1, ρu1, ρE1, γ)

    flux_ρ  = zero(ρ1)
    flux_ρu = p1 * n
    flux_ρE = zero(ρE1)

    return (λ_ρ1 ⋅ flux_ρ + λ_ρu1 ⋅ flux_ρu + λ_ρE1 ⋅ flux_ρE)
end

function sparse2vtk(
    a::AbstractSparseMatrix,
    name::String = string(@__DIR__, "/../myout/sparse"),
)
    vtk_write_array(name, Array(a), "my_property_name")
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
    ρ, ρu, ρE = vars

    # Mean cell values
    # name2val_mean = (;zip(get_name.(vars), mean_values.(vars, degquad))...)
    # p_mean = pressure.(name2val_mean[:ρ], name2val_mean[:ρu], name2val_mean[:ρE], params.stateInit.γ)

    vtk_degree = maximum(x -> get_degree(Bcube.get_function_space(get_fespace(x))), vars)
    vtk_degree = max(1, mesh_degree, vtk_degree)
    _ρ = var_on_nodes_discontinuous(ρ, mesh, vtk_degree)
    _ρu = var_on_nodes_discontinuous(ρu, mesh, vtk_degree)
    _ρE = var_on_nodes_discontinuous(ρE, mesh, vtk_degree)

    Cp = pressure_coefficient.(_ρ, _ρu, _ρE)
    Ma = mach.(_ρ, _ρu, _ρE)
    dict_vars_dg = Dict(
        "rho" => (_ρ, VTKPointData()),
        "rhou" => (_ρu, VTKPointData()),
        "rhoE" => (_ρE, VTKPointData()),
        "Cp" => (Cp, VTKPointData()),
        "Mach" => (Ma, VTKPointData()),
        "rho_mean" => (get_values(Bcube.cell_mean(ρ, params.dΩ)), VTKCellData()),
        "rhou_mean" => (get_values(Bcube.cell_mean(ρu, params.dΩ)), VTKCellData()),
        "rhoE_mean" => (get_values(Bcube.cell_mean(ρE, params.dΩ)), VTKCellData()),
        "lim_rho" => (get_values(params.limρ), VTKCellData()),
        "lim_all" => (get_values(params.limAll), VTKCellData()),
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

    _ρ_wall = var_on_bnd_nodes_discontinuous(ρ, params.Γ_wall, vtk_degree)
    _ρu_wall = var_on_bnd_nodes_discontinuous(ρu, params.Γ_wall, vtk_degree)
    _ρE_wall = var_on_bnd_nodes_discontinuous(ρE, params.Γ_wall, vtk_degree)

    Cp_wall = pressure_coefficient.(_ρ_wall, _ρu_wall, _ρE_wall)
    Ma_wall = pressure_coefficient.(_ρ_wall, _ρu_wall, _ρE_wall)

    dict_vars_wall = Dict(
        "rho" => (_ρ_wall, VTKPointData()),
        "rhou" => (_ρu_wall, VTKPointData()),
        "rhoE" => (_ρE_wall, VTKPointData()),
        "Cp" => (Cp_wall, VTKPointData()),
        "Mach" => (Ma_wall, VTKPointData()),
    )
    Bcube.write_vtk_bnd_discontinuous(
        vtk.basename * "_bnd_DG",
        1,
        0.0,
        params.Γ_wall,
        dict_vars_wall,
        vtk_degree;
        append = false,
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

function init!(q, dΩ, initstate)
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

    ρ0  = PhysicalFunction(x -> ρinf)
    ρu0 = PhysicalFunction(x -> SA[ρVxinf, ρVyinf])
    ρE0 = PhysicalFunction(x -> ρEinf)
    projection_l2!(q, (ρ0, ρu0, ρE0), dΩ)
    return nothing
end

function main(stateInit, stateBcFarfield, degree)
    @show degree, degquad

    mesh = read_msh(dir * "../input/mesh/naca0012_o" * string(mesh_degree) * ".msh", 2)
    scale!(mesh, 1.0 / 0.5334)

    dimcar = compute_dimcar(mesh)

    DMPrelax = DMPcurv₀ .* dimcar .^ 2

    # Then we create a `NamedTuple` to hold the simulation parameters.
    params = (
        degquad = degquad,
        stateInit = stateInit,
        stateBcFarfield = stateBcFarfield,
        DMPrelax = DMPrelax,
    )

    # Define measures for cell and interior face integrations
    dΩ = Measure(CellDomain(mesh), degquad)
    dΓ = Measure(InteriorFaceDomain(mesh), degquad)

    # Declare boundary conditions and
    # create associated domains and measures
    Γ_wall      = BoundaryFaceDomain(mesh, ("NACA",))
    Γ_farfield  = BoundaryFaceDomain(mesh, ("FARFIELD",))
    dΓ_wall     = Measure(Γ_wall, degquad)
    dΓ_farfield = Measure(Γ_farfield, degquad)

    params = (
        params...,
        Γ_wall = Γ_wall,
        dΓ = dΓ,
        dΩ = dΩ,
        dΓ_wall = dΓ_wall,
        dΓ_farfield = dΓ_farfield,
    )

    qLowOrder = nothing

    for deg in 0:degree
        params = (params..., degree = deg)

        fs = FunctionSpace(fspace, deg)
        Q_sca = TrialFESpace(fs, mesh, :discontinuous; size = 1) # DG, scalar
        Q_vec = TrialFESpace(fs, mesh, :discontinuous; size = 2) # DG, vectoriel
        V_sca = TestFESpace(Q_sca)
        V_vec = TestFESpace(Q_vec)
        Q = MultiFESpace(Q_sca, Q_vec, Q_sca)
        V = MultiFESpace(V_sca, V_vec, V_sca)

        q = FEFunction(Q)

        # select an initial configurations:
        if deg == 0
            init!(q, mesh, stateInit)
        else
            println("Start projection")
            projection_l2!(q, qLowOrder, dΩ)
            println("End projection")
        end

        # create CellData to store limiter values
        limρ = Bcube.MeshCellData(ones(ncells(mesh)))
        limAll = Bcube.MeshCellData(ones(ncells(mesh)))
        params = (params..., limρ = limρ, limAll = limAll)

        # Init vtk handler
        mkpath(outputpath)
        vtk = VtkHandler(
            outputpath * "euler_naca_mdeg" * string(mesh_degree) * "_deg" * string(deg),
        )

        # Init time
        time = 0.0

        # Save initial solution
        append_vtk(vtk, mesh, q, time, params)

        # Build the cache and store everything you want to compute only once (such as the mass matrice inverse...)

        cache = ()
        # Allocate buffer for compute_residual
        b_vol = zeros(Bcube.get_ndofs(Q))
        b_fac = zeros(Bcube.get_ndofs(Q))
        cache = (cache..., b_vol = b_vol, b_fac = b_fac)

        cache = (
            cache...,
            cacheCellMean = Bcube.build_cell_mean_cache(q, dΩ),
            mass = factorize(Bcube.build_mass_matrix(Q, V, dΩ)),
            mass_sca = factorize(Bcube.build_mass_matrix(Q_sca, V_sca, dΩ)),
            mass_vec = factorize(Bcube.build_mass_matrix(Q_vec, V_vec, dΩ)),
        )

        time, q = steady_solve!(Q, V, q, mesh, params, cache, vtk, deg)
        append_vtk(vtk, mesh, q, time, params)
        println("end steady_solve for deg=", deg, " !")

        deg < degree && (qLowOrder = deepcopy(q))
    end
    return nothing
end

function steady_solve!(Q, V, q, mesh, params, cache, vtk, deg)
    counter = [0]
    q0 = deepcopy(get_dof_values(q))
    ode_params =
        (Q = Q, V = V, params = params, cache = cache, counter = counter, vtk = vtk)

    rhs!(dq, q, p, t) = dq .= compute_residual(q, p.Q, p.V, p.params)

    # compute sparsity pattern and coloring
    println("computing jacobian cache...")
    if withbench
        _rhs!(dq, q) = rhs!(dq, q, ode_params, 0.0)
        @btime $_rhs!(similar($q0), $q0)
        q_bench = FEFunction(Q, q0)
        @btime $apply_limitation!($q_bench, $ode_params)
        @show length(q0)
    end

    #sparsity_pattern = Symbolics.jacobian_sparsity(_rhs!, similar(Q0), Q0)
    #tjac = @elapsed Symbolics.jacobian_sparsity(_rhs!, similar(Q0), Q0)
    #@show tjac
    sparsity_pattern = Bcube.build_jacobian_sparsity_pattern(Q, mesh)
    println("sparsity pattern computed !")
    display(sparsity_pattern)
    colors = matrix_colors(sparsity_pattern)
    println("coloring done!")
    @show maximum(colors)

    ode = ODEFunction(
        rhs!;
        mass_matrix = Bcube.build_mass_matrix(Q, V, params.dΩ),
        jac_prototype = sparsity_pattern,
        colorvec = colors,
    )

    Tfinal      = Inf
    problem     = ODEProblem(ode, q0, (0.0, Tfinal), ode_params)
    timestepper = ImplicitEuler(; nlsolve = NLNewton(; max_iter = 20))

    cb_cache  = DiscreteCallback(always_true, update_cache!; save_positions = (false, false))
    cb_vtk    = DiscreteCallback(always_true, output_vtk; save_positions = (false, false))
    cb_steady = TerminateSteadyState(1e-6, 1e-6, condition_steadystate)

    error = 1e-1

    sol = solve(
        problem,
        timestepper;
        initializealg = NoInit(),
        adaptive = true,
        abstol = error,
        reltol = error,
        progress = false,
        progress_steps = 1000,
        save_everystep = false,
        save_start = false,
        save_end = false,
        isoutofdomain = isoutofdomain,
        callback = CallbackSet(cb_cache, cb_vtk, cb_steady),
    )

    set_dof_values!(q, sol.u[end])
    return sol.t[end], q
end

always_true(args...) = true

function isoutofdomain(dof, p, t)
    any(isnan, dof) && return true

    q = FEFunction(p.Q, dof)
    q_mean = map(get_values, Bcube.cell_mean(q, p.cache.cacheCellMean))
    p_mean = pressure.(q_mean..., stateInit.γ)

    negative_ρ = any(x -> x < 0, q_mean[1])
    negative_p = any(x -> x < 0, p_mean)
    isout = negative_ρ || negative_p
    isout && @show negative_ρ, negative_p
    return isout
end

function update_cache!(integrator)
    Q = integrator.p.Q
    Q1, = Q
    deg = get_degree(Bcube.get_function_space(Q1))
    println(
        "deg=",
        deg,
        " update_cache! : iter=",
        integrator.p.counter[1],
        " dt=",
        integrator.dt,
    )

    q = FEFunction(integrator.p.Q, integrator.u)
    limiter_projection && apply_limitation!(q, integrator.p)
    return nothing
end

function output_vtk(integrator)
    u_modified!(integrator, false)
    mesh = get_mesh(get_domain(integrator.p.params.dΩ))
    q = FEFunction(integrator.p.Q, integrator.u)
    counter = integrator.p.counter
    counter .+= 1
    if (counter[1] % nout == 0)
        println("output_vtk ", counter[1])
        append_vtk(integrator.p.vtk, mesh, q, integrator.t, integrator.p.params)
    end
    return nothing
end

function condition_steadystate(integrator, abstol, reltol, min_t)
    u_modified!(integrator, false)
    if DiffEqBase.isinplace(integrator.sol.prob)
        testval = first(get_tmp_cache(integrator))
        @. testval = (integrator.u - integrator.uprev) / (integrator.t - integrator.tprev)
    else
        testval = (integrator.u - integrator.uprev) / (integrator.t - integrator.tprev)
    end

    if typeof(integrator.u) <: Array
        any(
            abs(d) > abstol && abs(d) > reltol * abs(u) for (d, abstol, reltol, u) in
            zip(testval, Iterators.cycle(abstol), Iterators.cycle(reltol), integrator.u)
        ) && (return false)
    else
        any((abs.(testval) .> abstol) .& (abs.(testval) .> reltol .* abs.(integrator.u))) &&
            (return false)
    end

    if min_t === nothing
        return true
    else
        return integrator.t >= min_t
    end
end

"""
Compute the characteristic dimension of each cell of `mesh`:
dimcar = (cell volume) / (cell surface)

# TODO :
to be moved to Bcube
"""
function compute_dimcar(mesh)
    fs = FunctionSpace(:Lagrange, 0)
    V = TestFESpace(fs, mesh; size = 1, isContinuous = false)

    # Define measures for cell and interior face integrations
    dΩ = Measure(CellDomain(mesh), degquad)
    dΓ = Measure(InteriorFaceDomain(mesh), degquad)
    dΓ_bc = Measure(BoundaryFaceDomain(mesh), degquad)

    f1 = PhysicalFunction(x -> 1.0)
    l(v) = ∫(f1 ⋅ v)dΩ
    l_face(v, dω) = ∫(side⁻(f1) ⋅ side⁻(v) + side⁺(f1) ⋅ side⁺(v))dω

    vol = assemble_linear(l, V)
    surf = assemble_linear(Base.Fix2(l_face, dΓ), V)
    surf += assemble_linear(Base.Fix2(l_face, dΓ_bc), V)
    return vol ./ surf
end

"""
References:
* Xiangxiong Zhang, Chi-Wang Shu, On positivity-preserving high order discontinuous
  Galerkin schemes for compressible Euler equations on rectangular meshes,
  Journal of Computational Physics, Volume 229, Issue 23, 2010.
  https://doi.org/10.1016/j.jcp.2010.08.016
* Zhang, X., Xia, Y. & Shu, CW. Maximum-Principle-Satisfying and Positivity-Preserving
  High Order Discontinuous Galerkin Schemes for Conservation Laws on Triangular Meshes.
  J Sci Comput 50, 29–62 (2012). https://doi.org/10.1007/s10915-011-9472-8
"""
function apply_limitation!(q::Bcube.AbstractFEFunction, ode_params)
    params = ode_params.params
    cache = ode_params.cache
    mesh = get_mesh(get_domain(params.dΩ))
    ρ, ρu, ρE = q

    ρ_mean, ρu_mean, ρE_mean = Bcube.cell_mean(q, cache.cacheCellMean)

    _limρ, ρ_proj = linear_scaling_limiter(
        ρ,
        params.dΩ;
        bounds = (ρmin₀, ρmax₀),
        DMPrelax = params.DMPrelax,
        mass = cache.mass_sca,
    )

    op_t = limiter_param_p ∘ (ρ_proj, ρu, ρE, ρ_mean, ρu_mean, ρE_mean)
    t = Bcube._minmax_cells(op_t, mesh, Val(params.degquad))
    tmin = Bcube.MeshCellData(getindex.(t, 1))

    if eltype(_limρ) == eltype(params.limρ) # skip Dual number case
        set_values!(params.limρ, get_values(_limρ))
        set_values!(params.limAll, get_values(tmin))
    end

    limited_var(u, ū, lim_u) = ū + lim_u * (u - ū)
    projection_l2!(ρ, limited_var(ρ_proj, ρ_mean, tmin), params.dΩ; mass = cache.mass_sca)
    projection_l2!(ρu, limited_var(ρu, ρu_mean, tmin), params.dΩ; mass = cache.mass_vec)
    projection_l2!(ρE, limited_var(ρE, ρE_mean, tmin), params.dΩ; mass = cache.mass_sca)
    return nothing
end

function limiter_param_p(ρ̂, ρu, ρE, ρ_mean, ρu_mean, ρE_mean)
    γ = stateInit.γ
    p = pressure(ρ̂, ρu, ρE, γ)

    if p ≥ pmin₀
        t = 1.0
    else
        @show p, ρ̂, ρu, ρE
        @show ρ_mean, ρu_mean, ρE_mean
        @show pressure(ρ_mean, ρu_mean, ρE_mean, γ)
        if pressure(ρ_mean, ρu_mean, ρE_mean, γ) > pmin₀
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
        else
            t = NaN
            println("t = NaN")
        end
    end

    return t
end

function pressure(ρ::Number, ρu::AbstractVector, ρE::Number, γ)
    vel = ρu ./ ρ
    ρuu = ρu * transpose(vel)
    p = (γ - 1) * (ρE - tr(ρuu) / 2)
    return p
end

compute_Pᵢ(P, γ, M) = P * (1 + 0.5 * (γ - 1) * M^2)^(γ / (γ - 1))
compute_Tᵢ(T, γ, M) = T * (1 + 0.5 * (γ - 1) * M^2)

function bc_state_farfield(AoA, M, P, T, r, γ)
    a = √(γ * r * T)
    vn = M * a
    ρ = P / r / T
    ρu = SA[ρ * vn * cos(AoA), ρ * vn * sin(AoA)]
    ρE = P / (γ - 1) + 0.5 * ρ * vn^2
    return (ρ, ρu, ρE)
end

function pressure_coefficient(ρ, ρu, ρE)
    (pressure(ρ, ρu, ρE, stateInit.γ) - stateInit.P_inf) /
    (stateBcFarfield.Pᵢ_inf - stateInit.P_inf)
end
function mach(ρ, ρu, ρE)
    norm(ρu ./ ρ) / √(stateInit.γ * max(0.0, pressure(ρ, ρu, ρE, stateInit.γ) / ρ))
end

const degreemax = 2 # Function-space degree
const mesh_degree = 2
const fspace = :Lagrange
const limiter_projection = true
const ρmin₀ = 1.0e-8
const ρmax₀ = 1.0e+10
const pmin₀ = 1.0e-8
const pmax₀ = 1.0e+10
const DMPcurv₀ = 10.0e3
const withbench = false

const stateInit = (
    AoA = deg2rad(1.25),
    M_inf = 0.8,
    P_inf = 101325.0,
    T_inf = 275.0,
    r_gas = 287.0,
    γ = 1.4,
)
const nite_max = 300 #300000 # Number of time iteration(s)
const nout = 1 # number of step between two vtk outputs
const mass_matrix_in_solve = true
const degquad = 6
const outputpath = string(@__DIR__, "/../myout/euler_naca_steady/")

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

main(stateInit, stateBcFarfield, degreemax)

end #hide
