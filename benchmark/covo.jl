module Covo #hide
println("Running covo example...") #hide

const dir = string(@__DIR__, "/")
using Bcube
using LinearAlgebra
using StaticArrays
using Profile
using StaticArrays
using InteractiveUtils
using BenchmarkTools
using UnPack

function compute_residual(_u, V, params, cache)
    u = get_fe_functions(_u)

    # alias on measures
    @unpack dΩ, dΓ, dΓ_perio_x, dΓ_perio_y = params

    # face normals for each face domain (lazy, no computation at this step)
    nΓ = get_face_normals(dΓ)
    nΓ_perio_x = get_face_normals(dΓ_perio_x)
    nΓ_perio_y = get_face_normals(dΓ_perio_y)

    # flux residuals from faces for all variables
    function l(v)
        ∫(flux_Ω(u, v))dΩ +
        -∫(flux_Γ(u, v, nΓ))dΓ +
        -∫(flux_Γ(u, v, nΓ_perio_x))dΓ_perio_x +
        -∫(flux_Γ(u, v, nΓ_perio_y))dΓ_perio_y
    end

    rhs = assemble_linear(l, V)

    return cache.mass \ rhs
end

"""
    flux_Ω(u, v)

Compute volume residual using the lazy-operators approach
"""
flux_Ω(u, v) = _flux_Ω ∘ cellvar(u, v)
cellvar(u, v) = (u, map(∇, v))
function _flux_Ω(u, ∇v)
    ρ, ρu, ρE, ϕ = u
    ∇λ_ρ, ∇λ_ρu, ∇λ_ρE, ∇λ_ϕ = ∇v

    vel = ρu ./ ρ
    ρuu = ρu * transpose(vel)
    p   = pressure(ρ, ρu, ρE, γ)

    flux_ρ  = ρu
    flux_ρu = ρuu + p * I
    flux_ρE = (ρE + p) .* vel
    flux_ϕ  = ϕ .* vel

    return ∇λ_ρ ⋅ flux_ρ + ∇λ_ρu ⊡ flux_ρu + ∇λ_ρE ⋅ flux_ρE + ∇λ_ϕ ⋅ flux_ϕ
end

"""
    flux_Γ(u,v,n)

Flux at the interface is defined by a composition of two functions:
* facevar(u,v,n) defines the input states which are needed for
  the riemann flux using operator notations
* flux_roe(w) defines the Riemann flux (as usual)
"""
flux_Γ(u, v, n) = flux_roe ∘ (side⁻(u), side⁺(u), jump(v), side⁻(n))

"""
    flux_roe(w)
"""
function flux_roe(ui, uj, δv, nij)
    # destructuring inputs given by `facevar` function

    nx, ny = nij
    ρ1, ρu1, ρE1, ϕ1 = ui
    ρ2, ρu2, ρE2, ϕ2 = uj
    δλ_ρ1, δλ_ρu1, δλ_ρE1, δλ_ϕ1 = δv
    ρux1, ρuy1 = ρu1
    ρux2, ρuy2 = ρu2

    # Closure
    u1 = ρux1 / ρ1
    v1 = ρuy1 / ρ1
    u2 = ρux2 / ρ2
    v2 = ρuy2 / ρ2
    p1 = pressure(ρ1, ρu1, ρE1, γ)
    p2 = pressure(ρ2, ρu2, ρE2, γ)

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

    v₁₂ = 0.5 * ((u1 + u2) * nx + (v1 + v2) * ny)
    fluxϕ = max(0.0, v₁₂) * ϕ1 + min(0.0, v₁₂) * ϕ2

    return (
        δλ_ρ1 * (flu11 + duv1) +
        δλ_ρu1 ⋅ (SA[flu21 + duv2, flu31 + duv3]) +
        δλ_ρE1 * (flu41 + duv4) +
        δλ_ϕ1 * (fluxϕ)
    )
end

"""
Time integration of `f(q, t)` over a timestep `Δt`.
"""
forward_euler(q, f::Function, t, Δt) = get_dof_values(q) .+ Δt .* f(q, t)

"""
    rk3_ssp(q, f::Function, t, Δt)

`f(q, t)` is the function to integrate.
"""
function rk3_ssp(q, f::Function, t, Δt)
    stepper(q, t) = forward_euler(q, f, t, Δt)
    _q0 = get_dof_values(q)

    _q1 = stepper(q, Δt)

    set_dof_values!(q, _q1)
    _q2 = (3 / 4) .* _q0 .+ (1 / 4) .* stepper(q, t + Δt)

    set_dof_values!(q, _q2)
    _q1 .= (1 / 3) * _q0 .+ (2 / 3) .* stepper(q, t + Δt / 2)

    return _q1
end

"""
    pressure(ρ, ρu, ρE, γ)

Computes pressure from perfect gaz law
"""
function pressure(ρ::Number, ρu::AbstractVector, ρE::Number, γ)
    vel = ρu ./ ρ
    ρuu = ρu * transpose(vel)
    p = (γ - 1) * (ρE - tr(ρuu) / 2)
    return p
end

"""
  Init field with a vortex (for the COVO test case)
"""
function covo!(q, dΩ)

    # Intermediate vars
    Cₚ = γ * r / (γ - 1)

    r²(x) = ((x[1] .- xvc) .^ 2 + (x[2] .- yvc) .^ 2) ./ Rc^2
    # Temperature
    T(x) = T₀ .- β^2 * U₀^2 / (2 * Cₚ) .* exp.(-r²(x))
    # Velocity
    ux(x) = U₀ .- β * U₀ / Rc .* (x[2] .- yvc) .* exp.(-r²(x) ./ 2)
    uy(x) = V₀ .+ β * U₀ / Rc .* (x[1] .- xvc) .* exp.(-r²(x) ./ 2)
    # Density
    ρ(x) = ρ₀ .* (T(x) ./ T₀) .^ (1.0 / (γ - 1))
    # momentum
    ρu(x) = SA[ρ(x) * ux(x), ρ(x) * uy(x)]
    # Energy
    ρE(x) = ρ(x) * ((Cₚ / γ) .* T(x) + (ux(x) .^ 2 + uy(x) .^ 2) ./ 2)
    # Passive scalar
    ϕ(x) = Rc^2 * r²(x) < 0.01 ? exp(-r²(x) ./ 2) : 0.0

    f = map(PhysicalFunction, (ρ, ρu, ρE, ϕ))
    projection_l2!(q, f, dΩ)
    return nothing
end

# Settings
if get(ENV, "BenchmarkMode", "false") == "false" #hide
    const cellfactor = 1
    const nx = 32 * cellfactor + 1
    const ny = 32 * cellfactor + 1
    const fspace = :Lagrange
    const timeScheme = :ForwardEuler
else #hide
    const nx = 128 + 1 #hide
    const ny = 128 + 1 #hide
    const fspace = :Lagrange
    const timeScheme = :ForwardEuler
end #hide
const nperiod = 1 # number of turn
const CFL = 0.1
const degree = 1 # FunctionSpace degree
const degquad = 2 * degree + 1
const γ = 1.4
const β = 0.2 # vortex intensity
const r = 287.15 # Perfect gaz constant
const T₀ = 300 # mean-flow temperature
const P₀ = 1e5 # mean-flow pressure
const M₀ = 0.5 # mean-flow mach number
const ρ₀ = 1.0 # mean-flow density
const xvc = 0.0 # x-center of vortex
const yvc = 0.0 # y-center of vortex
const Rc = 0.005 # Charasteristic vortex radius
const c₀ = √(γ * r * T₀) # Sound velocity
const U₀ = M₀ * c₀ # mean-flow velocity
const V₀ = 0.0 # mean-flow velocity
const ϕ₀ = 1.0
const l = 0.05 # half-width of the domain
const Δt = CFL * 2 * l / (nx - 1) / ((1 + β) * U₀ + c₀) / (2 * degree + 1)
#const Δt = 5.e-7
const nout = 100 # Number of time steps to save
const nite = Int(floor(nperiod * 2 * l / (U₀ * Δt))) + 1

function run_covo()
    println("Starting run_covo...")

    # Build mesh
    mesh = rectangle_mesh(
        nx,
        ny;
        xmin = -l,
        xmax = l,
        ymin = -l,
        ymax = l,
        bnd_names = ("West", "East", "South", "North"),
    )
    if (get(ENV, "BenchmarkMode", "false") == "true") &&
       (get(ENV, "MeshConfig", "quad") == "triquad") #hide
        mesh = read_mesh(
            joinpath(
                @__DIR__,
                "assets",
                "rectangle-mesh-tri-quad-nx129-ny129-lx1e-1-ly1e-1.msh22",
            );
            warn = false,
        )
    end #hide

    # Define variables and test functions
    fs = FunctionSpace(fspace, degree)
    U_sca = TrialFESpace(fs, mesh, :discontinuous; size = 1) # DG, scalar
    U_vec = TrialFESpace(fs, mesh, :discontinuous; size = 2) # DG, vectoriel
    V_sca = TestFESpace(U_sca)
    V_vec = TestFESpace(U_vec)
    U = MultiFESpace(U_sca, U_vec, U_sca, U_sca)
    V = MultiFESpace(V_sca, V_vec, V_sca, V_sca)
    u = FEFunction(U)

    @show Bcube.get_ndofs(U)

    # Define measures for cell and interior face integrations
    dΩ = Measure(CellDomain(mesh), degquad)
    dΓ = Measure(InteriorFaceDomain(mesh), degquad)

    # Declare periodic boundary conditions and
    # create associated domains and measures
    periodicBCType_x = PeriodicBCType(Translation(SA[-2l, 0.0]), ("East",), ("West",))
    periodicBCType_y = PeriodicBCType(Translation(SA[0.0, 2l]), ("South",), ("North",))
    Γ_perio_x = BoundaryFaceDomain(mesh, periodicBCType_x)
    Γ_perio_y = BoundaryFaceDomain(mesh, periodicBCType_y)
    dΓ_perio_x = Measure(Γ_perio_x, degquad)
    dΓ_perio_y = Measure(Γ_perio_y, degquad)

    params = (dΩ = dΩ, dΓ = dΓ, dΓ_perio_x = dΓ_perio_x, dΓ_perio_y = dΓ_perio_y)

    # Init solution
    t = 0.0

    covo!(u, dΩ)

    # cache mass matrices
    cache = (mass = factorize(Bcube.build_mass_matrix(U, V, dΩ)),)

    if get(ENV, "BenchmarkMode", "false") == "true" #hide
        return u, U, V, params, cache
    end

    # Time loop
    for i in 1:nite
        println("")
        println("")
        println("Iteration ", i, " / ", nite)

        ## Step forward in time
        rhs(u, t) = compute_residual(u, V, params, cache)
        if timeScheme == :ForwardEuler
            unew = forward_euler(u, rhs, time, Δt)
        elseif timeScheme == :RK3
            unew = rk3_ssp(u, rhs, time, Δt)
        else
            error("Unknown time scheme: $timeScheme")
        end

        set_dof_values!(u, unew)

        t += Δt
    end

    # Summary and benchmark                                 # ndofs total = 20480
    _rhs(u, t) = compute_residual(u, V, params, cache)
    @btime forward_euler($u, $_rhs, $time, $Δt)  # 5.639 ms (1574 allocations: 2.08 MiB)
    # stepper = w -> explicit_step(w, params, cache, Δt)
    # RK3_SSP(stepper, (u, v), cache)
    # @btime RK3_SSP($stepper, ($u, $v), $cache)
    println("ndofs total = ", Bcube.get_ndofs(U))
    Profile.init(; n = 10^7) # returns the current settings
    Profile.clear()
    Profile.clear_malloc_data()
    @profile begin
        for i in 1:100
            forward_euler(u, _rhs, time, Δt)
        end
    end
    @show Δt, U₀, U₀ * t
    @show boundary_names(mesh)
    return nothing
end

if get(ENV, "BenchmarkMode", "false") == "false"
    run_covo()
end

end #hide
