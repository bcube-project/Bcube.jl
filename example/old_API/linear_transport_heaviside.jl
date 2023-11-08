module LinearTransport #hide
println("Running linear transport heaviside example...") #hide
# # Linear transport
# ## Theory
# In this example, we solve the following linear transport equation using discontinuous elements:
# ```math
# \frac{\partial \phi}{\partial t} + \nabla \cdot (c \phi) = 0
# ```
# where ``c`` is a constant velocity. Using an explicit time scheme, one obtains:
# ```math
# \phi^{n+1} = \phi^n - \Delta t \nabla \cdot (c \phi^n)
# ```
# The corresponding weak form of this equation is:
# ```math
# \int_\Omega \phi^{n+1} v \mathrm{\,d}\Omega = \int_\Omega \phi^n v \mathrm{\,d}\Omega + \Delta t \left[
# \int_\Omega c \phi^n \cdot \nabla v \mathrm{\,d}\Omega - \oint_\Gamma \left( c \phi \cdot n \right) v \mathrm{\,d}\Gamma
# \right]
# ```
# where ``\Gamma = \delta \Omega``. Adopting the discontinuous Galerkin framework, this equation is written in every mesh cell
# ``\Omega_i``. The cell boundary term involves discontinuous quantities and is replaced by a "numerical flux",
# leading to the expression:
# ```math
# \int_{\Omega_i} \phi^{n+1} v \mathrm{\,d}\Omega_i = \int_{\Omega_i} \phi^n v \mathrm{\,d}\Omega_i + \Delta t \left[
# \int_{\Omega_i} c \phi^n \cdot \nabla v \mathrm{\,d}\Omega_i - \oint_{\Gamma_i} F^*(\phi) v \mathrm{\,d} \Gamma_i
# \right]
# ```
# For this example, an upwind flux will be used for ``F^*``. Using a matrix formulation, the above equation can be written as:
# ```math
# \phi^{n+1} = \phi^n + M^{-1}(f_\Omega - f_\Gamma)
# ```
# where ``M^{-1}`` is the inverse of the mass matrix, ``f_\Omega`` the volumic flux term and ``f_\Gamma`` the surfacic flux term.
#
# ## Solution with Bcube
# Start by importing the necessary packages:
# Load the necessary packages (Bcube is loaded only if not already loaded)
include(string(@__DIR__, "/../src/Bcube.jl"))
using .Bcube
using LinearAlgebra
using WriteVTK
using StaticArrays
using BenchmarkTools

function compute_residual(w, params, t)

    # destructuring : `ϕ` for cellvariable, `λ`for test function
    u, v = w
    ϕ, = u
    λ, = v

    # alias on measures
    dΓ = params.dΓ
    dΩ = params.dΩ
    dΓ_perio_x = params.dΓ_perio_x
    dΓ_perio_y = params.dΓ_perio_y

    c = params.c

    # @show limϕ[917],ϕ̅[917]
    # init a tuple of new CellVariables from `u` to store all residual contributions
    du = zeros.(u)

    # compute volume residuals
    du_Ω = ∫(flux_Ω((ϕ,), v, c))dΩ

    # Store volume residuals in `du`
    du += du_Ω

    # face normals for each face domain (lazy, no computation at this step)
    n_Γ = FaceNormals(dΓ)
    n_Γ_perio_x = FaceNormals(dΓ_perio_x)
    n_Γ_perio_y = FaceNormals(dΓ_perio_y)

    # flux residuals from interior faces for all variables
    du_Γ = ∫(flux_Γ((ϕ,), v, n_Γ))dΓ

    # flux residuals from bc faces for all variables
    du_Γ_perio_x = ∫(flux_Γ((ϕ,), v, n_Γ_perio_x))dΓ_perio_x
    du_Γ_perio_y = ∫(flux_Γ((ϕ,), v, n_Γ_perio_y))dΓ_perio_y

    # accumulate face-flux residuals to cell residuals for all variables
    # (this step will be improved when a better API will be available)
    du = ((du - du_Γ) - du_Γ_perio_x) - du_Γ_perio_y

    return du
end

"""
    flux_Ω(u,v)

Compute volume residual using the lazy-operators approach
"""
function flux_Ω(u, v, c)
    ϕ, = u
    λ, = v
    flux_ϕ = ∇(λ) * (ϕ .* c)
    (flux_ϕ,)
end

# Basically, we need three methods to solve the problem : one to compute the flux terms (looping on the mesh faces), one to assemble
# the problem (volumic + surfacic terms) and one to step forward in time. Before defining these three methods, let's define auxiliary
# ones that help improving the code readability. First we define our numerical flux function, the updwind flux:
"""
  Upwind flux. Face normal oriented from cell i to cell j.
  Here `c` is the constant convection velocity.
"""
function upwind(w)
    (ϕᵢ,), (ϕⱼ,), (λᵢ,), nᵢⱼ = w
    c = SA[1.0, 0.0] #TODO : put it as an input argument
    vij = c ⋅ nᵢⱼ
    # if ϕᵢ<0.0 || ϕᵢ>1.0 || ϕⱼ <0.0 || ϕᵢ > 1.0
    #     @show ϕᵢ, ϕⱼ
    #     error("ldksldksjqjslkd")
    # end
    if vij > zero(vij)
        flux = vij * ϕᵢ
    else
        flux = vij * ϕⱼ
    end
    return (λᵢ * flux,)
end

"""
    flux_Γ(u,v,n)

Flux at the interface is defined by a composition of two functions:
* `facevar(u,v,n)` defines the input states which are needed for
  the upwind flux using operator notations
* `upwind(w)` defines the upwind face flux (as usual)
"""
flux_Γ(u, v, n) = upwind ∘ facevar(u, v, n)
facevar(u, v, n) = (side⁻(u), side⁺(u), side⁻(v), n)

"""
    flux_Γ_out(u,v,n,c,dirichlet_out)
"""
function flux_Γ_out(u, v, n, c, dirichlet_out)
    ϕ, = u
    λ, = v
    cn = c ⋅ n
    flux = max(0, cn) * side⁻(ϕ) + min(0, cn) * dirichlet_out
    (side⁻(λ) * flux,)
end

"""
    flux_Γ_in(u,v,n,bc_in)
"""
function flux_Γ_in(u, v, n, bc_in)
    ϕ, = u
    λ, = v
    flux = bc_in ⋅ n
    (side⁻(λ) * flux,)
end

# Now let's write the assembling method which computes the volumic terms and assembles them with the surfacic flux terms.
# We call this method `explicit_step!` because it returns the ``\Delta \phi`` associated with the selected explicit time
# scheme.
function explicit_step(w, params, cache, Δt, t)
    (ϕ,), (λ,) = w

    if limiter_projection
        bounds = (params.ϕmin₀, params.ϕmax₀)
        perioBCs = (get_domain(params.dΓ_perio_x), get_domain(params.dΓ_perio_y))
        _limϕ, ϕproj = linear_scaling_limiter(
            ϕ,
            params.degquad;
            bounds = bounds,
            DMPrelax = params.DMPrelax,
            periodicBCs = perioBCs,
            invMass = cache.invMass_ϕ,
        )
        set_values!(ϕ, ϕproj)
        params.limϕ .= _limϕ
    end

    _dϕ, = compute_residual(w, params, t)
    ϕnew = get_values(ϕ) .+ cache.invMass_ϕ * (Δt .* get_values(_dϕ))

    return ϕnew
end

# We are all set to solve a linear transport equation. However, we will add two more things to ease the solution VTK output : a
# structure to store the vtk filename and the number of iteration:
mutable struct VtkHandler
    basename::Any
    ite::Any
    VtkHandler(basename) = new(basename, 0)
end

# ... and a method to interpolate the discontinuous solution on cell centers and to append it to the VTK file:
function append_vtk(vtk, mesh, vars, t, params)
    ## Values on center
    # Values on center
    name2val = (; zip(get_name.(vars), var_on_centers.(vars))...)
    name2val_avg = (;
        zip(
            Symbol.(string.(get_name.(vars)) .* "_avg"),
            map(v -> mean_values(v, Val(degquad)), vars),
        )...
    )

    name2val_lim = (limϕ = params.limϕ,)
    name2val = merge(name2val, name2val_avg)
    name2val = merge(name2val, name2val_lim)

    # Write
    dict_vars = Dict(
        "phi_avg" => (name2val[:ϕ_avg], VTKCellData()),
        "phi_center" => (name2val[:ϕ], VTKCellData()),
        "lim_phi" => (name2val[:limϕ], VTKCellData()),
    )

    write_vtk(vtk.basename, vtk.ite, t, mesh, dict_vars; append = vtk.ite > 0)

    # Update counter
    vtk.ite += 1

    return nothing
end

function RK3_SSP(w, params, cache, Δt, t)
    (ϕ,), v = w

    _ϕ₁ = explicit_step(((ϕ,), v), params, cache, Δt, t)
    ϕ₁ = zeros(ϕ)
    set_values!(ϕ₁, _ϕ₁)

    _ϕ₂ = explicit_step(((ϕ₁,), v), params, cache, Δt, t)
    ϕ₂ = zeros(ϕ)
    set_values!(ϕ₂, (3.0 / 4) .* get_values(ϕ) .+ (1.0 / 4) .* (_ϕ₂))

    _ϕ₃ = explicit_step(((ϕ₂,), v), params, cache, Δt, t)
    ϕ₃ = (1.0 / 3) .* get_values(ϕ) .+ (2.0 / 3) .* (_ϕ₃)

    return ϕ₃
end

function check_bounds(ϕ, mesh, params)
    ϕ̅ = mean_values(ϕ, Val(degquad))
    if !(minimum(ϕ̅) > -1.e-12 && maximum(ϕ̅) < 1.0 + 1.e-12)
        @show extrema(ϕ̅)
        imin = argmin(ϕ̅)
        imax = argmax(ϕ̅)
        c2n = connectivities_indices(mesh, :c2n)
        @show center(get_nodes(mesh, c2n[imin]), cells(mesh)[imin])
        @show center(get_nodes(mesh, c2n[imax]), cells(mesh)[imax])
        @show params.limϕ[imin], ϕ̅[imin]
        @show get_values(ϕ, imin)
        @show get_values(ϕ, imax)
        @show params.limϕ[imax], ϕ̅[imax]
        error("jkojojo")
    end
end

heaviside!(x, x₀, l₀) = all(abs.(x .- x₀) .< l₀ / 2) ? 1.0 : 0.0
sinus!(x, x₀, l₀) = 0.5 * (1 + sin(2π * 2 * (x[1] - x₀) / l₀ + π / 2))
function step3!(x, x₀, l₀)
    if x[1] < x₀ - l₀ / 2
        0.0
    elseif x[1] > x₀ + l₀ / 2
        1.0
    else
        ((x[1] - (x₀ - l₀ / 2)) / l₀)^2 * (3 - 2(x[1] - (x₀ - l₀ / 2)) / l₀)
    end
end

function run()
    @show degmass, degquad

    # Then generate the mesh of a rectangle using Gmsh and read it
    tmp_path = "tmp.msh"
    gen_rectangle_mesh(
        tmp_path,
        :quad;
        nx = nx,
        ny = ny,
        lx = lx,
        ly = ly,
        xc = 0.0,
        yc = 0.0,
    )
    mesh = read_msh(tmp_path)
    rm(tmp_path)

    # Create a `CellVariable`
    fs = FunctionSpace(fspace, degree)
    fes = FESpace(fs, :discontinuous; size = 1) #  size=1 for scalar variable
    ϕ = CellVariable(:ϕ, mesh, fes)

    # select an initial configurations:
    set_values!(ϕ, x -> heaviside!(x, x₀, l₀ * 0.999); degquad = degquad)
    #set_values!(ϕ, x-> x[1]<0.0 ? step3!(-x,lx/4,8*lx/(nx+1)) : step3!(x,lx/4,8*lx/(nx+1)) )
    #set_values!(ϕ, x-> sinus!(-x,lx/2,2lx))

    # user-defined limiter parameters
    DMPrelax = zeros(ncells(mesh))
    DMPrelax .= DMPcurv₀ * Δx₀^2
    params = (ϕmin₀ = ϕmin₀, ϕmax₀ = ϕmax₀, DMPrelax = DMPrelax)

    # Create a `TestFunction`
    λ = TestFunction(mesh, fes)

    u, v = ((ϕ,), (λ,))

    # Then we create a `NamedTuple` to hold the simulation parameters.
    params = (params..., c = c, degree = degree, degquad = degquad)

    # Define measures for cell and interior face integrations
    dΩ = Measure(CellDomain(mesh), degquad)
    dΓ = Measure(InteriorFaceDomain(mesh), degquad)

    # Declare periodic boundary conditions and
    # create associated domains and measures
    periodicBCType_x = PeriodicBCType(Translation(SA[-lx, 0.0]), ("East",), ("West",))
    periodicBCType_y = PeriodicBCType(Translation(SA[0.0, ly]), ("South",), ("North",))
    Γ_perio_x = BoundaryFaceDomain(mesh, periodicBCType_x)
    Γ_perio_y = BoundaryFaceDomain(mesh, periodicBCType_y)
    dΓ_perio_x = Measure(Γ_perio_x, degquad)
    dΓ_perio_y = Measure(Γ_perio_y, degquad)

    params = (params..., dΓ = dΓ, dΩ = dΩ, dΓ_perio_x = dΓ_perio_x, dΓ_perio_y = dΓ_perio_y)

    # create a vector `limϕ` to store limiter values
    # and write it in the vtk outputs
    limϕ = zeros(ncells(mesh))
    params = (params..., limϕ = limϕ)

    # Init vtk handler
    vtk = VtkHandler(
        string(@__DIR__, "/../myout/linear_transport_heaviside_deg" * string(degree)),
    )

    # Init time
    time = 0.0

    # Save initial solution
    append_vtk(vtk, mesh, (ϕ,), time, params)

    # Build the cache and store everything you want to compute only once (such as the mass matrice inverse...)
    invMass_ϕ = InvMassMatrix(ϕ, Val(degquad))
    cache = (invMass_ϕ = invMass_ϕ,)

    mass0 = sum(mean_values(ϕ, Val(degquad)))

    if limiter_projection
        bounds = (params.ϕmin₀, params.ϕmax₀)
        perioBCs = (get_domain(params.dΓ_perio_x), get_domain(params.dΓ_perio_y))
        _limϕ, ϕproj = linear_scaling_limiter(
            ϕ,
            degquad;
            bounds = bounds,
            DMPrelax = DMPrelax,
            periodicBCs = perioBCs,
            invMass = invMass_ϕ,
        )
        set_values!(ϕ, ϕproj)
        params.limϕ .= _limϕ
    end

    check_bounds(ϕ, mesh, params)

    ϕold = zeros(ϕ)

    # Let's loop to solve the equation.
    for i in 1:nite
        ## Infos
        println("Iteration ", i)

        set_values!(ϕold, get_values(ϕ))

        ## Step forward in time
        dϕ = RK3_SSP((u, v), params, cache, Δt, time)
        set_values!(ϕ, dϕ)

        any(isnan(x) for x in get_values(ϕ)) && error("NaN")

        time += Δt

        ## Write solution to file (respecting max. number of output)
        if (i % Int(max(floor(nite / nout), 1)) == 0)
            append_vtk(vtk, mesh, (ϕ,), time, params)
        end

        check_bounds(ϕ, mesh, params)

        ratio_mass = sum(mean_values(ϕ, Val(degquad))) / mass0
        println("Volume conservation = ", ratio_mass)
    end

    append_vtk(vtk, mesh, (ϕ,), time, params)

    @show degmass, degquad
    @btime explicit_step(($u, $v), $params, $cache, $Δt, $time)
    @show length(get_values(ϕ))

    # And here is an animation of the result:
    # ```@raw html
    # <img src="../assets/linear_transport_heavidise.gif" alt="drawing" width="700"/>
    # ```
end

# Now let's manipulate everything to solve a linear transport equation. First we define some constants, relative to
# the problem or the mesh :
const degree = 2# Function-space order (Taylor(0) = first order Finite Volume)
const fspace = :Lagrange
const limiter_projection = true

const ϕmin₀ = 0.0 # lower physical bound on cell average value (strictly enforced by limitation)
const ϕmax₀ = 1.0 # upper physical bound on cell average value (strictly enforced by limitation)
const DMPcurv₀ = 10.0 # coef. (≥0) used to relax bounds of the local discrete maximum principle when limitation is applied.
# It allows small oscillations at local extrema while physical bounds are satisfied.
# There is no relaxation if the coef. is set to zero.

const nite = 5 * 1600 # Number of time iteration(s)
const c = SA[1.0, 0.0] # Convection velocity (must be a vector)
const CFL = 1.0 / 20 / 2 # 0.1 for order 1
const nout = 100 # Number of time steps to save
const nx = 40 # Number of nodes in the x-direction
const ny = 40 # Number of nodes in the y-direction
const lx = 2.0 # Domain width
const ly = 2.0 # Domain height
const l₀ = SA[11 * lx / (nx - 1), 11 * ly / (ny - 1)]  # size of heaviside step
const x₀ = SA[0.0, 0.0] # center of heaviside step

const Δt = CFL * min(lx / nx, ly / ny) / norm(c) # Time step
const Δx₀ = lx / nx

const degmass = degree * 2 * 2   # *2 because ndim=2, and *2 because λᵢ*λⱼ
const degquad = 5# Int(ceil((degmass+3)/2))

run()

end #hide
