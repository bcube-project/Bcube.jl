module LinearTransportGpu

using Bcube
using LinearAlgebra
using KernelAbstractions
include("./adapt.jl")
include(joinpath(@__DIR__, "BcubeGPU.jl"))
using .BcubeGPU
using CUDA
using BcubeVTK

const nx = 50
const ny = 50
const nite = 400
const degree = 0
const c = SA[1.0, 0.0] # Convection velocity (must be a vector)
const CFL = 0.2
const Δt = CFL * min(1.0 / nx, 1.0 / ny) / norm(c)
const nout = 20

mutable struct VtkHandler
    basename::Any
    ite::Any
    mesh::Any
    VtkHandler(basename, mesh) = new(basename, 0, mesh)
end

function append_vtk(vtk, u::Bcube.AbstractFEFunction, t)
    # Write
    write_file(
        vtk.basename,
        vtk.mesh,
        Dict("u" => u),
        vtk.ite,
        t;
        discontinuous = true,
        collection_append = vtk.ite > 0,
    )

    # Update counter
    vtk.ite += 1
end

bc_in(t) = PhysicalFunction(x -> c .* cos(3 * x[2])) * sin(4 * t)

function upwind(ui, uj, nij)
    cij = c ⋅ nij
    if cij > zero(cij)
        flux = cij * ui
    else
        flux = cij * uj
    end
    flux
end

function main(nx, ny, nite, degree, backend)
    mesh_cpu = rectangle_mesh(nx, ny)
    mesh = adapt(backend, mesh_cpu)

    fs = FunctionSpace(:Lagrange, degree)
    U_cpu = TrialFESpace(fs, mesh_cpu, :discontinuous)
    U = adapt(backend, U_cpu)
    V = TestFESpace(U)
    u = FEFunction(U, KernelAbstractions.zeros(backend, Float64, get_ndofs(U)))

    Γ = InteriorFaceDomain(mesh)
    Γ_in = BoundaryFaceDomain(mesh, (:xmin,))
    Γ_out = BoundaryFaceDomain(mesh, (:xmax, :ymin, :ymax))

    dΩ = Measure(CellDomain(mesh), 2 * degree + 1)
    dΓ = Measure(Γ, 2 * degree + 1)
    dΓ_in = Measure(Γ_in, 2 * degree + 1)
    dΓ_out = Measure(Γ_out, 2 * degree + 1)

    println("Building normals")

    nΓ = get_face_normals(Γ)
    nΓ_in = get_face_normals(Γ_in)
    nΓ_out = get_face_normals(Γ_out)

    println("Building weak forms")

    m(u, v) = ∫(u ⋅ v)dΩ # Mass matrix
    l_Ω(v) = ∫((c * u) ⋅ ∇(v))dΩ

    l_Γ(v) = ∫((upwind ∘ (side⁻(u), side⁺(u), side⁻(nΓ))) * jump(v))dΓ
    l_Γ_in(v, t) = ∫((side⁻(bc_in(t)) ⋅ side⁻(nΓ_in)) * side⁻(v))dΓ_in
    l_Γ_out(v) = ∫((upwind ∘ (side⁻(u), 0.0, side⁻(nΓ_out))) * side⁻(v))dΓ_out

    println("Building mass matrix")

    M = assemble_bilinear(m, U, V; backend = backend)
    factoM = cholesky(adapt(backend, Array(M))) # TODO : avoid dense matrix

    ## Allocate buffers for linear assembling
    b_vol = KernelAbstractions.ones(backend, Float64, get_ndofs(U))
    b_fac = similar(b_vol)
    rhs = similar(b_vol)

    println("Starting time loop")

    t = 0.0

    # Write to file
    out_dir = joinpath(@__DIR__, "myout", "linear_transport")
    mkpath(out_dir)
    vtk = VtkHandler(joinpath(out_dir, "linear_transport.pvd"), mesh_cpu)
    u_cpu = adapt(get_backend(zeros(1)), u)
    append_vtk(vtk, u_cpu, t)

    for i in 1:nite
        (i % nout == 0) && println("$i / $nite")

        ## Reset pre-allocated vectors
        b_vol .= 0.0
        b_fac .= 0.0

        # Assembling linear form
        assemble_linear!(b_vol, l_Ω, V)
        assemble_linear!(b_fac, l_Γ, V)
        assemble_linear!(b_fac, l_Γ_out, V)
        tᵢ = copy(t)
        assemble_linear!(b_fac, v -> l_Γ_in(v, tᵢ), V)

        ## Compute rhs
        rhs .= Δt .* (factoM \ (b_vol - b_fac))

        ## Update solution
        u.dofValues .+= rhs

        ## Update time
        t += Δt

        if (i % nout) == 0
            u_cpu = adapt(get_backend(zeros(1)), u)
            append_vtk(vtk, u_cpu, t)
        end
    end
end

#main(nx, ny, nite, degree, get_backend(ones(2)))       ## CPU
main(nx, ny, nite, degree, get_backend(CUDA.ones(2)))   ## GPU

end