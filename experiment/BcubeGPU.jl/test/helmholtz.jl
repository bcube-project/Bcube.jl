function run_helmholtz(backend)
    mesh_cpu = rectangle_mesh(21, 21)
    mesh = adapt(backend, mesh_cpu)

    degree = 1
    fs = FunctionSpace(:Lagrange, degree)

    U_cpu = TrialFESpace(fs, mesh_cpu)
    U = adapt(backend, U_cpu)
    V_cpu = TestFESpace(U_cpu)
    V = adapt(backend, V_cpu)

    Ω_cpu = CellDomain(mesh_cpu)
    Ω = CellDomain(mesh)

    dΩ = Measure(Ω, 2 * degree + 1)

    # Build ReverseDofHandler
    rdhl_cpu = ReverseDofHandler(Ω_cpu, V_cpu)
    rdhl = adapt(backend, rdhl_cpu)

    # Build "storage" indirection for bilinear assembly
    ind_cpu = BcubeGPU.build_bilinear_storage_ind(U_cpu, V_cpu, rdhl_cpu)
    ind = adapt(backend, ind_cpu)

    a(u, v) = ∇(u) ⋅ ∇(v)
    b(u, v) = u ⋅ v

    A = BcubeGPU.kernabs_assemble_bilinear(backend, a, U, V, dΩ, rdhl, ind)
    B = BcubeGPU.kernabs_assemble_bilinear(backend, b, U, V, dΩ, rdhl, ind)

    # we could maybe compute the eigenvalues using KrylovKit or something like this

    # Ref solution on CPU
    dΩ_cpu = Measure(Ω_cpu, 2 * degree + 1)
    _a(u, v) = ∫(a(u, v))dΩ_cpu
    _b(u, v) = ∫(b(u, v))dΩ_cpu
    A = assemble_bilinear(_a, U_cpu, V_cpu)
    B = assemble_bilinear(_b, U_cpu, V_cpu)

    return (A == sparse(Array.(findnz(A))...)) && (B == sparse(Array.(findnz(B))...))
end