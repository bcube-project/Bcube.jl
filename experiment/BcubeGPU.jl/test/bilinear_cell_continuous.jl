function run_bilinear_cell_continuous(backend)
    # Mesh and domains
    mesh_cpu = rectangle_mesh(2, 4)
    mesh = adapt(backend, mesh_cpu)
    test_arg(backend, mesh)
    println("mesh on GPU!")

    Ω_cpu = CellDomain(mesh_cpu)
    Ω = CellDomain(mesh)
    test_arg(backend, Ω)
    println("Ω on GPU!")

    dΩ = Measure(Ω, 1)
    test_arg(backend, dΩ)
    println("dΩ on GPU!")

    # Build TrialFESpace and TestFESpace
    # The TrialFESpace must be first built on the CPU for now because the
    # underlying DofHandler constructor uses scalar indexing
    U_cpu = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh_cpu)
    U = adapt(backend, U_cpu)
    test_arg(backend, U)
    println("U on GPU!")

    V_cpu = TestFESpace(U_cpu)
    V = TestFESpace(U)
    test_arg(backend, V)
    println("V on GPU!")

    # Build ReverseDofHandler
    rdhl_cpu = ReverseDofHandler(Ω_cpu, V_cpu)
    rdhl = adapt(backend, rdhl_cpu)
    test_arg(backend, rdhl)
    println("rdhl on GPU!")

    # Build "storage" indirection for bilinear assembly
    ind_cpu = BcubeGPU.build_bilinear_storage_ind(U_cpu, V_cpu, rdhl_cpu)
    ind = adapt(backend, ind_cpu)

    # Define bilinear form and assemble
    f(u, v) = u ⋅ v
    A = BcubeGPU.kernabs_assemble_bilinear(backend, f, U, V, dΩ, rdhl, ind)
    _I, _J, _V = findnz(A)
    res_from_gpu = sparse(Array(_I), Array(_J), Array(_V))
    display(res_from_gpu)

    # Compare with CPU result
    a(u, v) = ∫(f(u, v))Measure(CellDomain(mesh_cpu), 1)
    res_cpu = assemble_bilinear(a, U_cpu, V_cpu)
    println("Result on CPU:")
    display(res_cpu)

    return (; res_cpu, res_from_gpu)

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end