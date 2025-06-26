function run_linear_face_discontinuous(backend)
    # Mesh and domains
    mesh_cpu = rectangle_mesh(3, 2)
    mesh = adapt(backend, mesh_cpu)
    test_arg(backend, mesh)
    println("mesh on GPU!")

    Γ_cpu = InteriorFaceDomain(mesh_cpu)
    Γ = InteriorFaceDomain(mesh)
    test_arg(backend, Γ)
    println("Γ on GPU!")

    dΓ = Measure(Γ, 1)
    test_arg(backend, dΓ)
    println("dΓ on GPU!")

    # Build TrialFESpace and TestFESpace
    # The TrialFESpace must be first built on the CPU for now because the
    # underlying DofHandler constructor uses scalar indexing
    g(x, t) = 3x[1]
    h(x, t) = 5x[1] + 2

    U_cpu = TrialFESpace(
        FunctionSpace(:Lagrange, 1),
        mesh_cpu,
        :discontinuous,
        Dict("xmin" => 5.0, "xmax" => g, "ymin" => h),
    )
    U = adapt(backend, U_cpu)
    test_arg(backend, U)
    println("U on GPU!")

    V = TestFESpace(U)
    test_arg(backend, V)
    println("V on GPU!")

    # Build ReverseDofHandler
    rdhl_cpu = ReverseDofHandler(Γ_cpu, U_cpu)
    rdhl = adapt(backend, rdhl_cpu)
    test_arg(backend, rdhl)
    println("rdhl on GPU!")

    # Build FEFunction
    u = FEFunction(U, KernelAbstractions.ones(backend, Float64, get_ndofs(U)))
    test_arg(backend, u)
    println("u on GPU!")

    # Define linear form and assemble
    f(φ) = side_n(u) * jump(φ)
    y = KernelAbstractions.zeros(backend, Float64, get_ndofs(U))
    BcubeGPU.kernabs_assemble_linear!(backend, y, f, V, dΓ, rdhl)
    display(y)
    res_from_gpu = Array(y)

    # Compare with CPU result
    u_cpu = FEFunction(U_cpu, ones(get_ndofs(U)))
    f_cpu(φ) = ∫(side_n(u_cpu) * jump(φ))Measure(InteriorFaceDomain(mesh_cpu), 1)
    y_cpu = assemble_linear(f_cpu, TestFESpace(U_cpu))
    println("Result on CPU:")
    display(y_cpu)

    return (; res_cpu, res_from_gpu)

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end