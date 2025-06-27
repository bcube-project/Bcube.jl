
function run_linear_cell_continuous_multi(backend)
    # Mesh and domains
    mesh_cpu = rectangle_mesh(2, 3)
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
    g(x, t) = 3x[1]
    h(x, t) = 5x[1] + 2

    U1_cpu = TrialFESpace(
        FunctionSpace(:Lagrange, 1),
        mesh_cpu,
        Dict("xmin" => 5.0, "xmax" => g, "ymin" => h),
    )
    U1 = adapt(backend, U1_cpu)
    U2_cpu = TrialFESpace(FunctionSpace(:Lagrange, 2), mesh_cpu)
    U2 = adapt(backend, U2_cpu)
    U = MultiFESpace(U1, U2)
    test_arg(backend, U)
    println("U on GPU!")

    V1_cpu = TestFESpace(U1_cpu)
    V2_cpu = TestFESpace(U2_cpu)
    V1 = TestFESpace(U1)
    V2 = TestFESpace(U2)
    V = MultiFESpace(V1, V2)
    println("V on GPU!")

    # Build ReverseDofHandler
    rdhl1_cpu = ReverseDofHandler(Ω_cpu, V1_cpu)
    rdhl2_cpu = ReverseDofHandler(Ω_cpu, V2_cpu)
    rdhl1 = adapt(backend, rdhl1_cpu)
    rdhl2 = adapt(backend, rdhl2_cpu)
    rdhl = (rdhl1, rdhl2)

    # Build FEFunction
    u1 = FEFunction(U1, KernelAbstractions.ones(backend, Float64, get_ndofs(U1)))
    u2 = FEFunction(U2, KernelAbstractions.ones(backend, Float64, get_ndofs(U2)))

    # Define linear form and assemble
    f((φ1, φ2)) = u1 * φ1 + 2 * u1 * φ2 + 3 * u2 * φ1 + 4 * u2 * φ2
    y = KernelAbstractions.zeros(backend, Float64, get_ndofs(V))
    BcubeGPU.kernabs_assemble_linear!(backend, y, f, V, dΩ, rdhl)
    display(y)
    res_from_gpu = Array(y)

    error("dbg")

    # Compare with CPU result
    u_cpu = FEFunction(U1_cpu, ones(get_ndofs(U1)))
    f_cpu(φ) = ∫(u_cpu * φ)Measure(CellDomain(mesh_cpu), 1)
    res_cpu = assemble_linear(f_cpu, TestFESpace(U1_cpu))
    println("Result on CPU:")
    display(res_cpu)

    return (; res_cpu, res_from_gpu)

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end