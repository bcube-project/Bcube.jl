
function run_linear_cell_continuous_vector(backend)
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
    U_cpu = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh_cpu; size = 2)
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

    # Build FEFunction
    u = FEFunction(U, KernelAbstractions.ones(backend, Float64, get_ndofs(U)))
    test_arg(backend, u)
    println("u on GPU!")

    # Define linear form and assemble
    f(φ) = u ⋅ φ
    y = KernelAbstractions.zeros(backend, Float64, get_ndofs(U))
    BcubeGPU.kernabs_assemble_linear!(backend, y, f, V, dΩ, rdhl)
    display(y)
    res_from_gpu = Array(y)

    # Compare with CPU result
    u_cpu = FEFunction(U_cpu, ones(get_ndofs(U)))
    f_cpu(φ) = ∫(u_cpu ⋅ φ)Measure(CellDomain(mesh_cpu), 1)
    res_cpu = assemble_linear(f_cpu, TestFESpace(U_cpu))
    println("Result on CPU:")
    display(res_cpu)

    return res_cpu == res_from_gpu

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end