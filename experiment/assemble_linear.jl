function kernabs_assemble_linear!(backend, y, f, V, measure, rdhl)
    quadrature = get_quadrature(measure) # not sure if it's needed here
    domain = get_domain(measure)

    # @code_warntype test_gpu_assemble_kernel!(1, y, f, domain, V, quadrature, rdhl)
    # error("dbg")

    assemble_linear_kernel!(backend, WORKGROUP_SIZE)(
        y,
        f,
        domain,
        V,
        quadrature,
        rdhl;
        ndrange = size(y),
    )
end

@kernel function assemble_linear_kernel!(
    b,
    @Const(f),
    @Const(domain),
    @Const(V),
    @Const(quadrature),
    @Const(rdhl)
)
    # Here  `I` is a global index of a dof
    I = @index(Global)

    assemble_linear_elemental!(I, b, f, domain, V, quadrature, rdhl)
end

"""
Assemble the idof-th element of a linear form
"""
function assemble_linear_elemental!(idof, b, f, domain, V, quadrature, rdhl)
    for i in 1:get_n_elts(rdhl, idof)
        ielt = get_ielt(rdhl, idof, i)
        iloc = get_iloc(rdhl, idof, i)
        eltInfo = _get_index(domain, ielt)

        φ = MyShapeFunction(V, iloc)
        fᵥ = Bcube.materialize(f(φ), eltInfo)
        value = integrate_on_ref_element(fᵥ, eltInfo, quadrature)
        b[idof] += value
    end
end

function run_linear_cell_continuous(backend)
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

    U_cpu = TrialFESpace(
        FunctionSpace(:Lagrange, 1),
        mesh_cpu,
        Dict("xmin" => 5.0, "xmax" => g, "ymin" => h),
    )
    U = adapt(backend, U_cpu)
    test_arg(backend, U)
    println("U on GPU!")

    V = TestFESpace(U)
    test_arg(backend, V)
    println("V on GPU!")

    # Build ReverseDofHandler
    rdhl_cpu = ReverseDofHandler(Ω_cpu, U_cpu) # strictly speaking it should be built with V_cpu
    rdhl = adapt(backend, rdhl_cpu)
    test_arg(backend, rdhl)
    println("rdhl on GPU!")

    # Build FEFunction
    u = FEFunction(U, KernelAbstractions.ones(backend, Float64, get_ndofs(U)))
    test_arg(backend, u)
    println("u on GPU!")

    # Define linear form and assemble
    f(φ) = u * φ
    # f(φ) = PhysicalFunction(x -> 1.0) * φ
    y = KernelAbstractions.zeros(backend, Float64, get_ndofs(U))
    kernabs_assemble_linear!(backend, y, f, V, dΩ, rdhl)
    display(y)

    # Compare with CPU result
    u_cpu = FEFunction(U_cpu, ones(get_ndofs(U)))
    f_cpu(φ) = ∫(u_cpu * φ)Measure(CellDomain(mesh_cpu), 1)
    println("Result on CPU:")
    display(assemble_linear(f_cpu, TestFESpace(U_cpu)))

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end

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
    kernabs_assemble_linear!(backend, y, f, V, dΓ, rdhl)
    display(y)

    # Compare with CPU result
    u_cpu = FEFunction(U_cpu, ones(get_ndofs(U)))
    f_cpu(φ) = ∫(side_n(u_cpu) * jump(φ))Measure(InteriorFaceDomain(mesh_cpu), 1)
    println("Result on CPU:")
    display(assemble_linear(f_cpu, TestFESpace(U_cpu)))

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end