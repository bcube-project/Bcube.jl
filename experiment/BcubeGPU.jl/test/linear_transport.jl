function run_linear_transport(backend)
    out_dir = joinpath(@__DIR__, "..", "..", "..", "myout", "linear_transport")
    mkpath(out_dir) #hide

    # First, we define some physical and numerical constant parameters
    degree = 0 # Function-space degree (Taylor(0) = first order Finite Volume)
    c = SA[1.0, 0.0] # Convection velocity (must be a vector)
    nite = 100 # Number of time iteration(s)
    CFL = 1 # 0.1 for degree 1
    nx = 41 # Number of nodes in the x-direction
    ny = 41 # Number of nodes in the y-direction
    lx = 2.0 # Domain width
    ly = 2.0 # Domain height
    Δt = CFL * min(lx / nx, ly / ny) / norm(c) # Time step

    # Then generate the mesh of a rectangle using Gmsh and read it
    tmp_path = joinpath(out_dir, "tmp.msh")
    BcubeGmsh.gen_rectangle_mesh(
        tmp_path,
        :quad;
        nx = nx,
        ny = ny,
        lx = lx,
        ly = ly,
        xc = 0.0,
        yc = 0.0,
        msh_format = 2.2,
    )
    mesh_cpu = read_mesh(Bcube.Gmsh22IoHandler(), tmp_path)
    mesh = adapt(backend, mesh_cpu)
    rm(tmp_path)

    # As seen in the previous tutorial, the definition of trial and test spaces needs a mesh and
    # a function space. Here, we select Taylor space, and build discontinuous FE spaces with it.
    # Then an FEFunction, that will represent our solution, is created.
    fs = FunctionSpace(:Taylor, degree)
    U_cpu = TrialFESpace(fs, mesh_cpu, :discontinuous)
    V_cpu = TestFESpace(U_cpu)
    U = adapt(backend, U_cpu)
    V = TestFESpace(U)
    u = FEFunction(U, KernelAbstractions.zeros(backend, Float64, get_ndofs(U)))

    # We can now init our `VtkHandler`
    basename = joinpath(out_dir, "linear_transport.pvd")
    write_file(
        basename,
        mesh_cpu,
        Dict("u" => FEFunction(U_cpu, Array(get_dof_values(u)))),
        0,
        0.0;
        discontinuous = true,
        collection_append = false,
    )

    # Define measures for cell and interior face integrations
    Ω_cpu = CellDomain(mesh_cpu)
    Γ_cpu = InteriorFaceDomain(mesh_cpu)
    Γ_in_cpu = BoundaryFaceDomain(mesh_cpu, "West")
    Γ_out_cpu = BoundaryFaceDomain(mesh_cpu, ("North", "East", "South"))

    Ω = CellDomain(mesh)
    Γ = InteriorFaceDomain(mesh)
    Γ_in = BoundaryFaceDomain(mesh, "West")
    Γ_out = BoundaryFaceDomain(mesh, ("North", "East", "South"))

    # println("running adapt...")
    # x = adapt(backend, Γ_in)
    # x = Γ_in # KO
    println("adapting...")
    x = adapt(backend, Γ_in_cpu)
    println("testing")
    test_arg(backend, x.mesh)
    println("1")
    test_arg(backend, x.bc)
    println("2")
    @show x.labels
    @show typeof(x.labels)
    test_arg(backend, x.labels)
    println("3")
    test_arg(backend, x.cache)
    println("4")
    test_arg(backend, x)
    println("5")

    rdhl_Ω_cpu = ReverseDofHandler(Ω_cpu, V_cpu)
    rdhl_Γ_cpu = ReverseDofHandler(Γ_cpu, V_cpu)
    rdhl_Γ_in_cpu = ReverseDofHandler(Γ_in_cpu, V_cpu)
    rdhl_Γ_out_cpu = ReverseDofHandler(Γ_out_cpu, V_cpu)

    rdhl_Ω = adapt(backend, rdhl_Ω_cpu)
    rdhl_Γ = adapt(backend, rdhl_Γ_cpu)
    rdhl_Γ_in = adapt(backend, rdhl_Γ_in_cpu)
    rdhl_Γ_out = adapt(backend, rdhl_Γ_out_cpu)

    ind_Ω_cpu = BcubeGPU.build_bilinear_storage_ind(U_cpu, V_cpu, rdhl_Ω_cpu)
    ind_Ω = adapt(backend, ind_Ω_cpu)

    dΩ = Measure(Ω, 2 * degree + 1)
    dΓ = Measure(Γ, 2 * degree + 1)
    dΓ_in = Measure(Γ_in, 2 * degree + 1)
    dΓ_out = Measure(Γ_out, 2 * degree + 1)

    # We will also need the face normals associated to the different face domains.
    # Note that this operation is lazy, `nΓ` is just an abstract representation on
    # face normals of `Γ`.
    nΓ = get_face_normals(Γ)
    nΓ_in = get_face_normals(Γ_in)
    nΓ_out = get_face_normals(Γ_out)

    # Let's move on to the bilinear and linear forms. First, the two easiest ones:
    m_Ω(u, v) = u ⋅ v
    l_Ω(v) = (c * u) ⋅ ∇(v)

    # For the flux term, we first need to define a numerical flux. It is convenient to define it separately
    # in a dedicated function. Here is the definition of simple upwind flux.

    # We then define the "flux" as the composition of the upwind function and the needed entries: namely the
    # solution on the negative side of the face, the solution on the positive face, and the face normal. The
    # orientation negative/positive is arbitrary, the only convention is that the face normals are oriented from
    # the negative side to the positive side.
    l_Γ(v) = flux(u, c, nΓ) * jump(v)

    # l_Γ_in(v, t) = side⁻(bc_in(t)) ⋅ side⁻(nΓ_in) * side⁻(v)
    l_Γ_out(v) = flux_out(u, c, nΓ_out) * side⁻(v)

    # Assemble the (constant) mass matrix. The returned matrix is a sparse matrix. To simplify the
    # tutorial, we will directly compute the inverse mass matrix. But note that way more performant
    # strategies should be employed to solve such a problem (since we don't need the inverse, only the
    # matrix-vector product).
    M = BcubeGPU.kernabs_assemble_bilinear(backend, m_Ω, U, V, dΩ, rdhl_Ω, ind_Ω)

    # Let's also create three vectors to avoid allocating them at each time step
    nd = get_ndofs(V)
    b_vol = KernelAbstractions.zeros(backend, Float64, nd)
    b_fac = similar(b_vol)
    rhs = similar(b_vol)

    # The time loop is trivial : at each time step we compute the linear forms using
    # the `assemble_` methods, we complete the rhs, perform an explicit step and write
    # the solution.
    t = 0.0
    for i in 1:nite

        ## Reset pre-allocated vectors
        b_vol .= 0.0
        b_fac .= 0.0

        ## Compute linear forms
        BcubeGPU.kernabs_assemble_linear!(backend, b_vol, l_Ω, V, dΩ, rdhl_Ω)
        BcubeGPU.kernabs_assemble_linear!(backend, b_fac, l_Γ, V, dΓ, rdhl_Γ)
        test_arg(backend, b_fac)
        test_arg(backend, Base.Fix2(l_Γ_in, t))
        test_arg(backend, V)
        test_arg(backend, dΓ_in)
        test_arg(backend, rdhl_Γ_in)
        BcubeGPU.kernabs_assemble_linear!(
            backend,
            b_fac,
            v -> l_Γ_in(v, t),
            V,
            dΓ_in,
            rdhl_Γ_in,
        )
        BcubeGPU.kernabs_assemble_linear!(backend, b_fac, l_Γ_out, V, dΓ_out, rdhl_Γ_out)

        ## Assemble rhs
        rhs .= Δt .* M \ (b_vol .- b_fac)

        ## Update solution
        u.dofValues .+= rhs

        ## Update time
        t += Δt

        ## Write to file
        write_file(
            basename,
            mesh_cpu,
            Dict("u" => FEFunction(U_cpu, Array(get_dof_values(u)))),
            i,
            t;
            discontinuous = true,
            collection_append = true,
        )

        return true
    end
end

function upwind(ui, uj, c, nij)
    cij = c ⋅ nij
    return cij > zero(cij) ? cij * ui : cij * uj
end

flux(u, c, n) = upwind ∘ (side⁻(u), side⁺(u), c, side⁻(n))
flux_out(u, c, n) = upwind ∘ (side⁻(u), 0.0, c, side⁻(n))

bc_in(t) = PhysicalFunction(x -> c .* cos(3 * x[2]) * sin(4 * t)) # flux

l_Γ_in(v, t) = side⁻(bc_in(t)) ⋅ side⁻(nΓ_in) * side⁻(v)