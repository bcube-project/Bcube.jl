function kernabs_assemble_bilinear(backend, f, U, V, measure, rdhl_V, ind)
    quadrature = get_quadrature(measure) # not sure if it's needed here
    domain = get_domain(measure)

    n = length(ind.values)
    _I = KernelAbstractions.zeros(backend, Int, n)
    _J = KernelAbstractions.zeros(backend, Int, n)
    _V = KernelAbstractions.zeros(backend, Float64, n) # Warning Float64 forced here

    assemble_bilinear_kernel_v2!(backend, WORKGROUP_SIZE)(
        _I,
        _J,
        _V,
        f,
        domain,
        U,
        V,
        quadrature,
        rdhl_V,
        ind;
        ndrange = get_ndofs(V),
    )

    return sparse(_I, _J, _V)
end

@kernel function assemble_bilinear_kernel_v2!(
    _I,
    _J,
    _V,
    @Const(f),
    @Const(domain),
    @Const(U),
    @Const(V),
    @Const(quadrature),
    @Const(rdhl_V),
    @Const(ind),
)
    # Here  `I` is a global index of a dof
    I = @index(Global)

    assemble_bilinear_elemental_v2!(I, _I, _J, _V, f, domain, U, V, quadrature, rdhl_V, ind)
end

"""
In this version, the parallelization is only performed on the rows of the matrix (A[i,:]), not
the elements (A[i,j])
"""
function assemble_bilinear_elemental_v2!(
    idof,
    _I,
    _J,
    _V,
    f,
    domain,
    U,
    V,
    quadrature,
    rdhl_V,
    ind,
)
    dhl_U = _get_dhl(U)

    # j here is an index designating the j-th dof in relation with idof, all elements concerned.
    # So it starts at 1, increases to the number of dofs in the first element surround idof, then
    # increases again with the dofs in the second element etc
    j = 1

    # Loop on elements "surrounding" idof
    for i in 1:get_n_elts(rdhl_V, idof)
        ielt = get_ielt(rdhl_V, idof, i)
        iloc = get_iloc(rdhl_V, idof, i)
        eltInfo = _get_index(domain, ielt)
        φi = MyShapeFunction(V, iloc)

        # Loop on dofs of U in this cell
        # Warning : this is only valid for CellDomain assembly!
        for (jloc, jdof) in enumerate(get_dof(dhl_U, get_element_index(eltInfo))) # Warning icomp=1
            φj = MyShapeFunction(U, jloc)
            fᵤᵥ = Bcube.materialize(f(φj, φi), eltInfo)
            value = integrate_on_ref_element(fᵤᵥ, eltInfo, quadrature)
            k = get_elt(ind, idof, j)
            _I[k] = idof
            _J[k] = jdof
            _V[k] = value
            # @show idof, jdof, iloc, jloc, k, value
            j += 1
        end
    end
end

"""
In this version, the parallelization is performed on the elements (A[i,j]).
This function must not be called for all idof ∈ V, jdof ∈ U, but only for
(idof,jdof) sharing at least on element
"""
function assemble_bilinear_elemental_v1!(
    idof,
    jdof,
    _I,
    _J,
    _V,
    f,
    domain,
    U,
    V,
    quadrature,
    rdhl_U,
    rhdl_V,
)
    offset_V = rdhl_V.offset[idof]
    # Loop on elements "surrounding" idof
    for i in 1:rdhl.nelts[idof]
        ielt = rdhl_V.ielts[offset_V + i]
    end
end

"""
Imagine a COO representation of a sparse matrix by a bilinear assembly on (U,V).
We want to know the position of the couple (idof,jdof) in the scalar indexing (I,J,V).
For convenience reason, we actually return the scalar index corresponding to an entry
(idof, jloc) where 'jloc' in the "j-th" dof in relation with idof (i.e accross all cells
surrounding idof).

Rq : we could build it on the GPU using two consecutive kernels:
one to count the dofs in connection to each dof in each cell, the
second to assign a global number

TODO : rewrite the whole function, it's really ugly
"""
function build_bilinear_storage_ind(U, V, rdhl_V)
    ndofs_by_dof = map(1:get_ndofs(V)) do idof
        sum(ielt -> get_ndofs(_get_dhl(U), ielt), 1:get_n_elts(rdhl_V, idof))
    end
    offset = zeros(Int, get_ndofs(V))
    for idof in 1:get_ndofs(V)
        if idof > 1
            offset[idof] = offset[idof - 1] + ndofs_by_dof[idof - 1]
        end
    end
    values = collect(1:sum(ndofs_by_dof))
    @show ndofs_by_dof
    @show offset
    @show values
    return DenseRowsSparseCols(offset, values)
end

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
    ind_cpu = build_bilinear_storage_ind(U_cpu, V_cpu, rdhl_cpu)
    ind = adapt(backend, ind_cpu)

    # Define bilinear form and assemble
    f(u, v) = u ⋅ v
    A = kernabs_assemble_bilinear(backend, f, U, V, dΩ, rdhl, ind)
    _I, _J, _V = findnz(A)
    A_from_gpu = sparse(Array(_I), Array(_J), Array(_V))
    display(A_from_gpu)

    # Compare with CPU result
    a(u, v) = ∫(f(u, v))Measure(CellDomain(mesh_cpu), 1)
    A_cpu = assemble_bilinear(a, U_cpu, V_cpu)
    println("Result on CPU:")
    display(A_cpu)

    @assert A_cpu == A_from_gpu

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end

function run_bilinear_cell_continuous_vector(backend)
    # Mesh and domains
    mesh_cpu = rectangle_mesh(2, 3)
    # mesh_cpu = one_cell_mesh(:line)
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

    # Build "storage" indirection for bilinear assembly
    ind_cpu = build_bilinear_storage_ind(U_cpu, V_cpu, rdhl_cpu)
    ind = adapt(backend, ind_cpu)

    # Define bilinear form and assemble
    f(u, v) = u ⋅ v
    A_cpu = kernabs_assemble_bilinear(backend, f, U, V, dΩ, rdhl, ind)
    _I, _J, _V = findnz(A_cpu)
    A_from_gpu = sparse(Array(_I), Array(_J), Array(_V))
    display(A_from_gpu)
    writedlm(
        joinpath(@__DIR__, "..", "myout", "bilinear-cell-continuous-vector-GPU.csv"),
        Array(A_from_gpu),
        ",",
    )

    # Compare with CPU result
    a(u, v) = ∫(f(u, v))Measure(CellDomain(mesh_cpu), 1)
    println("Result on CPU:")
    A_cpu = assemble_bilinear(a, U_cpu, V_cpu)
    display(A_cpu)
    writedlm(
        joinpath(@__DIR__, "..", "myout", "bilinear-cell-continuous-vector-CPU.csv"),
        Array(A_cpu),
        ",",
    )

    @assert A_cpu == A_from_gpu

    # CUDA.@device_code_typed interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @device_code_warntype interactive = true @cuda cuda_kernel!(res, g, cells, quadrature)
    # @cuda cuda_kernel!(res, g, cells, quadrature)
end