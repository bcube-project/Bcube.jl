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
    synchronize(backend)

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
    return DenseRowsSparseCols(offset, values)
end