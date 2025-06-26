# In this file I place stuff that should be deleted when dev is done

function lagrange_dof_to_coords(mesh, feSpace)
    # Alias
    dhl = Bcube._get_dhl(feSpace)

    dof2coords = zeros(get_ndofs(feSpace), Bcube.spacedim(mesh))

    for cinfo in DomainIterator(CellDomain(mesh))
        # Cell infos
        icell = get_element_index(cinfo)
        ctype = get_element_type(cinfo)

        # Get Lagrange dofs/nodes coordinates in ref space
        coords = get_coords(get_function_space(feSpace), shape(ctype))

        # Loop over these coords
        for (iloc, ξ) in enumerate(coords)
            cpoint = CellPoint(ξ, cinfo, ReferenceDomain())
            cpoint = Bcube.change_domain(cpoint, PhysicalDomain())

            iglob = get_dof(dhl, icell, 1, iloc)
            dof2coords[iglob, :] .= get_coords(cpoint)
        end
    end
    return dof2coords
end

@kernel function test_arg_kernel(x, @Const(arg))
    I = @index(Global)
    x[I] += 1
end

function test_arg(backend, arg)
    x = KernelAbstractions.zeros(backend, Float32, 10)
    test_arg_kernel(backend, WORKGROUP_SIZE)(x, arg; ndrange = size(x))
end