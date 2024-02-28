"""
    var_on_vertices(f::AbstractFEFunction, mesh::Mesh)

Interpolate solution on mesh vertices.

The result is a (nnodes, ncomps) matrix if ncomps > 1, or a (nnodes) vector otherwise.

WARNING : for now, the contribution to one vertice is the arithmetic mean of
all the data obtained from the neighbor cells of this node. We could use
the surface area (among other possible choices).
"""
function var_on_vertices(f::AbstractLazy, mesh::Mesh)
    N, T = _codim_and_type(f, mesh)
    @assert length(N) <= 1 "N = $(length(N)) > 1 not supported yet"
    values = zeros(T, nnodes(mesh), N[1])
    _var_on_vertices!(values, f, mesh)
    return N[1] == 1 ? vec(values) : values
end

function _var_on_vertices!(values, f::AbstractLazy, mesh::Mesh)
    # Alias
    c2n = connectivities_indices(mesh, :c2n)
    cellTypes = cells(mesh)

    # Number of contributions per nodes
    ncontributions = zeros(Int, nnodes(mesh))

    # Loop over cells
    for icell in 1:ncells(mesh)
        # Cell information
        ctype = cellTypes[icell]
        cshape = shape(ctype)
        _c2n = c2n[icell]
        cnodes = get_nodes(mesh, _c2n)
        cInfo = CellInfo(icell, ctype, cnodes)

        # Materialize function on CellInfo
        _f = materialize(f, cInfo)

        # Loop over the cell nodes (in ref element)
        for (inode, ξ) in zip(_c2n, coords(cshape))
            cPoint = CellPoint(ξ, cInfo, ReferenceDomain())
            ncontributions[inode] += 1
            values[inode, :] .+= _f(cPoint)
        end
    end

    # Arithmetic mean
    for ic in size(values, 2)
        values[:, ic] .= values[:, ic] ./ ncontributions
    end
end

"""
Evaluate codimension of `f` and returned type. The returned codimension
is always a Tuple of codimension(s), even for a scalar.
"""
function _codim_and_type(f::AbstractLazy, mesh::Mesh)
    # Get info about first cell of the mesh
    cInfo = CellInfo(mesh, 1)

    # Materialize FE function on CellInfo
    _f = materialize(f, cInfo)

    # Evaluate the function on the center of the cell
    cPoint = CellPoint(center(shape(celltype(cInfo))), cInfo, ReferenceDomain())
    value = _f(cPoint)

    # Codim and type
    N = value isa Number ? (1,) : size(value)
    T = eltype(value)
    return N, T
end

"""
    var_on_centers(f::AbstractLazy, mesh::AbstractMesh)

Interpolate solution on mesh centers.

The result is a (ncells, ncomps) matrix if ncomps > 1, or a (ncells) vector otherwise.
"""
function var_on_centers(f::AbstractLazy, mesh::AbstractMesh)
    N, T = _codim_and_type(f, mesh)
    @assert length(N) <= 1 "N = $(length(N)) > 1 not supported yet"
    values = zeros(T, ncells(mesh), N[1])
    _var_on_centers!(values, f, mesh)
    return N[1] == 1 ? vec(values) : values
end

function _var_on_centers!(values, f::AbstractLazy, mesh::AbstractMesh)
    # Alias
    c2n = connectivities_indices(mesh, :c2n)
    celltypes = cells(mesh)

    # Loop over cells
    for icell in 1:ncells(mesh)
        ctype = celltypes[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        cInfo = CellInfo(icell, ctype, cnodes)
        _f = materialize(f, cInfo)

        ξc = center(shape(celltypes[icell]))
        cPoint = CellPoint(ξc, cInfo, ReferenceDomain())
        values[icell, :] .= _f(cPoint)
    end
end

"""
    var_on_nodes_discontinuous(
        f::AbstractFEFunction,
        mesh::AbstractMesh,
        degree::Integer = max(1, get_degree(get_function_space(get_fespace(f)))),
    )

Returns an array containing the values of `f` interpolated to new DoFs.
The DoFs correspond to those of a discontinuous cell variable with a `:Lagrange` function space of selected `degree`.
"""
function var_on_nodes_discontinuous(
    f::AbstractFEFunction,
    mesh::AbstractMesh,
    degree::Integer = max(1, get_degree(get_function_space(get_fespace(f)))),
)
    @assert degree ≥ 1 "degree must be ≥ 1"
    fs = FunctionSpace(:Lagrange, degree) # here, we suppose that the mesh is composed of Lagrange elements only
    _var_on_nodes_discontinuous(f, mesh, fs)
end

"""
Apply the FEFunction on the nodes of the mesh using the `FunctionSpace` representation
for the cells.
"""
function _var_on_nodes_discontinuous(
    f::AbstractFEFunction,
    mesh::AbstractMesh,
    fs::FunctionSpace,
)
    celltypes = cells(mesh)
    c2n = connectivities_indices(mesh, :c2n)

    # Loop on mesh cells
    values = map(1:ncells(mesh)) do icell
        ctype = celltypes[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        cInfo = CellInfo(icell, ctype, cnodes)
        _f = materialize(f, cInfo)

        return map(coords(fs, shape(ctype))) do ξ
            cPoint = CellPoint(ξ, cInfo, ReferenceDomain())
            materialize(_f, cPoint)
        end
    end
    return rawcat(values)
end

"""
    var_on_bnd_nodes_discontinuous(f::AbstractFEFunction, fdomain::BoundaryFaceDomain, degree::Integer=max(1, get_degree(get_function_space(get_fespace(f)))))

Returns an array containing the values of `f` interpolated to new DoFs on `fdomain`.
The DoFs locations on `fdomain` correspond to those of a discontinuous `FESpace`
with a `:Lagrange` function space of selected `degree`.
"""
function var_on_bnd_nodes_discontinuous(
    f::AbstractFEFunction,
    fdomain::BoundaryFaceDomain,
    degree::Integer = max(1, get_degree(get_function_space(get_fespace(f)))),
)
    @assert degree ≥ 1 "degree must be ≥ 1"
    fs = FunctionSpace(:Lagrange, degree) # here, we suppose that the mesh is composed of Lagrange elements only
    _var_on_bnd_nodes_discontinuous(f, fdomain, fs)
end

"""
Apply the FEFunction on the nodes of the `fdomain` using the `FunctionSpace` representation
for the cells.
"""
function _var_on_bnd_nodes_discontinuous(
    f::AbstractFEFunction,
    fdomain::BoundaryFaceDomain,
    fs::FunctionSpace,
)
    mesh = get_mesh(fdomain)
    celltypes = cells(mesh)
    c2n = connectivities_indices(mesh, :c2n)
    f2n = connectivities_indices(mesh, :f2n)
    f2c = connectivities_indices(mesh, :f2c)

    bndfaces = get_cache(fdomain)

    values = map(bndfaces) do iface
        icell = f2c[iface][1]
        ctype = celltypes[icell]
        _c2n = c2n[icell]
        cnodes = get_nodes(mesh, _c2n)
        cinfo = CellInfo(icell, ctype, cnodes, _c2n)
        _f = materialize(f, cinfo)

        side = cell_side(ctype, _c2n, f2n[iface])
        localfacedofs = idof_by_face_with_bounds(fs, shape(ctype))[side]
        ξ_on_face = coords(fs, shape(ctype))[localfacedofs]

        return map(ξ_on_face) do ξ
            cPoint = CellPoint(ξ, cinfo, ReferenceDomain())
            materialize(_f, cPoint)
        end
    end
    return rawcat(values)
end
