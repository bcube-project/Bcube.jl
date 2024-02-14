"""
    write_vtk(basename::String, it::Int,time::Real, mesh::AbstractMesh{topoDim,spaceDim}, vars::Dict{String,Tuple{V,L}}; append=false) where{topoDim,spaceDim,V,L<:WriteVTK.AbstractFieldData}

Write a set of variables on the mesh nodes or cell centers to a VTK file.

# Example
```julia
mesh = basic_mesh()
u = rand(ncells(mesh))
v = rand(nnodes(mesh))
dict_vars = Dict( "u" => (u, VTKCellData()),  "v" => (v, VTKPointData()) )
write_vtk("output", 0, 0.0, mesh, dict_vars)
```
"""
function write_vtk(
    basename::String,
    it::Int,
    time::Real,
    mesh::AbstractMesh{topoDim, spaceDim},
    vars::Dict{String, Tuple{V, L}};
    append = false,
) where {topoDim, spaceDim, V, L <: WriteVTK.AbstractFieldData}
    pvd = paraview_collection(basename; append = append)

    # Create coordinates arrays
    vtknodes = reshape(
        [coords(n)[idim] for n in get_nodes(mesh) for idim in 1:spaceDim],
        spaceDim,
        nnodes(mesh),
    )

    # Connectivity
    c2n = connectivities_indices(mesh, :c2n)
    # Create cell array
    vtkcells =
        [MeshCell(vtk_entity(cells(mesh)[icell]), c2n[icell]) for icell in 1:ncells(mesh)]

    # Define mesh for vtk
    new_name = _build_fname_with_iterations(basename, it)
    vtkfile = vtk_grid(new_name, vtknodes, vtkcells)

    for (varname, (value, loc)) in vars
        vtkfile[varname, loc] = value
    end

    pvd[float(time)] = vtkfile
    vtk_save(pvd) # also triggers `vtk_save(vtkfile)`
end

"""
VTK writer for a set of discontinuous functions. `vars` is a dictionnary of
variable name => (values, values_location)

where values is an array of numbers.
"""
function write_vtk_discontinuous(
    basename::String,
    it::Int,
    time::Real,
    mesh::AbstractMesh{topoDim, spaceDim},
    vars::Dict{String, Tuple{V, L}},
    degree::Int;
    append = false,
) where {topoDim, spaceDim, V, L <: WriteVTK.AbstractFieldData}
    pvd = paraview_collection(basename; append = append)

    # Connectivity
    c2n = connectivities_indices(mesh, :c2n)

    celltypes = cells(mesh)

    _degree = max(1, degree)
    fs = FunctionSpace(:Lagrange, max(1, degree)) # here, we implicitly impose that the mesh is composed of Lagrange elements only

    # Create coordinates arrays
    # vtknodes = reshape([coords(n)[idim] for i in 1:ncells(mesh) for n in get_nodes(mesh,c2n[i]) for idim in 1:spaceDim],spaceDim,:)
    _coords = [
        mapping(celltypes[i], get_nodes(mesh, c2n[i]), ξ)[idim] for i in 1:ncells(mesh)
        for ξ in _vtk_coords_from_lagrange(shape(celltypes[i]), _degree) for
        idim in 1:spaceDim
    ]
    vtknodes = reshape(_coords, spaceDim, :)

    # Create cell array
    vtkcells = MeshCell[]
    count = 0
    for icell in 1:ncells(mesh)
        _nnode = ndofs(fs, shape(celltypes[icell])) #length(c2n[icell])
        push!(
            vtkcells,
            MeshCell(
                vtk_entity(shape(cells(mesh)[icell]), Val(_degree)),
                collect((count + 1):(count + _nnode)),
            ),
        )
        count += _nnode
    end

    _index_val = [
        _vtk_lagrange_node_index_vtk_to_bcube(shape(celltypes[i]), _degree) for
        i in 1:ncells(mesh)
    ]
    index_val = similar(rawcat(_index_val))
    offset = 0
    for ind in _index_val
        index_val[(offset + 1):(offset + length(ind))] .= (offset .+ ind)
        offset += length(ind)
    end

    # Define mesh for vtk
    #vtk_save(paraview_collection(basename))
    new_name = _build_fname_with_iterations(basename, it)
    vtkfile = vtk_grid(new_name, vtknodes, vtkcells)

    for (varname, (value, loc)) in vars
        if isa(loc, VTKPointData)
            vtkfile[varname, loc] = value[index_val]
        else
            vtkfile[varname, loc] = value
        end
    end

    pvd[float(time)] = vtkfile
    vtk_save(pvd)
end

function write_vtk_bnd_discontinuous(
    basename::String,
    it::Int,
    time::Real,
    domain::BoundaryFaceDomain,
    vars::Dict{String, Tuple{V, L}},
    degree::Int;
    append = false,
) where {V, L <: WriteVTK.AbstractFieldData}
    pvd = paraview_collection(basename; append = append)

    mesh = get_mesh(domain)
    sdim = spacedim(mesh)

    # Connectivities
    c2n = connectivities_indices(mesh, :c2n)
    f2n = connectivities_indices(mesh, :f2n)
    f2c = connectivities_indices(mesh, :f2c)

    # Cell and face types
    celltypes = cells(mesh)

    bndfaces = get_cache(domain)

    fs = FunctionSpace(:Lagrange, max(1, degree)) # here, we implicitly impose that the mesh is composed of Lagrange elements only

    a = map(bndfaces) do iface
        icell = f2c[iface][1]
        sideᵢ = cell_side(celltypes[icell], c2n[icell], f2n[iface])
        localfacedofs = idof_by_face_with_bounds(fs, shape(celltypes[icell]))[sideᵢ]
        ξ = coords(fs, shape(celltypes[icell]))[localfacedofs]
        xdofs = map(_ξ -> mapping(celltypes[icell], get_nodes(mesh, c2n[icell]), _ξ), ξ)
        ftype = entity(face_shapes(shape(celltypes[icell]), sideᵢ), Val(degree))
        ftype, rawcat(xdofs)
    end
    ftypes = getindex.(a, 1)
    vtknodes = reshape(rawcat(getindex.(a, 2)), sdim, :)

    # Create elements array
    vtkcells = MeshCell[]
    count = 0
    for ftype in ftypes
        _nnode = ndofs(fs, shape(ftype)) #length(c2n[icell])
        push!(vtkcells, MeshCell(vtk_entity(ftype), collect((count + 1):(count + _nnode))))
        count += _nnode
    end

    # Define mesh for vtk
    new_name = _build_fname_with_iterations(basename, it)
    vtkfile = vtk_grid(new_name, vtknodes, vtkcells)

    for (varname, (value, loc)) in vars
        vtkfile[varname, loc] = value
    end

    pvd[float(time)] = vtkfile
    vtk_save(pvd)
end

"""
write_vtk(basename::String, mesh::AbstractMesh{topoDim,spaceDim}) where{topoDim,spaceDim}

Write the mesh to a VTK file.

# Example
```julia
write_vtk("output", basic_mesh())
```
"""
function write_vtk(
    basename::String,
    mesh::AbstractMesh{topoDim, spaceDim},
) where {topoDim, spaceDim}
    dict_vars = Dict{String, Tuple{Any, WriteVTK.AbstractFieldData}}()
    write_vtk(basename, 1, 0.0, mesh, dict_vars)
end

"""
    vtk_entity(t::AbstractEntityType)

Convert an `AbstractEntityType` into a `VTKCellType`. To find the correspondance, browse the `WriteVTK`
package AND check the Doxygen (for numbering) : https://vtk.org/doc/nightly/html/classvtkTriQuadraticHexahedron.html
"""
function vtk_entity(t::AbstractEntityType)
    error("Entity type $t doesn't have a VTK correspondance")
end

vtk_entity(::Node_t) = VTKCellTypes.VTK_VERTEX
vtk_entity(::Bar2_t) = VTKCellTypes.VTK_LINE
vtk_entity(::Bar3_t) = VTKCellTypes.VTK_LAGRANGE_CURVE
vtk_entity(::Bar4_t) = VTKCellTypes.VTK_LAGRANGE_CURVE
#vtk_entity(::Bar5_t)    = error("undefined")
vtk_entity(::Tri3_t)  = VTKCellTypes.VTK_TRIANGLE
vtk_entity(::Tri6_t)  = VTKCellTypes.VTK_LAGRANGE_TRIANGLE #VTK_QUADRATIC_TRIANGLE
vtk_entity(::Tri9_t)  = error("undefined")
vtk_entity(::Tri10_t) = VTKCellTypes.VTK_LAGRANGE_TRIANGLE
#vtk_entity(::Tri12_t)   = error("undefined")
vtk_entity(::Quad4_t)   = VTKCellTypes.VTK_QUAD
vtk_entity(::Quad8_t)   = VTKCellTypes.VTK_QUADRATIC_QUAD
vtk_entity(::Quad9_t)   = VTKCellTypes.VTK_BIQUADRATIC_QUAD
vtk_entity(::Quad16_t)  = VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL
vtk_entity(::Tetra4_t)  = VTKCellTypes.VTK_TETRA
vtk_entity(::Tetra10_t) = VTKCellTypes.VTK_QUADRATIC_TETRA
vtk_entity(::Penta6_t)  = VTKCellTypes.VTK_WEDGE
vtk_entity(::Hexa8_t)   = VTKCellTypes.VTK_HEXAHEDRON
#vtk_entity(::Hexa27_t) = VTK_TRIQUADRATIC_HEXAHEDRON # NEED TO CHECK NODE NUMBERING : https://vtk.org/doc/nightly/html/classvtkTriQuadraticHexahedron.html

vtk_entity(::Line, ::Val{Degree}) where {Degree}     = VTKCellTypes.VTK_LAGRANGE_CURVE
vtk_entity(::Square, ::Val{Degree}) where {Degree}   = VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL
vtk_entity(::Triangle, ::Val{Degree}) where {Degree} = VTKCellTypes.VTK_LAGRANGE_TRIANGLE

get_vtk_name(c::VTKCellType) = Val(Symbol(c.vtk_name))
const VTK_LAGRANGE_QUADRILATERAL = get_vtk_name(VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL)
const VTK_LAGRANGE_TRIANGLE = get_vtk_name(VTKCellTypes.VTK_LAGRANGE_TRIANGLE)

"""
Return the node numbering of the node designated by its position in the x and y direction.

See https://www.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/.
"""
function _point_index_from_IJK(t::Val{:VTK_LAGRANGE_QUADRILATERAL}, degree, i, j)
    1 + _point_index_from_IJK_0based(t, degree, i - 1, j - 1)
end

# see : https://github.com/Kitware/VTK/blob/675adbc0feeb3f62730ecacb2af87917af124543/Filters/Sources/vtkCellTypeSource.cxx
#        https://github.com/Kitware/VTK/blob/265ca48a79a36538c95622c237da11133608bbe5/Common/DataModel/vtkLagrangeQuadrilateral.cxx#L558
function _point_index_from_IJK_0based(::Val{:VTK_LAGRANGE_QUADRILATERAL}, degree, i, j)
    # 0-based algo
    ni = degree
    nj = degree
    ibnd = ((i == 0) || (i == ni))
    jbnd = ((j == 0) || (j == nj))
    # How many boundaries do we lie on at once?
    nbnd = (ibnd > 0 ? 1 : 0) + (jbnd > 0 ? 1 : 0)

    if nbnd == 2 # Vertex DOF
        return (i > 0 ? (j > 0 ? 2 : 1) : (j > 0 ? 3 : 0))
    end

    offset = 4
    if nbnd == 1 # Edge DOF
        ibnd == 0 && (return (i - 1) + (j > 0 ? degree - 1 + degree - 1 : 0) + offset)
        jbnd == 0 &&
            (return (j - 1) + (i > 0 ? degree - 1 : 2 * (degree - 1) + degree - 1) + offset)
    end

    offset = offset + 2 * (degree - 1 + degree - 1)
    # Face DOF
    return (offset + (i - 1) + (degree - 1) * (j - 1))
end

function _vtk_lagrange_node_index_vtk_to_bcube(shape::AbstractShape, degree)
    return 1:length(coords(FunctionSpace(Lagrange(), degree), shape))
end

"""
Bcube node numbering -> VTK node numbering (in a cell)
"""
function _vtk_lagrange_node_index_bcube_to_vtk(shape::Union{Square, Cube}, degree)
    n = _get_num_nodes_per_dim(
        QuadratureRule(shape, Quadrature(QuadratureUniform(), Val(degree))),
    )
    IJK = CartesianIndices(ntuple(i -> 1:n[i], length(n)))

    vtk_cell_name = get_vtk_name(vtk_entity(shape, Val(degree)))

    # We loop over all the nodes designated by (i,j,k) and find the corresponding
    # VTK index for each node.
    index = map(vec(IJK)) do ijk
        _point_index_from_IJK(vtk_cell_name, degree, Tuple(ijk)...)
    end
    return index
end

"""
VTK node numbering (in a cell) -> Bcube node numbering
"""
function _vtk_lagrange_node_index_vtk_to_bcube(shape::Union{Square, Cube}, degree)
    return invperm(_vtk_lagrange_node_index_bcube_to_vtk(shape, degree))
end

function _vtk_coords_from_lagrange(shape::AbstractShape, degree)
    return coords(FunctionSpace(Lagrange(), degree), shape)
end

"""
Coordinates of the nodes in the VTK cell, ordered as expected by VTK.
"""
function _vtk_coords_from_lagrange(shape::Union{Square, Cube}, degree)
    fs = FunctionSpace(Lagrange(), degree)
    c = coords(fs, shape)
    index = _vtk_lagrange_node_index_vtk_to_bcube(shape, degree)
    return c[index]
end

"""
    write_vtk_lagrange(
        basename::String,
        vars::Dict{String, F},
        mesh::AbstractMesh,
        U_export::AbstractFESpace,
        it::Integer = -1,
        time::Real = 0.0;
        collection_append = false,
        vtk_kwargs...,
    ) where {F <: AbstractLazy}

Write the provided FEFunction on the mesh with the precision of the Lagrange FESpace provided.

`vars` is a dictionnary of variable name => FEFunction to write.

# Example
```julia
mesh = rectangle_mesh(6, 7; xmin = -1, xmax = 1.0, ymin = -1, ymax = 1.0)
u = FEFunction(TrialFESpace(FunctionSpace(:Lagrange, 4), mesh))
projection_l2!(u, PhysicalFunction(x -> x[1]^2 + x[2]^2), mesh)

vars = Dict("u" => u, "grad_u" => ∇(u))

for degree_export in 1:5
    U_export = TrialFESpace(FunctionSpace(:Lagrange, degree_export), mesh)
    Bcube.write_vtk_lagrange(
        joinpath(@__DIR__, "output"),
        vars,
        mesh,
        U_export,
    )
end
```

# Dev notes
- in order to write an ASCII file, you must pass both `ascii = true` and `append = false`
- `collection_append` is not named `append` to enable passing correct `kwargs` to `vtk_grid`
- remove (once fully validated) : `write_vtk_discontinuous`
"""
function write_vtk_lagrange(
    basename::String,
    vars::Dict{String, F},
    mesh::AbstractMesh,
    U_export::AbstractFESpace,
    it::Integer = -1,
    time::Real = 0.0;
    collection_append = false,
    vtk_kwargs...,
) where {F <: AbstractLazy}
    _vars = collect(values(vars))

    # FE space stuff
    fs_export = get_function_space(U_export)
    @assert get_type(fs_export) <: Lagrange "Only FunctionSpace of type Lagrange are supported for now"
    degree_export = get_degree(fs_export)
    dhl_export = Bcube._get_dhl(U_export)
    nd = ndofs(dhl_export)

    # Find the number of components of each variable by evaluating it on the
    # center of the first cell of the mesh
    cinfo = CellInfo(mesh, 1)
    ξ = center(shape(celltype(cinfo)))
    cpoint = CellPoint(ξ, cinfo, ReferenceDomain())
    ncomps = [length(materialize(var, cinfo)(cpoint)) for var in values(vars)]

    # VTK stuff
    coords_vtk = zeros(spacedim(mesh), nd)
    values_vtk = ntuple(i -> zeros(ncomps[i], nd), length(_vars))
    cells_vtk = MeshCell[]
    sizehint!(cells_vtk, ncells(mesh))

    # Loop over mesh cells
    for cinfo in DomainIterator(CellDomain(mesh))
        # Cell infos
        icell = cellindex(cinfo)
        ctype = celltype(cinfo)
        _shape = shape(ctype)

        # Get Lagrange dofs/nodes coordinates in ref space (Bcube order)
        coords_bcube = coords(fs_export, _shape)

        # Get VTK <-> BCUBE dofs/nodes mapping
        v2b = Bcube._vtk_lagrange_node_index_vtk_to_bcube(_shape, degree_export)

        # Add nodes coordinates to coords_vtk
        for (iloc, ξ) in enumerate(coords_bcube)
            # Global number of the node
            iglob = dof(dhl_export, icell, 1, iloc)

            # Create CellPoint (in reference domain)
            cpoint = CellPoint(ξ, cinfo, ReferenceDomain())

            # Map it to the physical domain and assign to VTK
            coords_vtk[:, iglob] .= get_coord(change_domain(cpoint, PhysicalDomain()))

            # Evaluate all vars on this node
            for (ivar, var) in enumerate(_vars)
                _var = materialize(var, cinfo)
                values_vtk[ivar][:, iglob] .= _var(cpoint)
            end
        end

        # Create the VTK cells with the correct ordering
        iglobs = dof(dhl_export, icell)
        push!(
            cells_vtk,
            MeshCell(vtk_entity(_shape, Val(degree_export)), view(iglobs, v2b)),
        )
    end

    # Write VTK file
    pvd = paraview_collection(basename; append = collection_append)
    new_name = _build_fname_with_iterations(basename, it)
    vtkfile = vtk_grid(new_name, coords_vtk, cells_vtk; vtk_kwargs...)

    for (varname, value) in zip(keys(vars), values_vtk)
        vtkfile[varname, VTKPointData()] = value
    end

    pvd[float(time)] = vtkfile
    vtk_save(pvd)
end

"""
Append the number of iteration (if positive) to the basename
"""
function _build_fname_with_iterations(basename::String, it::Integer)
    it >= 0 ? @sprintf("%s_%08i", basename, it) : basename
end
