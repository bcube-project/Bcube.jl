"""
The `DofHandler` handles the degree of freedom numbering. To each degree of freedom
is associated a unique integer.

# Constructor
`DofHandler(mesh::Mesh, fSpace::AbstractFunctionSpace, ncomponents::Int, isContinuous::Bool; periodicity = nothing, geom_factor = 1.)`

For continuous spaces with periodicity conditions on the mesh, use the keyword argument `periodicity`
by giving it a set of [`BoundaryFaceDomain`](@ref) built with a `PeriodicBCType`.

# Notes
For continuous FESpaces, the `DofHandler` handles the numbering of common dofs between several
cells (dofs on vertices, on edges, on faces...). In 1D and 2D, this research is performed topologically.
In 3D, a geometrical research is performed for function spaces with degree greater than 2.

!!! tip
    See [check_numbering](@ref) to perform a check that all dofs shared by several entities have an unique
    identifier.
"""
struct DofHandler{A, B}
    # N : number of components

    # Naturally, `iglob` would be a (ndofs, ncell) array. Since
    # the first dimension, the number of dofs in a cell, depends on the cell
    # (ex tri vs quad), a flattened array is used. So here `iglob` is Vector
    # whose size is the total number of dofs of the problem.
    # Warning : for a complete discontinous discretization, `iglob` is simply
    # `iglob = 1:ndofs_tot`,
    # but if a continous variable is present, `iglob` reflects that two dofs
    # of different cells can share the same global index.
    iglob::A

    # Array of size (ncell, ncomps) indicating the positions in `iglob` of the
    # global dof indices for a given cell and a given component (=1 for scalar).
    # So `offset[icell,icomp] + idof` is the position, in `iglob` of the `idof`
    # local dof for the component `icomp` in cell `icell`
    offset::B

    # Number of dofs in each cell for each component. This can be computed from `offset`
    # but easily. It's faster and easier to store the information in this (ncells, ncomps)
    ndofs::B

    # Total number of unique DoFs
    ndofs_tot::Int
end

function DofHandler(
    mesh::Mesh,
    fSpace::AbstractFunctionSpace,
    ncomponents::Int,
    isContinuous::Bool;
    periodicity = nothing,
    geom_factor = 1.0,
)
    # Get cell types
    celltypes = cells(mesh)

    # Allocate
    # `offset` indicates, for each cell, the "position" of the dofs of the cell in the `iglob` vector
    #  `_ndofs` indicates the number of dofs of each cell
    offset = zeros(Int, ncells(mesh), ncomponents)
    _ndofs = zeros(Int, ncells(mesh), ncomponents)

    # First we assume a "discontinuous" type
    ndofs_tot = sum(cell -> ncomponents * get_ndofs(fSpace, shape(cell)), cells(mesh))
    iglob = collect(1:ndofs_tot)
    curr = 0 # Current offset value. Init to obtain '0' as the first element of `offset`
    next = 0 # Next offset value
    for icell in 1:ncells(mesh)
        for icomp in 1:ncomponents
            # Fill offset
            offset[icell, icomp] = curr + next
            curr = offset[icell, icomp]
            next = get_ndofs(fSpace, shape(celltypes[icell]))

            # Fill ndofs
            _ndofs[icell, icomp] = get_ndofs(fSpace, shape(celltypes[icell]))
        end
    end

    # At this point, we have everything we need for the DofHandler of a discontinuous variable.
    # The lines below handle the case of a continuous variable,
    if isContinuous

        # We get some connectivites
        c2n = connectivities_indices(mesh, :c2n) # cell -> node

        # Create dictionnaries (explanations to be completed)
        # Dict([kvar, Set(inodes)] => [icell, Set(idofs_g)])
        # dict : key = set of global indices of nodes of a face, values = (cell index, global indices of dofs)
        dict_n = Dict{Tuple{Int, Int}, Tuple{Int, Vector{Int}}}()
        dict_e = Dict{Tuple{Int, Set{Int}}, Tuple{Int, Vector{Int}}}()
        dict_f = Dict{Tuple{Int, Set{Int}}, Tuple{Int, Vector{Int}}}()

        # Below, a '_l' suffix means "local" by opposition with the '_g' suffix meaning "global"
        # Loop on mesh cells
        for icell in 1:ncells(mesh)
            # Global indices of the cell's nodes
            inodes_g = c2n[icell]

            # Cell type and shape
            ctype = celltypes[icell]
            cshape = shape(ctype)

            # Cell edges, defined by tuples of vertex absolute indices
            # @ghislainb the second line should be improved, I just want to map the "local indices"
            # tuple of tuple ((1,2), (3,4)) into global indices array of arrays [[23,109],[948, 653]]
            # (arrays instead of tuples because your function "oriented_cell_side" need arrays)
            _e2n = edges2nodes(ctype)
            e2n_g = [[inodes_g[i] for i in edge] for edge in _e2n]

            # Cell faces, defined by tuples of vertex absolute indices
            _f2n = faces2nodes(ctype)
            f2n_g = [[inodes_g[i] for i in face] for face in _f2n]

            # Loop over the variables
            for icomp in 1:ncomponents
                # Remark : we need to distinguish vertices, edges, faces because two cells
                # can share dofs with an edge without having a face in common.

                if topodim(mesh) ≤ 2

                    #--- Deal with dofs on vertices
                    _deal_with_dofs_on_vertices!(
                        dict_n,
                        iglob,
                        offset,
                        icell,
                        inodes_g,
                        cshape,
                        icomp,
                        fSpace,
                    )

                    #--- Deal with dofs on edges
                    if topodim(mesh) > 1
                        _deal_with_dofs_on_edges!(
                            dict_e,
                            iglob,
                            offset,
                            c2n,
                            celltypes,
                            icell,
                            e2n_g,
                            icomp,
                            fSpace,
                        )
                    end
                else # topodim ≥ 3
                    # If degree ≤ 2, we perform everything topologically
                    if get_degree(fSpace) ≤ 2
                        #--- Deal with dofs on vertices
                        _deal_with_dofs_on_vertices!(
                            dict_n,
                            iglob,
                            offset,
                            icell,
                            inodes_g,
                            cshape,
                            icomp,
                            fSpace,
                        )

                        #--- Deal with dofs on edges
                        _deal_with_dofs_on_edges!(
                            dict_e,
                            iglob,
                            offset,
                            c2n,
                            celltypes,
                            icell,
                            e2n_g,
                            icomp,
                            fSpace,
                        )

                        #--- Deal with dofs on faces
                        _deal_with_dofs_on_faces_topological!(
                            dict_f,
                            iglob,
                            offset,
                            c2n,
                            celltypes,
                            icell,
                            f2n_g,
                            fSpace,
                            icomp,
                        )
                    else
                        # If degree ≥ 3, we perform everything geometrically
                        _deal_with_dofs_on_faces_geometrical!(;
                            dict = dict_f,
                            iglob,
                            offset,
                            mesh_nodes = get_nodes(mesh),
                            c2n,
                            celltypes,
                            icell,
                            f2n_g,
                            fs = fSpace,
                            icomp,
                            with_bounds = true,
                            geom_factor,
                        )
                    end # if/else degree
                end # if/else topodim
            end # loop on icomp
        end # loop on cells
    end # if isContinuous

    # Apply periodicity (if any and if continuous)
    if !isnothing(periodicity)
        isContinuous &&
            apply_periodicity!(iglob, offset, fSpace, ncomponents, periodicity; geom_factor)
    end

    # Create a cell number remapping to ensure a dense numbering
    densify!(iglob)
    ndofs_tot = length(unique(iglob))
    return DofHandler{typeof(iglob), typeof(offset)}(iglob, offset, _ndofs, ndofs_tot)
end

@inline get_offset(dhl::DofHandler) = dhl.offset
@inline get_offset(dhl::DofHandler, icell::Int, icomp::Int) = dhl.offset[icell, icomp]
@inline get_iglob(dhl::DofHandler, i) = dhl.iglob[i]
@inline get_iglob(dhl::DofHandler) = dhl.iglob

"""
    _deal_with_dofs_on_vertices!(
        dict::Dict{Tuple{Int, Int}, Tuple{Int, Vector{Int}}},
        iglob,
        offset,
        icell::Int,
        inodes_g,
        s::AbstractShape,
        icomp::Int,
        fs::AbstractFunctionSpace,
    )

Function dealing with dofs shared by different cell through a vertex connection.


# Arguments
- `dict` may be modified by this routine
- `iglob` may be modified by this routine
- `offset` may be modified by this routine
- `icell` : cell index
- `inodes_g` : global indices of nodes of `icell`
- `s` : shape of `icell`-th cell
- `icomp` : component
- `fs` : FunctionSpace

# Dev notes
remove kvar
"""
function _deal_with_dofs_on_vertices!(
    dict::Dict{Tuple{Int, Int}, Tuple{Int, Vector{Int}}},
    iglob,
    offset,
    icell::Int,
    inodes_g,
    s::AbstractShape,
    icomp::Int,
    fs::AbstractFunctionSpace,
)
    # Local indices of the dofs on each vertex of the shape
    idofs_array_l = idof_by_vertex(fs, s)

    # Loop over shape vertices
    # ivertex is an Int
    # idofs_l is an Array of Int (local indices of dofs of ith node)
    for (inode_g, idofs_l) in zip(inodes_g, idofs_array_l)

        # Skip the vertex if no dof is lying on it
        length(idofs_l) == 0 && continue

        key = (icomp, inode_g)

        # If the dict already contains the vertex :
        # - we get the neighbour cell index
        # - we copy all the global indices of dofs of `jcell` in the corresponding
        #   global indices of `icell` dofs (`jcell` is useless here actually...)
        if haskey(dict, key)
            jcell, jdofs_g = dict[key]
            for d in eachindex(jdofs_g)
                iglob[offset[icell, icomp] + idofs_l[d]] = jdofs_g[d]
            end

        else
            # If the dict doesn't contain this vertex, we add the global indices
            # of `icell`
            idofs_g = iglob[offset[icell, icomp] .+ idofs_l]
            dict[key] = (icell, idofs_g)
        end
    end
end

"""
    _deal_with_dofs_on_edges!(
        dict::Dict{Tuple{Int, Set{Int}}, Tuple{Int, Vector{Int}}},
        iglob,
        offset,
        c2n,
        celltypes,
        icell::Int,
        e2n_g,
        s::AbstractShape,
        icomp::Int,
        fs::AbstractFunctionSpace,
    )

Function dealing with dofs shared by different cell through an edge connection (excluding bord vertices).

# Arguments
- `dict` may be modified by this routine
- `iglob` may be modified by this routine
- `offset` may be modified by this routine
- `c2n` : global indices of nodes of `icell`
- `celltypes` : mesh cell types
- `icell` : cell index
- `e2n_g` : edge to nodes connectivity for this cell
- `icomp` : component index
- `fs` : FunctionSpace

# Dev notes
remove icomp
"""
function _deal_with_dofs_on_edges!(
    dict::Dict{Tuple{Int, Set{Int}}, Tuple{Int, Vector{Int}}},
    iglob,
    offset,
    c2n,
    celltypes,
    icell::Int,
    e2n_g,
    icomp::Int,
    fs::AbstractFunctionSpace,
)
    ict = celltypes[icell]
    is = shape(ict)

    # Local indices of the dofs on each edges of the shape
    idofs_array_l = idof_by_edge(fs, is)

    # Loop over the cell edges
    # inodes_g is a Tuple of Int (global indices of nodes defining the edge)
    # idofs_l is an Array of Int (local indices of dofs of edge 'i')
    for (inodes_g, idofs_l) in zip(e2n_g, idofs_array_l)

        # Skip the face if no dof is lying on it
        length(idofs_l) == 0 && continue

        key = (icomp, Set(inodes_g))

        # If the dict already contains the edge :
        # - we get the neighbour cell index
        # - we find the local index of the shared edge in jcell
        # - we copy all the global indices of dofs of `jcell` in the corresponding
        #   global indices of `icell` dofs
        if haskey(dict, key)
            jcell, jdofs_g = dict[key]

            # Retrieve local index of the edge in jcell
            jside = oriented_cell_side(celltypes[jcell], c2n[jcell], inodes_g)

            # Reverse dofs array if jside is negative
            # Rq: on edges in 2D, the dofs are always numbered incrementally from on extremity
            # of the edge to the other. In other words, the "middle dof" (if any), is
            # always in the middle of the edge-dof numbering.
            jdofs_reordered_g = (jside > 0) ? jdofs_g : reverse(jdofs_g)

            # Copy global indices
            for d in 1:length(jdofs_g)
                iglob[offset[icell, icomp] + idofs_l[d]] = jdofs_reordered_g[d]
            end

        else
            # If the dict doesn't contain this edge, we add the global indices
            # of `icell`
            idofs_g = iglob[offset[icell, icomp] .+ idofs_l]
            dict[key] = (icell, idofs_g)
        end
    end
end

"""
    _deal_with_dofs_on_faces_topological!(
        dict,
        iglob,
        offset,
        c2n,
        celltypes,
        icell::Int,
        f2n_g::Vector{Vector{Int}},
        fs::AbstractFunctionSpace,
        icomp::Int,
    )

Topological identification of dofs lying on faces of cell `icell`.

# Arguments
- `dict` may be modified by this routine
- `iglob` may be modified by this routine
- `offset` may be modified by this routine
- `c2n` : global indices of nodes of `icell`
- `celltypes` : mesh cell types
- `icell` : cell index
- `f2n_g`` : local face index -> global nodes indices
- `fs` : FunctionSpace
- `icomp` : component index

# Dev notes
remove icomp
"""
function _deal_with_dofs_on_faces_topological!(
    dict,
    iglob,
    offset,
    c2n,
    celltypes,
    icell::Int,
    f2n_g::Vector{Vector{Int}},
    fs::AbstractFunctionSpace,
    icomp::Int,
)
    ict = celltypes[icell]
    is = shape(ict)

    # Local indices of the dofs on each face of the shape, excluding the boundary (nodes and/or edges)
    idofs_array_l = idof_by_face(fs, is) # This is a Tuple of Vector{Int}

    # Loop over cell faces
    # iface_nodes_g is a Tuple of Int (global indices of nodes defining the face)
    # idofs_l is an Array of Int (local indices of dofs of ith face)
    for (iface_nodes_g, idofs_l) in zip(f2n_g, idofs_array_l)
        ne = nedges(is)

        # Skip the face if no dof is lying on it
        length(idofs_l) == 0 && continue

        # Create a Set from the global indices of the face nodes to "tag" the face.
        key = (icomp, Set(iface_nodes_g))

        # If the dict already contains the face :
        # - we get the neighbour cell index
        # - we find the local index of the shared face in jcell
        # - we find the permutation between the two faces
        # - we copy all the global indices of dofs of `jcell` in the corresponding
        #   global indices of `icell` dofs
        if haskey(dict, key)
            jcell, jdofs_g = dict[key]

            # Cell nodes and type
            jcell_nodes_g = c2n[jcell]
            jct = celltypes[jcell]

            # Retrieve local index of the face in jcell
            jside = oriented_cell_side(jct, jcell_nodes_g, iface_nodes_g)
            jface_l = abs(jside) # local index of the face of `jcell` corresponding to `iface`

            # Global indices of the face nodes and mapping between `iface` and `jface`
            jface_nodes_g = [jcell_nodes_g[j] for j in faces2nodes(jct, jface_l)]
            i2j = indexin(iface_nodes_g, jface_nodes_g) # jface_nodes_g[i2j] == iface_nodes_g

            # Number of dofs "by edge" (= "by node") (these dofs are not on a edge, we are just looking
            # for a multiple of the number of edges).
            # Note the use of `÷` because if there is a center dof, we want to exclude it
            nd_by_edge = length(jdofs_g) ÷ ne

            # We want to loop inside `jdofs_g`, but starting with the dofs "corresponding" to the first
            # node of face i. If the faces starts with the same node, offset is 0. If "iface-node-1"
            # corresponds to "jface-node-3", we want to skip the 3*nd_by_edge first dofs.
            i_offset = nd_by_edge * (i2j[1] - 1)

            # Reorder dofs
            # `jdofs_reordered_g` is similar to jdofs_g, but reordered in the same way as "idofs_g"
            jdofs_reordered_g = Int[] # we know the final size, but it is easier to init it like this
            sizehint!(jdofs_reordered_g, length(jdofs_g))
            if (nd_by_edge > 0) # need this check (for instance only a center dof) otherwise error is raised with iterator
                iterator = Iterators.cycle(jdofs_g[1:(nd_by_edge * ne)]) # this removes, eventually, any "center dof"
                (jside < 0) && (iterator = Iterators.reverse(iterator))
                iterator = Iterators.rest(iterator, i_offset)
                for (j, jdof_g) in enumerate(iterator)
                    push!(jdofs_reordered_g, jdof_g)

                    (j == length(jdofs_reordered_g)) && break
                end
            end

            # Add any remaining center dofs (skipped if not needed)
            for j in (length(jdofs_reordered_g) + 1):length(jdofs_g)
                push!(jdofs_reordered_g, jdofs_g[j])
            end

            # Copy global indices
            for d in eachindex(jdofs_reordered_g)
                iglob[offset[icell, icomp] + idofs_l[d]] = jdofs_reordered_g[d]
            end

        else
            # If the dict doesn't contain this face, we add the global indices
            # of `icell`
            idofs_g = iglob[offset[icell, icomp] .+ idofs_l]
            dict[key] = (icell, idofs_g)
        end
    end
end

"""
    _deal_with_dofs_on_faces_geometrical!(;
        dict,
        iglob,
        offset,
        mesh_nodes,
        c2n,
        celltypes,
        icell::Int,
        f2n_g::Vector{Vector{Int}},
        fs::AbstractFunctionSpace,
        icomp::Int,
        with_bounds::Bool,
    )

Geometrical identification of dofs lying on faces of cell `icell`.

# Arguments
- `dict` may be modified by this routine
- `iglob` may be modified by this routine
- `offset` may be modified by this routine
- `c2n` : global indices of nodes of `icell`
- `celltypes` : mesh cell types
- `icell` : cell index
- `f2n_g`` : local face index -> global nodes indices
- `fs` : FunctionSpace
- `icomp` : component index
- `with_bounds` : indicates if the identification should concerns only interior dofs (`with_bounds = false`) or all face dofs
- `geom_factor` : a scaling factor to check that dofs distance is below the given tolerance

# Dev notes
* in the following, we loop over faces of cell `i`, so everything related to this
face is named with `i`: `iface`, `idofs_l` etc. This face may be in connection to a cell `j`.
So everything related to `j` designates this neighbor cell or the face (same as before) but
seen from cell `j`.
* remove icomp
"""
function _deal_with_dofs_on_faces_geometrical!(;
    dict,
    iglob,
    offset,
    mesh_nodes,
    c2n,
    celltypes,
    icell::Int,
    f2n_g::Vector{Vector{Int}},
    fs::AbstractFunctionSpace,
    icomp::Int,
    with_bounds::Bool,
    geom_factor,
)
    # Alias
    icell_node_idx_g = c2n[icell]
    icell_nodes = mesh_nodes[icell_node_idx_g]
    ict = celltypes[icell]
    is = shape(ict)

    # Local indices of the dofs on each face of the shape, excluding the boundary (nodes and/or edges)
    # `idofs_by_face` is a Tuple of Vector{Int}
    idofs_by_face = with_bounds ? idof_by_face_with_bounds(fs, is) : idof_by_face(fs, is)

    # Loop over cell faces
    # iface_nodes_g is a Tuple of Int (global indices of nodes defining the face)
    # idofs_l is an Array of Int (local indices of dofs of ith face)
    for (iface_nodes_g, idofs_l) in zip(f2n_g, idofs_by_face)
        # Skip the face if no dof is lying on it
        length(idofs_l) == 0 && continue

        # Create a Set from the global indices of the face nodes to "tag" the face.
        key = (icomp, Set(iface_nodes_g))

        # If the dict already contains the face :
        # - we get the neighbour cell index
        # - we find the local index of the shared face in jcell
        # - we find the permutation between the two faces
        # - we copy all the global indices of dofs of `jcell` in the corresponding
        #   global indices of `icell` dofs
        if haskey(dict, key)
            jcell, jdofs_g = dict[key]

            # Cell nodes and type
            jcell_node_idx_g = c2n[jcell]
            jcell_nodes = mesh_nodes[jcell_node_idx_g]
            jct = celltypes[jcell]
            js = shape(jct)

            # Retrieve local index of the face in jcell
            jside = oriented_cell_side(jct, jcell_node_idx_g, iface_nodes_g)
            jface_l = abs(jside) # local index of the face of `jcell` corresponding to `iface`

            # Local indices of the dofs on jface (in cell j)
            jdofs_by_face =
                with_bounds ? idof_by_face_with_bounds(fs, js) : idof_by_face(fs, js)
            jdofs_l = jdofs_by_face[jface_l]

            # Identify the pairs. See `identify_face_dofs_from_coords` for the meaning of `i2j`
            i2j = identify_face_dofs_from_coords(
                fs,
                ict,
                icell_nodes,
                idofs_l,
                jct,
                jcell_nodes,
                jdofs_l;
                geom_factor,
            )

            # Copy global indices
            for (iface_dof_l, jface_dof_l) in enumerate(i2j)
                idof_l = idofs_l[iface_dof_l]
                jdof_g = jdofs_g[jface_dof_l] # `jdofs_g` are global dof indices of the face
                iglob[offset[icell, icomp] + idof_l] = jdof_g
            end

        else
            # If the dict doesn't contain this face, we add the global indices
            # of `icell`
            idofs_g = iglob[offset[icell, icomp] .+ idofs_l]
            dict[key] = (icell, idofs_g)
        end
    end
end

"""
    identify_face_dofs_from_coords(
        fSpace::AbstractFunctionSpace,
        ctype_i,
        cnodes_i,
        face_dofs_i,
        ctype_j,
        cnodes_j,
        face_dofs_j;
        transformation = identity,
        geom_factor = 1.0,
    )

Identify the correspondence between degrees of freedom (dofs) located on a face
shared by two neighboring elements by comparing their physical coordinates.

# Arguments
- `fSpace::AbstractFunctionSpace`: function space (used to obtain `get_coords`).
- `ctype_i`, `ctype_j`: element types for elements `i` and `j`.
- `cnodes_i`, `cnodes_j`: arrays of physical node coordinates for elements `i` and `j`.
- `face_dofs_i`, `face_dofs_j`: local indices of the face dofs in each element.
- `geom_factor` is a scaling coefficient to check the maximum admissible distance between two identified dof (leave 1 by default)

# Returns
- `Vector{Int}`: the output vector `i2j` indicates that for
the `ith` dof of the face (ie `face_dofs_i[i]` of  cell i), the corresponding dof
of face `j` is `i2j[i]` (ie `face_dofs_j[i2j]` of cell j).
"""

function identify_face_dofs_from_coords(
    fSpace::AbstractFunctionSpace,
    ctype_i,
    cnodes_i,
    face_dofs_i,
    ctype_j,
    cnodes_j,
    face_dofs_j;
    geom_factor = 1.0,
)
    shape_i = shape(ctype_i)
    shape_j = shape(ctype_j)

    # Get ref coordinates of dofs of each face
    iface_ξ = get_coords(fSpace, shape_i)[face_dofs_i]
    jface_ξ = get_coords(fSpace, shape_j)[face_dofs_j]

    # Map into physical space
    iface_x = [mapping(ctype_i, cnodes_i, ξ) for ξ in iface_ξ]
    jface_x = [mapping(ctype_j, cnodes_j, ξ) for ξ in jface_ξ]

    # Compute point wise "distances"
    # TODO: there might be a way to build and SMatrix because the number of dofs
    # is statically known from the function space and the shape
    D = zeros(length(iface_x), length(jface_x))
    for i in eachindex(iface_x)
        for j in eachindex(jface_x)
            d = iface_x[i] - jface_x[j]
            D[i, j] = d ⋅ d
        end
    end

    # Check the distances
    # xc = center(jcell_nodes_g)
    # cell_radius = minimum(x -> norm(x - xc), )
    extremas = [extrema(@view D[i, :]) for i in axes(D, 1)]
    max_dist = maximum(last.(extremas))
    # @show first.(extremas)
    # @show iface_x, jface_x
    @assert all(first.(extremas) .< max(geom_factor*1e-20*max_dist, 10*eps(0.0)))

    # Identify the pairs. The array `i2j` indicates that for
    # the `ith` dof of the face (ie `face_dofs_i[i]` of  cell i), the corresponding dof
    # of face `j` is `i2j[i]` (ie `face_dofs_j[i2j]` of cell j).
    i2j = [argmin(@view D[i, :]) for i in axes(D, 1)]

    return i2j
end

"""
    max_ndofs(dhl::DofHandler)

Count maximum number of dofs per cell, all components mixed

"""
max_ndofs(dhl::DofHandler) = maximum(dhl.ndofs)

"""
    get_ndofs(dhl::DofHandler)
    get_ndofs(dhl::DofHandler, icell)
    get_ndofs(dhl::DofHandler, icell, icomp::Int)
    get_ndofs(dhl::DofHandler, icell, icomp::Vector{Int})


Number of dofs in a given cell (with `icell` or for the whole space).

If only `icell` is provided, the total (accross all components) number of
dofs is returned.

# Example
```julia
mesh = one_cell_mesh(:line)
U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
dhl = Bcube.get_dhl(U)
@show Bcube.get_ndofs(dhl, 1, 1)
@show Bcube.get_ndofs(dhl, 1, [1, 2])
```
"""
@inline get_ndofs(dhl::DofHandler, icell, icomp::Int) = dhl.ndofs[icell, icomp]

@inline function get_ndofs(dhl::DofHandler, icell, icomp::AbstractVector{Int})
    sum(dhl.ndofs[icell, icomp])
end
@inline function get_ndofs(dhl::DofHandler, icell, icomp::UnitRange{Int})
    sum(view(dhl.ndofs, icell:icell, icomp))
end

get_ndofs(dhl::DofHandler, icell) = sum(view(dhl.ndofs, icell, :))

get_ndofs(dhl::DofHandler) = dhl.ndofs_tot

"""
    get_dof(dhl::DofHandler, icell)
    get_dof(dhl::DofHandler, icell, icomp::Int)
    get_dof(dhl::DofHandler, icell, icomp::Int, idof::Int)

Global indices (of index) of the dofs in a given cell `icell`. The dofs relative
to a specific component can be obtained by precising `icomp`; and a specific dof number
can be obtained by futher precising the local dof `idof`.

# Example
```julia
mesh = one_cell_mesh(:line)
U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
dhl = Bcube.get_dhl(U)
@show get_dof(dhl, 1, 1, 1)
```
"""
function get_dof(dhl::DofHandler, icell, icomp::Int, idof::Int)
    dhl.iglob[dhl.offset[icell, icomp] + idof]
end

function get_dof(dhl::DofHandler, icell, icomp::Int)
    view(dhl.iglob, dhl.offset[icell, icomp] .+ (1:get_ndofs(dhl, icell, icomp)))
end
function get_dof(dhl::DofHandler, icell)
    view(dhl.iglob, dhl.offset[icell, 1] .+ (1:get_ndofs(dhl, icell)))
end
function get_dof(dhl::DofHandler, icell, ::Val{N}) where {N}
    dhl.iglob[dhl.offset[icell, 1] .+ SVector{N}(1:N)]
end
function get_dof(dhl::DofHandler, icell::UnitRange)
    view(
        dhl.iglob,
        dhl.offset[first(icell), 1] .+
        (1:(dhl.offset[last(icell), 1] + get_ndofs(dhl, last(icell)))),
    )
end
function get_dof(dhl::DofHandler, icell, icomp::Int, ::Val{N}) where {N}
    @assert N == get_ndofs(dhl, icell, icomp) "error N ≠ ndofs"
    dhl.iglob[dhl.offset[icell, icomp] .+ SVector{N}(1:N)]
end

"""
    get_ncomponents(dhl::DofHandler)

Number of components handled by a DofHandler
"""
get_ncomponents(dhl::DofHandler) = size(dhl.offset, 2)

"""
    apply_periodicity!(
        iglob,
        offset,
        fSpace::AbstractFunctionSpace,
        ncomps::Integer,
        periodicity::BoundaryFaceDomain{M, BC};
        geom_factor = 1.0,
    ) where {M, BC <: PeriodicBCType}

Helper to edit the `iglob` array to take into account periodicity. For continuous
spaces, periodicity involves some dofs (in connection through the periodicity)
sharing the same identifier.
"""
function apply_periodicity!(
    iglob,
    offset,
    fSpace::AbstractFunctionSpace,
    ncomps::Integer,
    periodicity::BoundaryFaceDomain{M, BC};
    geom_factor = 1.0,
) where {M, BC <: PeriodicBCType}
    foreach_element(periodicity) do face, _, _
        # Unpack cells infos
        # Rq: because we are dealing with a PeriodicBCType,
        # `cnodes_p` already received the periodic-transformation
        cell_n = get_cellinfo_n(face)
        cell_p = get_cellinfo_p(face)
        cnodes_n = nodes(cell_n)
        cnodes_p = nodes(cell_p)
        ctype_n = get_element_type(cell_n)
        ctype_p = get_element_type(cell_p)
        cell_idx_n = get_element_index(cell_n)
        cell_idx_p = get_element_index(cell_p)
        cshape_n = shape(ctype_n)
        cshape_p = shape(ctype_p)

        # Unpack face infos
        side_n = abs(get_cell_side_n(face))
        side_p = abs(get_cell_side_p(face))

        # Identify dofs lying on this face on both sides
        face_dofs_n = idof_by_face_with_bounds(fSpace, cshape_n)[side_n]
        face_dofs_p = idof_by_face_with_bounds(fSpace, cshape_p)[side_p]

        # Geometric identification
        i2j = identify_face_dofs_from_coords(
            fSpace,
            ctype_n,
            cnodes_n,
            face_dofs_n,
            ctype_p,
            cnodes_p,
            face_dofs_p;
            geom_factor,
        )

        # Now, erase the value of iglob for all `p` entries with `n` entries
        for icomp in 1:ncomps
            offset_n = offset[cell_idx_n, icomp]
            offset_p = offset[cell_idx_p, icomp]
            for (i, j) in enumerate(i2j)
                dof_n = face_dofs_n[i]
                dof_p = face_dofs_p[j]
                iglob[offset_p + dof_p] = iglob[offset_n + dof_n]
            end
        end # loop on ncomps
    end # loop on faces of periodic domain
end

function apply_periodicity!(
    iglob,
    offset,
    fSpace::AbstractFunctionSpace,
    ncomps::Integer,
    periodicities;
    geom_factor = 1.0,
)
    foreach(periodicities) do periodicity
        apply_periodicity!(iglob, offset, fSpace, ncomps, periodicity; geom_factor)
    end
end
