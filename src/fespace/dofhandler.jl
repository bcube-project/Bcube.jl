"""
The `DofHandler` handles the degree of freedom numbering. To each degree of freedom
is associated a unique integer.
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

"""
DofHandler(mesh::Mesh, fSpace::AbstractFunctionSpace, ncomponents::Int, isContinuous::Bool)

Constructor of a DofHandler for a `SingleFESpace` on a `Mesh`.
"""
function DofHandler(
    mesh::Mesh,
    fSpace::AbstractFunctionSpace,
    ncomponents::Int,
    isContinuous::Bool,
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
            ct = celltypes[icell]
            s = shape(ct)

            # Cell edges, defined by tuples of vertex absolute indices
            # @ghislainb the second line should be improved, I just want to map the "local indices"
            # tuple of tuple ((1,2), (3,4)) into global indices array of arrays [[23,109],[948, 653]]
            # (arrays instead of tuples because your function "oriented_cell_side" need arrays)
            _e2n = edges2nodes(ct)
            e2n_g = [[inodes_g[i] for i in edge] for edge in _e2n]

            # Cell faces, defined by tuples of vertex absolute indices
            _f2n = faces2nodes(ct)
            f2n_g = [[inodes_g[i] for i in face] for face in _f2n]

            # Loop over the variables
            for icomp in 1:ncomponents
                # Remark : we need to distinguish vertices, edges, faces because two cells
                # can share dofs with an edge without having a face in common.

                #--- Deal with dofs on vertices
                _deal_with_dofs_on_vertices!(
                    dict_n,
                    iglob,
                    offset,
                    icell,
                    inodes_g,
                    s,
                    icomp,
                    fSpace,
                )

                #--- Deal with dofs on edges
                (topodim(mesh) > 1) && _deal_with_dofs_on_edges!(
                    dict_e,
                    iglob,
                    offset,
                    c2n,
                    celltypes,
                    icell,
                    e2n_g,
                    s,
                    icomp,
                    fSpace,
                )

                #--- Deal with dofs on faces
                (topodim(mesh) > 2) && _deal_with_dofs_on_faces!(
                    dict_f,
                    iglob,
                    offset,
                    c2n,
                    celltypes,
                    icell,
                    f2n_g,
                    s,
                    fSpace,
                    icomp,
                )
            end
        end
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
    deal_with_dofs_on_vertices!(dict, iglob, offset, icell::Int, inodes_g, s::AbstractShape, kvar::Int, fs)

Function dealing with dofs shared by different cell through a vertex connection.

TODO : remove kvar

# Arguments
- `dict` may be modified by this routine
- `iglob` may be modified by this routine
- `offset` may be modified by this routine
- `fs` : FunctionSpace of var `kvar`
- `icell` : cell index
- `kvar` : var index
- `s` : shape of `icell`-th cell
- `inodes_g` : global indices of nodes of `icell`
"""
function _deal_with_dofs_on_vertices!(
    dict::Dict{Tuple{Int, Int}, Tuple{Int, Vector{Int}}},
    iglob,
    offset,
    icell::Int,
    inodes_g,
    s::AbstractShape,
    kvar::Int,
    fs::AbstractFunctionSpace,
)
    # Local indices of the dofs on each vertex of the shape
    idofs_array_l = idof_by_vertex(fs, s)

    # Exit prematurely if there are no dof on any vertex of the shape
    length(idofs_array_l[1]) > 0 || return

    # Loop over shape vertices
    for i in 1:nvertices(s)
        inode_g = inodes_g[i] # This is an Int
        idofs_l = idofs_array_l[i] # This is an Array of Int (local indices of dofs of node 'i')

        key = (kvar, inode_g)

        # If the dict already contains the vertex :
        # - we get the neighbour cell index
        # - we copy all the global indices of dofs of `jcell` in the corresponding
        #   global indices of `icell` dofs (`jcell` is useless here actually...)
        if haskey(dict, key)
            jcell, jdofs_g = dict[key]
            for d in 1:length(jdofs_g)
                iglob[offset[icell, kvar] + idofs_l[d]] = jdofs_g[d]
            end

            # If the dict doesn't contain this vertex, we add the global indices
            # of `icell`
        else
            idofs_g = iglob[offset[icell, kvar] .+ idofs_l]
            dict[key] = (icell, idofs_g)
        end
    end
end

"""
    deal_with_dofs_on_edges!(dict, iglob, offset, c2n, celltypes, icell::Int, inodes_g, e2n_g, s::AbstractShape, kvar::Int, fs)

Function dealing with dofs shared by different cell through an edge connection (excluding bord vertices).

TODO : remove kvar

# Arguments
- `dict` may be modified by this routine
- `iglob` may be modified by this routine
- `offset` may be modified by this routine
- `fs` : FunctionSpace of var `kvar`
- `icell` : cell index
- `kvar` : var index
- `s` : shape of `icell`-th cell
- `inodes_g` : global indices of nodes of `icell`
"""
function _deal_with_dofs_on_edges!(
    dict::Dict{Tuple{Int, Set{Int}}, Tuple{Int, Vector{Int}}},
    iglob,
    offset,
    c2n,
    celltypes,
    icell::Int,
    e2n_g,
    s::AbstractShape,
    kvar::Int,
    fs::AbstractFunctionSpace,
)
    # Local indices of the dofs on each edges of the shape
    idofs_array_l = idof_by_edge(fs, s)

    # Exit prematurely if there are no dof on any edge of the shape
    length(idofs_array_l[1]) > 0 || return

    # Loop over the cell edges
    # inodes_g is a Tuple of Int (global indices of nodes defining the edge)
    # idofs_l is an Array of Int (local indices of dofs of edge 'i')
    for (inodes_g, idofs_l) in zip(e2n_g, idofs_array_l)

        # Skip the face if no dof is lying on it
        length(idofs_l) == 0 && continue

        key = (kvar, Set(inodes_g))

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
            jdofs_reordered_g = (jside > 0) ? jdofs_g : reverse(jdofs_g)

            # Copy global indices
            for d in 1:length(jdofs_g)
                iglob[offset[icell, kvar] + idofs_l[d]] = jdofs_reordered_g[d]
            end

            # If the dict doesn't contain this edge, we add the global indices
            # of `icell`
        else
            idofs_g = iglob[offset[icell, kvar] .+ idofs_l]
            dict[key] = (icell, idofs_g)
        end
    end
end

"""
TODO : remove kvar

# Arguments
- f2n_g : local face index -> global nodes indices
"""
function _deal_with_dofs_on_faces!(
    dict,
    iglob,
    offset,
    c2n,
    celltypes,
    icell::Int,
    f2n_g::Vector{Vector{Int}},
    s::AbstractShape,
    fs::AbstractFunctionSpace,
    kvar::Int,
)
    # Local indices of the dofs on each face of the shape, excluding the boundary (nodes and/or edges)
    idofs_array_l = idof_by_face(fs, s) # This is a Tuple of Vector{Int}

    # Exit prematurely if there are no dof on any face of the shape
    sum(length.(idofs_array_l)) > 0 || return

    # Loop over cell faces
    # iface_nodes_g is a Tuple of Int (global indices of nodes defining the face)
    # idofs_l is an Array of Int (local indices of dofs of face 'i')
    for (iface_nodes_g, idofs_l) in zip(f2n_g, idofs_array_l)
        ne = nedges(s)

        # Skip the face if no dof is lying on it
        length(idofs_l) == 0 && continue

        # Create a Set from the global indices of the face nodes to "tag" the face.
        key = (kvar, Set(iface_nodes_g))

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
                iglob[offset[icell, kvar] + idofs_l[d]] = jdofs_reordered_g[d]
            end

            # If the dict doesn't contain this face, we add the global indices
            # of `icell`
        else
            idofs_g = iglob[offset[icell, kvar] .+ idofs_l]
            dict[key] = (icell, idofs_g)
        end
    end
end

"""
    max_ndofs(dhl::DofHandler)

Count maximum number of dofs per cell, all components mixed

"""
max_ndofs(dhl::DofHandler) = maximum(dhl.ndofs)

"""
    get_ndofs(dhl, icell, kvar::Int)

Number of dofs for a given variable in a given cell.

# Example
```julia
mesh = one_cell_mesh(:line)
dhl = DofHandler(mesh, Variable(:u, FunctionSpace(:Lagrange, 1)))
@show get_ndofs(dhl, 1, 1)
```
"""
@inline get_ndofs(dhl::DofHandler, icell, kvar::Int) = dhl.ndofs[icell, kvar]

"""
    get_ndofs(dhl, icell, icomp::Vector{Int})

Number of dofs for a given set of components in a given cell.


# Example
```julia
mesh = one_cell_mesh(:line)
dhl = DofHandler(mesh, Variable(:u, FunctionSpace(:Lagrange, 1); size = 2))
@show get_ndofs(dhl, 1, [1, 2])
```
"""
@inline function get_ndofs(dhl::DofHandler, icell, icomp::AbstractVector{Int})
    sum(dhl.ndofs[icell, icomp])
end
@inline function get_ndofs(dhl::DofHandler, icell, icomp::UnitRange{Int})
    sum(view(dhl.ndofs, icell:icell, icomp))
end

"""
    get_ndofs(dhl::DofHandler, icell)

Number of dofs for a given cell.

Note that for a vector variable, the total (accross all components) number of dofs is returned.

# Example
```julia
mesh = one_cell_mesh(:line)
dhl = DofHandler(mesh, Variable(:u, FunctionSpace(:Lagrange, 1)))
@show get_ndofs(dhl, 1, :u)
```
"""
get_ndofs(dhl::DofHandler, icell) = sum(view(dhl.ndofs, icell, :))

"""
    get_ndofs(dhl::DofHandler)

Total number of dofs. This function takes into account that dofs can be shared by multiple cells.

# Example
```julia
mesh = one_cell_mesh(:line)
dhl = DofHandler(mesh, Variable(:u, FunctionSpace(:Lagrange, 1)))
@show get_ndofs(dhl::DofHandler)
```
"""
get_ndofs(dhl::DofHandler) = dhl.ndofs_tot

"""
    get_dof(dhl::DofHandler, icell, icomp::Int, idof::Int)

Global index of the `idof` local degree of freedom of component `icomp` in cell `icell`.

# Example
```julia
mesh = one_cell_mesh(:line)
dhl = DofHandler(mesh, Variable(:u, FunctionSpace(:Lagrange, 1)))
@show get_dof(dhl, 1, 1, 1)
```
"""
function get_dof(dhl::DofHandler, icell, icomp::Int, idof::Int)
    dhl.iglob[dhl.offset[icell, icomp] + idof]
end

"""
    get_dof(dhl::DofHandler, icell, icomp::Int)

Global indices of all the dofs of a given component in a given cell

# Example
```julia
mesh = one_cell_mesh(:line)
dhl = DofHandler(mesh, Variable(:u, FunctionSpace(:Lagrange, 1)))
@show get_dof(dhl, 1, 1)
```
"""
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
Number of components handled by a DofHandler
"""
get_ncomponents(dhl::DofHandler) = size(dhl.offset, 2)
