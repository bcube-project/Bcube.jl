function Bcube.read_file(
    ::Bcube.HDF5IoHandler,
    filepath::String;
    domainNames = String[],
    varnames = nothing,
    topodim = 0,
    spacedim = 0,
    verbose = false,
    kwargs...,
)
    # Open the file
    file = h5open(filepath, "r")
    root = file

    # Read (unique) CGNS base
    cgnsBase = get_cgns_base(root)

    # Read base dimensions (topo and space)
    dims = get_value(cgnsBase)
    topodim = topodim > 0 ? topodim : dims[1]
    spacedim = spacedim > 0 ? spacedim : dims[2]
    verbose && println("topodim = $topodim, spacedim = $spacedim")

    # Find the list of Zone_t
    zones = get_children(cgnsBase; type = "Zone_t")
    if length(zones) == 0
        error("Could not find any Zone_t node in the file")
    elseif length(zones) > 1
        error("The file contains several Zone_t nodes, only one zone is supported for now")
    end
    zone = first(zones)

    # Read zone
    zoneCGNS = read_zone(zone, varnames, topodim, spacedim, verbose)

    # Close the file
    close(file)

    # Build Bcube Mesh
    mesh = cgns_mesh_to_bcube_mesh(zoneCGNS)

    # Build Bcube MeshCellData & MeshPointData
    if !isnothing(varnames)
        data = flow_solutions_to_bcube_data(zoneCGNS.fSols)
    else
        data = nothing
    end

    # Should we return something when pointData and/or cellData is nothing? Or remove it completely from the returned Tuple?
    return (; mesh, data)
end

"""
Read a CGNS Zone node.

Return a NamedTuple with node coordinates, cell-to-type connectivity (type is a integer),
cell-to-node connectivity, boundaries (see `read_zoneBC`), and a dictionnary of flow solutions (see `read_solutions`)

-> (; coords, c2t, c2n, bcs, fSols)
"""
function read_zone(zone, varnames, topo_dim, space_dim, verbose)
    # Preliminary check
    zoneType = get_value(get_child(zone; type = "ZoneType_t"))
    @assert zoneType == "Unstructured" "Only unstructured zone are supported"

    # Number of elements
    nvertices, ncells, nbnd = get_value(zone)
    verbose && println("nvertices = $nvertices, ncells = $ncells")

    # Read GridCoordinates
    gridCoordinates = get_child(zone; type = "GridCoordinates_t")
    coordXNode = get_child(gridCoordinates; name = "CoordinateX")
    X = get_value(coordXNode)
    coords = zeros(eltype(X), nvertices, space_dim)
    coords[:, 1] .= X
    suffixes = ["X", "Y", "Z"]
    for (idim, suffix) in enumerate(suffixes[1:space_dim])
        node = get_child(gridCoordinates; name = "Coordinate" * suffix)
        coords[:, idim] .= get_value(node)
    end

    # Read all elements
    elts = map(read_connectivity, get_children(zone; type = "Elements_t"))

    # Filter "volumic" elements to build the volumic connectivities arrays
    volumicElts = filter(elt -> is_volumic_entity(first(elt.c2t), topo_dim), elts)
    c2t = mapreduce(elt -> elt.c2t, vcat, volumicElts)
    c2n = mapreduce(elt -> elt.c2n, vcat, volumicElts)

    # Read all BCs and then keep only the ones whose topo dim is equal to the base topo dim minus 1
    bcs = read_zoneBC(zone, elts, verbose)
    filter!(bc -> (bc.bcdim == topo_dim - 1) || (bc.bcdim == -1), bcs)

    # Read FlowSolutions
    if !isnothing(varnames)
        fSols = read_solutions(zone, varnames, verbose)
    else
        fSols = nothing
    end

    # @show coords
    # @show c2t
    # @show c2n
    # @show bcs
    # @show fSols
    return (; coords, c2t, c2n, bcs, fSols)
end

"""
Read an "Elements_t" node and returns a named Tuple of three elements:
* `erange`, the content of the `ElementRange` node
* `c2t`, the cell -> entity_type connectivity
* `c2n`, the cell -> node connectivity, flattened if `reshape = false`, as an array (nelts, nnodes_by_elt) if `reshape = true`
* `name`, only for dbg
"""
function read_connectivity(node, reshape = false)
    @assert get_cgns_type(node) == "Elements_t"

    # Build cell to (cgns) type
    code, _ = get_value(node)
    erange = get_value(get_child(node; name = "ElementRange"))
    nelts = erange[2] - erange[1] + 1
    c2t = fill(code, nelts)

    # Build cell to node and reshapce
    c2n = get_value(get_child(node; name = "ElementConnectivity"))

    nnodes_by_elt = nnodes(cgns_entity_to_bcube_entity(code))
    reshape && (c2n = reshape(c2n, nelts, nnodes_by_elt))

    return (; erange, c2t, c2n, name = get_name(node))
end

"""
Read the "ZoneBC_t" node to build bnd connectivities.

See `read_bc` for more information of what is returned.
"""
function read_zoneBC(zone, elts, verbose)
    zoneBC = get_child(zone; type = "ZoneBC_t")
    bcs = map(bc -> read_bc(bc, elts, verbose), get_children(zoneBC; type = "BC_t"))
    return bcs
end

"""
Read a BC node.

Return a named Tuple (bcname, bcnodes, bcdim) where bcnodes is an array of the nodes
belonging to this BC.
"""
function read_bc(bc, elts, verbose)
    # BC name
    familyName = get_child(bc; type = "FamilyName_t")
    bcname = isnothing(familyName) ? get_name(bc) : get_value(familyName)
    verbose && println("Reading BC '$bcname'")

    # BC connectivity
    bc_type = get_value(get_child(bc; type = "GridLocation_t"))
    indexRange = get_child(bc; type = "IndexRange_t")
    pointList = get_child(bc; type = "IndexArray_t")

    # BC topodim : it's not always possible to determine it, so it's negative by default
    bcdim = -1

    if bc_type in ["CellCenter", "FaceCenter"]
        if !isnothing(indexRange)
            verbose && println("GridLocation is $(bc_type) with IndexRange")

            # This is a bit complex because nothing prevents an IndexRange to span over multiples Elements_t
            erange = get_value(indexRange)

            # Allocate the array of node indices corresponding to the BC
            nelts_bc = erange[2] - erange[1] + 1
            T = eltype(first(elts).c2n[1])
            bcnodes = T[]
            sizehint!(bcnodes, nelts_bc * 4) # we assume 4 nodes by elements

            nelts_found = 0

            # Loop over all the Elements_t 'nodes'
            for elt in elts
                # verbose && println("Searching for elements in Elements_t '$(elt.name)'")
                i1, i2 = elt.erange
                etype = cgns_entity_to_bcube_entity(first(elt.c2t))
                nnodes_by_elt = nnodes(etype)

                if i1 <= erange[1] <= i2
                    # Compute how many nodes are concerned in this Elements_t,
                    # and the offset in the connectivity
                    nelts_concerned = min(i2, erange[2]) - erange[1] + 1
                    nnodes_concerned = nelts_concerned * nnodes_by_elt
                    offset = (erange[1] - i1) * nnodes_by_elt

                    push!(bcnodes, elt.c2n[(1 + offset):(offset + nnodes_concerned)]...)

                    nelts_found += nelts_concerned
                    verbose && println("$(nelts_concerned) elts found in '$(elt.name)'")

                    bcdim = Bcube.topodim(etype)

                    # Check if we've found all the elements in this connectivity
                    (erange[2] <= i2) && break
                end

                if i1 <= erange[2] <= i2
                    # Compute how many nodes are concerned in this Elements_t,
                    # and the offset in the connectivity
                    nelts_concerned = erange[2] - max(i1, erange[1]) + 1
                    nnodes_concerned = nelts_concerned * nnodes_by_elt
                    offset = (max(i1, erange[1]) - i1) * nnodes_by_elt

                    push!(bcnodes, elt.c2n[(1 + offset):(offset + nnodes_concerned)]...)

                    nelts_found += nelts_concerned
                    verbose && println("$(nelts_concerned) elts found in '$(elt.name)'")

                    bcdim = Bcube.topodim(etype)
                end
            end

            @assert nelts_found == nelts_bc "Missing elements for BC"

            # Once we've found all the nodes, we must remove duplicates
            # Note : using sort! + unique! is much more efficient than calling "unique"
            sort!(bcnodes)
            unique!(bcnodes)

        elseif !isnothing(pointList)
            # Elements indices
            elts_ind = vec(get_value(pointList))
            sort!(elts_ind)

            # Allocate the array of node indices corresponding to the BC
            nelts_bc = length(elts_ind)
            T = eltype(first(elts).c2n[1])
            bcnodes = T[]
            sizehint!(bcnodes, nelts_bc * 4) # we assume 4 nodes by elements

            icurr = 1
            for elt in elts
                verbose && println("Searching for elements in Elements_t '$(elt.name)'")
                i1, i2 = elt.erange
                etype = cgns_entity_to_bcube_entity(first(elt.c2t))
                nnodes_by_elt = nnodes(etype)

                (icurr < i1) && continue
                (icurr > i2) && continue

                if elts_ind[end] >= i2
                    iEnd = i2
                else
                    iEnd = findfirst(i -> i > i2, view(elts_ind, icurr:nelts_bc)) - 1
                end
                offset = (elts_ind[icurr] - i1) * nnodes_by_elt
                push!(bcnodes, elt.c2n[(1 + offset):(nnodes_by_elt * (iEnd - i1 + 1))]...)
                icurr = iEnd + 1

                (icurr > nelts_bc) && break

                # Element-wise version (OK, but very slow)
                # while i1 <= elts_ind[icurr] <= i2
                #     offset = (elts_ind[icurr] - i1) * nnodes_by_elt
                #     push!(bcnodes, elt.c2n[(1 + offset):(offset + nnodes_by_elt)]...)
                #     (icurr == nelts_bc) && break
                #     icurr += 1
                # end
            end

            @assert icurr >= nelts_bc
        else
            error("Could not find either the PointRange nor the PointList")
        end
    elseif bc_type == "Vertex"
        if !isnothing(pointList)
            bcnodes = get_value(pointList)
        elseif !isnothing(indexRange)
            erange = get_value(indexRange)
            bcnodes = collect(erange[1]:erange[2])
        else
            error("Could not find either the PointRange nor the PointList")
        end

        # TODO : we could try to guess `bcdim` by search the Elements_t containing
        # the points of the PointList
    else
        error("BC GridLocation '$(bc_type)' not implemented")
    end

    return (; bcname, bcnodes, bcdim)
end

"""
Read all the flow solutions in the Zone, filtering data arrays whose name is not in the `varnames` list

# TODO : check if all varnames have been found
"""
function read_solutions(zone, varnames, verbose)
    # fSols =
    #     map(fs -> read_solution(fs, varnames), get_children(zone; type = "FlowSolution_t"))

    # n_vertex_fsol = count(fs -> fs.gridLocation == "Vertex", fSols)
    # n_cell_fsol = count(fs -> fs.gridLocation == "CellCenter", fSols)

    # if verbose
    #     (n_vertex_fsol > 1) && println(
    #         "WARNING : found more than one Vertex FlowSolution, reading them all...",
    #     )
    #     (n_cell_fsol > 1) && println(
    #         "WARNING : found more than one CellCenter FlowSolution, reading them all...",
    #     )
    # end

    fSols = Dict(
        get_name(fs) => read_solution(zone, fs, varnames) for
        fs in get_children(zone; type = "FlowSolution_t")
    )

    return fSols
end

"""
Read a FlowSolution node.

Return a NamedTuple with flow solution name, grid location and array of vectors.
"""
function read_solution(zone, fs, varnames)
    # Read GridLocation : we could deal with a missing GridLocation node, by later comparing
    # the length of the DataArray to the number of cells / nodes of the zone. Let's do this
    # later.
    node = get_child(fs; type = "GridLocation_t")
    if isnothing(node)
        _nnodes, _ncells, _ = get_zone_dims(zone)
        dArray = get_child(fs; type = "DataArray_t")
        err_msg = "Could not determine GridLocation in FlowSolution '$(get_name(fs))'"
        @assert !isnothing(dArray) err_msg
        x = get_value(dArray)
        if length(x) == _nnodes
            gridLocation = "Vertex"
        elseif length(x) == _ncells
            gridLocation = "CellCenter"
        else
            error(err_msg)
        end
        @warn "Missing GridLocation in FlowSolution '$(get_name(fs))', autoset to '$gridLocation'"
    else
        gridLocation = get_value(node)
    end

    # Read variables matching asked "varnames"
    dArrays = get_children(fs; type = "DataArray_t")
    if varnames != "*"
        # filter to obtain only the desired variables names
        filter!(dArray -> get_name(dArray) in varnames, dArrays)
    end
    data = Dict(get_name(dArray) => get_value(dArray) for dArray in dArrays)

    # Flow solution name
    name = get_name(fs)

    return (; name, gridLocation, data)
end

"""
Convert CGNS Zone information into a Bcube `Mesh`.
"""
function cgns_mesh_to_bcube_mesh(zoneCGNS)
    nodes = [Bcube.Node(zoneCGNS.coords[i, :]) for i in 1:size(zoneCGNS.coords, 1)]
    # nodes = map(row -> Bcube.Node(row), eachrow(zoneCGNS.coords)) # problem with Node + Slice

    c2n = Int.(zoneCGNS.c2n)
    c2t = map(cgns_entity_to_bcube_entity, zoneCGNS.c2t)
    c2nnodes = map(nnodes, c2t)
    bc_names = Dict(i => bc.bcname for (i, bc) in enumerate(zoneCGNS.bcs))
    bc_nodes = Dict(i => Int.(bc.bcnodes) for (i, bc) in enumerate(zoneCGNS.bcs))

    return Bcube.Mesh(nodes, c2t, Bcube.Connectivity(c2nnodes, c2n); bc_names, bc_nodes)
end

"""
The input `fSols` is suppose to be a dictionnary FlowSolutionName => (gridlocation, Dict(varname => array))

The output is a dictionnary FlowSolutionName => dictionnary(varname => MeshData)
"""
function flow_solutions_to_bcube_data(fSols)
    # length(fSols) == 0 && (return Dict(), Dict())

    # pointSols = filter(fs -> fs.gridLocation == "Vertex", fSols)
    # cellSols = filter(fs -> fs.gridLocation == "CellCenter", fSols)

    # pointDicts = [fs.data for fs in pointSols]
    # cellDicts = [fs.data for fs in cellSols]

    # pointDict = merge(pointDicts)
    # cellDict = merge(cellDicts)

    # pointDict = Dict(key => MeshPointData(val) for (key, val) in pointDict)
    # cellDict = Dict(key => MeshCellData(val) for (key, val) in cellDict)

    # return pointDict, cellDict

    return Dict(
        fname => Dict(
            varname =>
                t.gridLocation == "Vertex" ? MeshPointData(array) : MeshCellData(array)
            for (varname, array) in t.data
        ) for (fname, t) in fSols
    )
end

"""
Indicate if the Elements node contains "volumic" entities (with respect
to the `topo_dim` argument)
"""
function is_volumic_entity(obj, topo_dim)
    @assert get_cgns_type(obj) == "Elements_t"
    code, _ = get_value(obj)
    return is_volumic_entity(code, topo_dim)
end

function is_volumic_entity(code::Integer, topo_dim)
    Bcube.topodim(cgns_entity_to_bcube_entity(code)) == topo_dim
end

"""
Return nnodes, ncells, nbnd
"""
function get_zone_dims(zone)
    d = get_value(zone)
end