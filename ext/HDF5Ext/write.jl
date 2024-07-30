function Bcube.write_file(
    ::Bcube.HDF5IoHandler,
    basename::String,
    mesh::Bcube.AbstractMesh,
    U_export::Bcube.AbstractFESpace,
    data = nothing,
    it::Integer = -1,
    time::Real = 0.0;
    collection_append = false,
    verbose = false,
    update_mesh = false,
    skip_iterative_data = false,
    kwargs...,
)
    @assert is_fespace_supported(U_export) "Export FESpace not supported yet"

    mode = collection_append ? "r+" : "w"
    fcpl = HDF5.FileCreateProperties(; track_order = true)
    # fcpl = HDF5.FileCreateProperties()

    h5open(basename, mode; fcpl) do file
        if collection_append
            append_to_cgns_file(
                file,
                mesh,
                data,
                U_export,
                it,
                time;
                verbose,
                update_mesh,
                skip_iterative_data,
            )
        else
            create_cgns_file(
                file,
                mesh,
                data,
                U_export,
                it,
                time;
                verbose,
                skip_iterative_data,
            )
        end
    end
end

"""
Append data to an existing file.
"""
function append_to_cgns_file(
    file,
    mesh,
    data,
    U_export,
    it,
    time;
    verbose = false,
    update_mesh = false,
    skip_iterative_data,
)
    @assert !update_mesh "Mesh update not implemented yet"

    # Read (unique) CGNS base
    cgnsBase = get_cgns_base(file)

    # Find the list of Zone_t
    zones = get_children(cgnsBase; type = "Zone_t")
    if length(zones) == 0
        error("Could not find any Zone_t node in the file")
    elseif length(zones) > 1
        error("The file contains several Zone_t nodes, only one zone is supported for now")
    end
    zone = first(zones)

    # If it >= 0, update/create BaseIterativeData
    if it >= 0 && !skip_iterative_data
        append_to_base_iterative_data(cgnsBase, get_name(zone), it, time)
    else
        verbose &&
            println("Skipping BaseIterativeData because iteration < 0 or asked to skip it")
    end

    # Append solution
    create_flow_solutions(mesh, zone, data, U_export, it, true; verbose)

    # Append to ZoneIterativeData
    if it >= 0 && !skip_iterative_data
        append_to_zone_iterative_data(zone, it; verbose)
    end
end

"""
Create the whole file from scratch
"""
function create_cgns_file(
    file,
    mesh,
    data,
    U_export,
    it,
    time;
    verbose,
    skip_iterative_data,
)
    # @show file.id
    # @show HDF5.API.h5i_get_file_id(file)
    # HDF5.API.h5f_set_libver_bounds(
    #     # file.id,
    #     file,
    #     HDF5.API.H5F_LIBVER_V18,
    #     HDF5.API.H5F_LIBVER_LATEST,
    # )
    # HDF5.API.h5f_flush(file.id)

    # Root element
    set_cgns_root!(file)

    # Library version
    create_cgns_node(
        file,
        "CGNSLibraryVersion",
        "CGNSLibraryVersion_t";
        type = "R4",
        value = Float32[3.2],
    )

    # Base
    cgnsBase = create_cgns_node(
        file,
        "Base",
        "CGNSBase_t";
        value = Int32[Bcube.topodim(mesh), Bcube.spacedim(mesh)],
    )

    # Zone
    zone = create_cgns_node(
        cgnsBase,
        "Zone",
        "Zone_t";
        value = reshape(Int32[nnodes(mesh), ncells(mesh), 0], 1, 3),
    )

    # ZoneType
    create_cgns_node(
        zone,
        "ZoneType",
        "ZoneType_t";
        value = Int8.(Vector{UInt8}("Unstructured")),
        type = "C1",
    )

    # GridCoordinates
    gridCoordinates =
        create_cgns_node(zone, "GridCoordinates", "GridCoordinates_t"; type = "MT")
    suffixes = ("X", "Y", "Z")
    for idim in 1:Bcube.spacedim(mesh)
        create_cgns_node(
            gridCoordinates,
            "Coordinate$(suffixes[idim])",
            "DataArray_t";
            type = "R8",
            value = [get_coords(node, idim) for node in get_nodes(mesh)],
        )
    end

    # Write elements
    create_cgns_elements(mesh, zone; write_bnd_faces = false, verbose)

    # Write BCs
    create_cgns_bcs(mesh, zone; verbose)

    # Write solution
    if !isnothing(data)
        create_flow_solutions(mesh, zone, data, U_export, it, false; verbose)
    end

    # Base and zone iterative data
    if it >= 0 && !skip_iterative_data
        verbose && println("Creating BaseIterativeData and ZoneIterativeData")
        create_cgns_base_iterative_data(cgnsBase, get_name(zone), it, time)
        create_cgns_zone_iterative_data(zone, it; verbose)
    end
end

"""
Special stuff for the root node
Adapted from Maia :
https://github.com/onera/Maia/blob/3a5030aa3b0696cdbae0c9dd08ac641842be33a3/maia/io/hdf/_hdf_cgns.py#L95
"""
function set_cgns_root!(root)
    # fcpl = HDF5.get_create_properties(root)
    # HDF5.set_track_order!(fcpl, true)
    add_cgns_string_attr!(root, "label", "Root Node of HDF5 File", 33)
    add_cgns_string_attr!(root, "name", "HDF5 MotherNode", 33)
    add_cgns_string_attr!(root, "type", "MT", 3)

    v = HDF5.libversion
    version = "HDF5 Version $(v.major).$(v.minor).$(v.patch)"
    root[" hdf5version"] = str2int8_with_fixed_length(version, 33)

    # @show HDF5.API.H5T_NATIVE_DOUBLE
    # @show HDF5.API.H5T_NATIVE_FLOAT
    # @show HDF5.API.H5T_IEEE_F32BE
    # @show HDF5.API.H5T_IEEE_F32LE
    # @show HDF5.API.H5T_IEEE_F64BE
    # @show HDF5.API.H5T_IEEE_F64LE
    if HDF5.API.H5T_NATIVE_FLOAT == HDF5.API.H5T_IEEE_F32BE
        format = "IEEE_BIG_32"
    elseif HDF5.API.H5T_NATIVE_FLOAT == HDF5.API.H5T_IEEE_F32LE
        format = "IEEE_LITTLE_32"
    elseif HDF5.API.H5T_NATIVE_FLOAT == HDF5.API.H5T_IEEE_F64BE
        format = "IEEE_BIG_64"
    elseif HDF5.API.H5T_NATIVE_FLOAT == HDF5.API.H5T_IEEE_F64LE
        format = "IEEE_LITTLE_64"
    else
        @warn "Could determine float type, assuming IEEE_LITTLE_32"
        format = "IEEE_LITTLE_32"
    end
    root[" format"] = str2int8(format)
end

function create_cgns_elements(mesh, zone; write_bnd_faces = false, verbose = false)
    @assert !write_bnd_faces "not implemented yet"

    verbose && println("Writing elements")

    # Found the different type of cells in the mesh
    celltypes = union_types(eltype(Bcube.entities(mesh, :cell)))

    # Count number of elements for each type
    typeCount = Dict(ct => 0 for ct in celltypes)
    foreach(ct -> typeCount[typeof(ct)] += 1, Bcube.cells(mesh))

    # Allocate connectivity array for each type
    conn = Dict(ct => zeros(typeCount[ct], nnodes(ct)) for ct in celltypes)
    offset = Dict(ct => 1 for ct in celltypes)

    # Fill it
    for cInfo in Bcube.DomainIterator(CellDomain(mesh))
        ct = typeof(Bcube.celltype(cInfo))
        i = offset[ct]
        conn[ct][i, :] .= Bcube.get_nodes_index(cInfo)
        offset[ct] += 1
    end

    # Create Elements nodes
    i = 0
    for ct in celltypes
        eltsName = string(ct)
        eltsName = replace(eltsName, "Bcube." => "")
        eltsName = eltsName[1:(end - 2)]
        verbose && println("Writing $eltsName")
        elts = create_cgns_node(
            zone,
            eltsName,
            "Elements_t";
            value = Int32[BCUBE_ENTITY_TO_CGNS_ENTITY[ct], 0],
        )
        create_cgns_node(
            elts,
            "ElementRange",
            "IndexRange_t";
            value = Int32[i + 1, i + typeCount[ct]],
        )
        create_cgns_node(
            elts,
            "ElementConnectivity",
            "DataArray_t";
            value = Int32.(vec(transpose(conn[ct]))),
        )
        i += typeCount[ct]
    end
end

"""
Create the ZoneBC node and the associated BC nodes.

For now, BCs are defined by a list of nodes. We could implement a version where BCs
are defined by a list of faces (this necessitates faces nodes in the Elements_t part).
"""
function create_cgns_bcs(mesh, zone; verbose = false)
    zoneBC = create_cgns_node(zone, "ZoneBC", "ZoneBC_t"; type = "MT")
    for (tag, name) in boundary_names(mesh)
        # for (name, nodes) in zip(boundary_names(mesh), Bcube.boundary_nodes(mesh))
        bcnodes = Bcube.boundary_nodes(mesh, tag)
        bc = create_cgns_node(zoneBC, name, "BC_t"; type = "C1", value = str2int8("BCWall"))
        create_grid_location(bc, "Vertex")
        create_cgns_node(
            bc,
            "PointList",
            "IndexArray_t";
            type = "I4",
            value = Int32.(transpose(bcnodes)),
        )
    end
end

"""
`append` indicates if the FlowSolution(s) may already exist (and hence completed) or not.
"""
function create_flow_solutions(mesh, zone, data, U_export, it, append; verbose = false)
    if valtype(data) <: Dict
        verbose && println("Named FlowSolution(s) detected")
        for (fname, _data) in data
            # Then, we create a flowsolution
            create_flow_solutions(mesh, zone, _data, U_export, fname, it, append; verbose)
        end
    else
        create_flow_solutions(mesh, zone, data, U_export, "", it, append; verbose)
    end
end

"""
This function is a trick to refactor some code
"""
function create_flow_solutions(
    mesh,
    zone,
    data,
    U_export,
    fname,
    it,
    append;
    verbose = false,
)
    cellCenter = filter(((name, var),) -> var isa Bcube.MeshData{Bcube.CellData}, data)
    _fname = isempty(fname) ? "FlowSolutionCell" : fname
    (it >= 0) && (_fname *= iteration_to_string(it))
    create_flow_solution(
        zone,
        cellCenter,
        _fname,
        false,
        Base.Fix2(var_on_centers, mesh),
        append;
        verbose,
    )

    nodeCenter = filter(((name, var),) -> !(var isa Bcube.MeshData{Bcube.CellData}), data)
    _fname = isempty(fname) ? "FlowSolutionVertex" : fname
    (it >= 0) && (_fname *= iteration_to_string(it))
    create_flow_solution(
        zone,
        nodeCenter,
        _fname,
        true,
        Base.Fix2(var_on_vertices, mesh),
        append;
        verbose,
    )
end

"""
WARNING : vector variables are not supported for now !!
"""
function create_flow_solution(zone, data, fname, isVertex, projection, append; verbose)
    isempty(data) && return

    verbose && println("Writing FlowSolution '$fname'")

    # Try to get an existing node. If it exists and append=true,
    # an error is raised.
    # Otherwise, a new node is created.
    fnode = get_child(zone; name = fname, type = "FlowSolution_t")
    if !isnothing(fnode) && !append
        error("The node '$fname' already exists")
    else
        fnode = create_cgns_node(zone, fname, "FlowSolution_t"; type = "MT")
        gdValue = isVertex ? "Vertex" : "CellCenter"
        create_grid_location(fnode, gdValue)
    end

    for (name, var) in data
        y = projection(var)

        if ndims(y) == 1
            # Scalar field
            create_cgns_node(fnode, name, "DataArray_t"; type = "R8", value = y)

        elseif ndims(y) == 2 && size(y, 2) <= 3
            # Vector field
            for (col, suffix) in zip(eachcol(y), ("X", "Y", "Z"))
                create_cgns_node(
                    fnode,
                    name * suffix,
                    "DataArray_t";
                    type = "R8",
                    value = col,
                )
            end
        else
            error("wrong dimension for y")
        end
    end
    return fnode
end

function create_cgns_node(parent, name, label; type = "I4", value = nothing)
    child = create_group(parent, name; track_order = true)
    set_cgns_attr!(child, name, label, type)
    if !isnothing(value)
        append_cgns_data(child, value)
    end
    return child
end

function create_grid_location(parent, value)
    return create_cgns_node(
        parent,
        "GridLocation",
        "GridLocation_t";
        type = "C1",
        value = str2int8(value),
    )
end

function is_fespace_supported(U)
    Bcube.is_discontinuous(U) && return false

    fs = Bcube.get_function_space(U)
    if Bcube.get_type(fs) <: Bcube.Lagrange && Bcube.get_degree(fs) <= 1
        return true
    elseif Bcube.get_type(fs) <: Bcube.Taylor && Bcube.get_degree(fs) <= 0
        return true
    end
    return false
end

function create_cgns_base_iterative_data(parent, zonename, it, time)
    bid = create_cgns_node(
        parent,
        "BaseIterativeData",
        "BaseIterativeData_t";
        type = "I4",
        value = Int32(1),
    )
    create_cgns_node(bid, "NumberOfZones", "DataArray_t"; type = "I4", value = Int32(1))
    create_cgns_node(bid, "TimeValues", "DataArray_t"; type = "R8", value = time)
    create_cgns_node(bid, "IterationValues", "DataArray_t"; type = "I4", value = Int32(it))
    str = str2int8_with_fixed_length(zonename, 32)
    create_cgns_node(
        bid,
        "ZonePointers",
        "DataArray_t";
        type = "C1",
        value = reshape(str, 32, 1, 1),
    )

    return bid
end

"""
# Warnings
* CGNS also allows for a "TimeDurations" node instead of "IterationValues"
* a unique zone is assumed
"""
function append_to_base_iterative_data(cgnsBase, zonename, it, time)
    bid = get_child(cgnsBase; type = "BaseIterativeData_t")

    # Check if node exists, if not -> create it
    if isnothing(bid)
        verbose && println("BaseIterativeData not found, creating it")
        return create_cgns_base_iterative_data(cgnsBase, zonename, it, time)
    end

    # First, we check if the given iteration is not already known from the
    # BaseIterativeData. If it is the case, we don't do anything.
    # Note that in the maia example, the node IterationValues does not exist,
    # hence we skip this part if the node is not found.
    iterationValues = get_child(bid; name = "IterationValues")
    if !isnothing(iterationValues)
        iterations = get_value(iterationValues)
        (it ∈ iterations) && return

        # Append iteration to the list of iteration values
        push!(iterations, it)
        update_cgns_data(iterationValues, iterations)

        # Increase the number of iterations stored in this BaseIterativeData
        update_cgns_data(bid, length(iterations))
    end

    numberOfZones = get_child(bid; name = "NumberOfZones")
    data = get_value(numberOfZones)
    nsteps = length(data)
    push!(data, 1)
    update_cgns_data(numberOfZones, data)

    timeValues = get_child(bid; name = "TimeValues")
    data = push!(get_value(timeValues), time)
    update_cgns_data(timeValues, data)

    zonePointers = get_child(bid; name = "ZonePointers")
    data = read(zonePointers[" data"])
    new_data = zeros(eltype(data), (32, 1, nsteps + 1))
    str = str2int8_with_fixed_length(zonename, 32)
    for i in 1:nsteps
        new_data[:, 1, i] .= data[:, 1, i]
    end
    new_data[:, 1, end] .= str
    update_cgns_data(zonePointers, new_data)
end

"""
Call to this function must be performed AFTER the FlowSolution creation
"""
function create_cgns_zone_iterative_data(zone, it; verbose)
    # Find the FlowSolution name that matches the iteration
    fsname = "NotFound"
    for fs in get_children(zone; type = "FlowSolution_t")
        _fsname = get_name(fs)
        if endswith(_fsname, iteration_to_string(it))
            fsname = _fsname
            break
        end
    end
    verbose && println("Attaching iteration $it to FlowSolution '$fsname'")

    # Create node
    zid = create_cgns_node(zone, "ZoneIterativeData", "ZoneIterativeData_t"; type = "MT")
    str = str2int8_with_fixed_length(fsname, 32)
    create_cgns_node(
        zid,
        "FlowSolutionPointers",
        "DataArray_t";
        type = "C1",
        value = reshape(str, 32, 1),
    )

    return zid
end

"""
Call to this function must be performed AFTER the FlowSolution creation

# Warning
CGNS does seem to allow multiple FlowSolution nodes for one time step in a Zone.
"""
function append_to_zone_iterative_data(zone, it; verbose)
    # Find the FlowSolution name that matches the iteration
    fsname = "NotFound"
    for fs in get_children(zone; type = "FlowSolution_t")
        _fsname = get_name(fs)
        if endswith(_fsname, iteration_to_string(it))
            fsname = _fsname
            break
        end
    end
    verbose && println("Attaching iteration $it to FlowSolution '$fsname'")

    # Get ZoneIterativeData node (or create it)
    zid = get_child(zone; type = "ZoneIterativeData_t")
    isnothing(zid) && (return create_cgns_zone_iterative_data(zone, it))

    # Append to FlowSolutionPointers
    fsPointers = get_child(zid; name = "FlowSolutionPointers")
    data = read(fsPointers[" data"])
    n = size(data, 2)
    new_data = zeros(eltype(data), (32, n + 1))
    str = str2int8_with_fixed_length(fsname, 32)
    for i in 1:n
        new_data[:, i] .= data[:, i]
    end
    new_data[:, end] .= str
    update_cgns_data(fsPointers, new_data)
end

function append_cgns_data(obj, data)
    _data = data isa AbstractArray ? data : [data]
    dset = create_dataset(obj, " data", eltype(_data), size(_data))
    write(dset, _data)
end

function update_cgns_data(obj, data)
    delete_object(obj, " data")
    append_cgns_data(obj, data)
end

function set_cgns_attr!(obj, name, label, type)
    attributes(obj)["flags"] = Int32[1]
    add_cgns_string_attr!(obj, "label", label, 33)
    add_cgns_string_attr!(obj, "name", name, 33)
    add_cgns_string_attr!(obj, "type", type, 3)
end

"""
It is not trivial to match the exact datatype for attributes.

https://discourse.julialang.org/t/hdf5-jl-variable-length-string/98808/11

TODO : now that I know that even with the correct datatype the file is not
correct according to cgnscheck, once I've solved the cgnscheck problem I should
check that the present function is still necessary.
"""
function add_cgns_string_attr!(obj, name, value, length = sizeof(value))
    dtype = build_cgns_string_dtype(length)
    attr = create_attribute(obj, name, dtype, HDF5.dataspace(value))
    try
        write_attribute(attr, dtype, value)
    catch exc
        delete_attribute(obj, name)
        rethrow(exc)
    finally
        close(attr)
        close(dtype)
    end
end

function build_cgns_string_dtype(length)
    type_id = HDF5.API.h5t_copy(HDF5.hdf5_type_id(String))
    HDF5.API.h5t_set_size(type_id, length)
    HDF5.API.h5t_set_cset(type_id, HDF5.API.H5T_CSET_ASCII)
    dtype = HDF5.Datatype(type_id)
    # @show d
    return dtype
end

str2int8(str) = return Int8.(Vector{UInt8}(str))

function str2int8_with_fixed_length(str, n)
    buffer = zeros(UInt8, n)
    buffer[1:length(str)] .= Vector{UInt8}(str)
    return Int8.(buffer)
end

union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type) = (x,)

iteration_to_string(it) = "#$it"
