function Bcube.write_file(
    ::Bcube.HDF5IoHandler,
    basename::String,
    mesh::Bcube.AbstractMesh,
    data = nothing,
    it::Integer = -1,
    time::Real = 0.0;
    collection_append = false,
    verbose = false,
    update_mesh = false,
    kwargs...,
)
    mode = collection_append ? "r+" : "w"
    fcpl = HDF5.FileCreateProperties(; track_order = true)
    # fcpl = HDF5.FileCreateProperties()

    h5open(basename, mode; fcpl) do file
        if collection_append
            append_to_cgns_file(file, mesh, data, it, time; verbose, update_mesh)
        else
            create_cgns_file(file, mesh, data; verbose)
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
    it,
    time;
    verbose = false,
    update_mesh = false,
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
    if it >= 0
        bid = get_child(cgnsBase; type = "BaseIterativeData_t")
        if isnothing(bid)
            verbose && println("BaseIterativeData not found, creating it")
            bid = create_cgns_base_iterative_data(cgnsBase, get_name(zone), it, time)
        else
            verbose && println("Appending to BaseIterativeData")
            append_to_base_iterative_data(bid, get_name(zone), it, time)
        end
    else
        verbose && println("Skipping BaseIterativeData because iteration < 0")
    end
end

"""
Create the whole file from scratch
"""
function create_cgns_file(file, mesh, data; verbose)
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
        create_flow_solutions(mesh, zone, data; verbose)
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

function create_flow_solutions(mesh, zone, data; verbose = false)
    if valtype(data) <: Dict
        verbose && println("Named FlowSolution(s) detected")
        for (fname, _data) in data

            # First, we check that the data location is the same
            # for all variables
            # -> a bit ugly, but ok for now
            vars = values(_data)
            U = first(vars)[2]
            @assert all(x -> x[2] == U, vars) "The variables sharing the same FlowSolution must share the same FESpace to export"

            # Then, we create a flowsolution
            create_flow_solutions(mesh, zone, _data, fname; verbose)
        end
    else
        create_flow_solutions(mesh, zone, data, ""; verbose)
    end
end

"""
This function is a trick to refactor some code
"""
function create_flow_solutions(mesh, zone, data, fname; verbose = false)
    @assert all(x -> is_fespace_supported(x[2]), values(data))

    cellCenter =
        filter(((k, v),) -> Bcube.get_degree(Bcube.get_function_space(v[2])) == 0, data)
    create_flow_solution(
        zone,
        cellCenter,
        isempty(fname) ? "FlowSolutionCell" : fname,
        false,
        Base.Fix2(var_on_centers, mesh);
        verbose,
    )

    nodeCenter =
        filter(((k, v),) -> Bcube.get_degree(Bcube.get_function_space(v[2])) == 1, data)
    create_flow_solution(
        zone,
        nodeCenter,
        isempty(fname) ? "FlowSolutionVertex" : fname,
        true,
        Base.Fix2(var_on_vertices, mesh);
        verbose,
    )
end

function create_flow_solution(zone, data, fname, isVertex, projection; verbose)
    isempty(data) && return

    verbose && println("Writing FlowSolution '$fname'")

    fnode = create_cgns_node(zone, fname, "FlowSolution_t"; type = "MT")
    gdValue = isVertex ? "Vertex" : "CellCenter"
    create_grid_location(fnode, gdValue)

    for (name, val) in data
        var = val[1]
        y = projection(var)
        create_cgns_node(fnode, name, "DataArray_t"; type = "R8", value = y)
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

function append_to_base_iterative_data(bid, zonename, it, time)
    numberOfZones = get_child(bid; name = "NumberOfZones")
    nzones = first(get_value(numberOfZones)) # 'first' because `get_value` returns an Array
    update_cgns_data(numberOfZones, nzones + Int32(1))

    timeValues = get_child(bid; name = "TimeValues")
    data = push!(get_value(timeValues), time)
    update_cgns_data(timeValues, data)

    # Warning : CGNS also allows for a "TimeDurations" node instead of "IterationValues"
    iterationValues = get_child(bid; name = "IterationValues")
    data = push!(get_value(iterationValues), it)
    update_cgns_data(iterationValues, data)

    zonePointers = get_child(bid; name = "ZonePointers")
    data = read(zonePointers[" data"])
    new_data = zeros(eltype(data), (32, 1, nzones + 1))
    str = str2int8_with_fixed_length(zonename, 32)
    for i in 1:nzones
        new_data[:, 1, i] .= data[:, 1, i]
    end
    new_data[:, 1, end] .= str
    update_cgns_data(zonePointers, new_data)
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