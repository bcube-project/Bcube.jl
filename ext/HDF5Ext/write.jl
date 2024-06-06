function Bcube.write_file(
    ::Bcube.HDF5IoHandler,
    basename::String,
    mesh::Bcube.AbstractMesh,
    vars::Dict{String, F} = Dict{String, Bcube.AbstractLazy}(),
    U_export::Bcube.AbstractFESpace = SingleFESpace(FunctionSpace(:Lagrange, 1), mesh),
    it::Integer = -1,
    time::Real = 0.0;
    collection_append = false,
    verbose = false,
    kwargs...,
) where {F <: Bcube.AbstractLazy}
    mode = collection_append ? "a+" : "w"
    # fcpl = HDF5.FileCreateProperties(; track_order = true)
    fcpl = HDF5.FileCreateProperties()

    h5open(basename, mode; fcpl) do file
        set_cgns_root!(file)

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
        for (i, suffix) in enumerate(("X", "Y", "Z"))
            create_cgns_node(
                gridCoordinates,
                "Coordinate$suffix",
                "DataArray_t";
                type = "R8",
                value = [get_coords(node, i) for node in get_nodes(mesh)],
            )
        end

        # Write elements
        create_cgns_elements(mesh, zone; write_bnd_faces = false, verbose)
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
    buffer = zeros(UInt8, 33)
    buffer[1:length(version)] .= Vector{UInt8}(version)
    root[" hdf5version"] = Int8.(buffer)

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
    root[" format"] = Int8.(Vector{UInt8}(format))
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
            value = Int32.(vec(conn[ct])),
        )
        i += typeCount[ct]
    end
end

function create_cgns_node(parent, name, label; type = "I4", value = nothing)
    child = create_group(parent, name; track_order = true)
    set_cgns_attr!(child, name, label, type)
    if !isnothing(value)
        append_data(child, value)
    end
    return child
end

function append_data(obj, data)
    _data = data isa AbstractArray ? data : [data]
    dset = create_dataset(obj, " data", eltype(_data), size(_data))
    write(dset, _data)
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
    dtype = _build_cgns_string_dtype(length)
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

function _build_cgns_string_dtype(length)
    type_id = HDF5.API.h5t_copy(HDF5.hdf5_type_id(String))
    HDF5.API.h5t_set_size(type_id, length)
    HDF5.API.h5t_set_cset(type_id, HDF5.API.H5T_CSET_ASCII)
    dtype = HDF5.Datatype(type_id)
    # @show d
    return dtype
end

union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type) = (x,)