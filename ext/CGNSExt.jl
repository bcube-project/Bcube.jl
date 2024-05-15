module CGNSExt
using Bcube
using JLD2

function Bcube.read_file(
    ::Bcube.CGNSIoHandler,
    filepath::String;
    domainNames = String[],
    varnames = nothing,
    kwargs...,
)
    # Preliminary check
    @assert length(domainNames) == 0 "domainNames arg is not supported for now, leave it empty"

    # Open the file
    f = jldopen(filepath, "r")
    root = Node(f)

    # Find the list of CGNSBase_t
    cgnsBases = get_children(root; type = "CGNSBase_t")
    if length(cgnsBases) == 0
        error("Could not find any CGNSBase_t node in the file")
    elseif length(cgnsBases) > 1
        error("The file contains several CGNSBase_t nodes, only one base is supported")
    end
    cgnsBase = first(cgnsBases)

    # Read base dimensions (topo and space)
    topo_dim, space_dim = get_value(cgnsBase)
    @show topo_dim, space_dim

    # Find the list of Zone_t
    zones = get_children(cgnsBase; type = "Zone_t")
    if length(zones) == 0
        error("Could not find any Zone_t node in the file")
    elseif length(zones) > 1
        error("The file contains several Zone_t nodes, only one base is supported")
    end
    zone = first(zones)

    # Read zone
    read_zone(zone, topo_dim, space_dim)

    # Close the file
    close(f)

    println("hello world")
    return 3
end

"""
Wrapper for the JLD2.Group struct to keep track of the attributes associated to a Group
without needing the parent.
"""
struct Node{T}
    node::T
    name::String
    attrs::Dict{Symbol, String}
end

get_wrapped_node(n::Node) = n.node
get_attribute(n::Node, name::Symbol) = haskey(n.attrs, name) ? n.attrs[name] : nothing
get_cgns_type(n::Node) = get_attribute(n, :label)
get_data_type(n::Node) = get_attribute(n, :type)
get_cgns_name(n::Node) = get_attribute(n, :name)
get_name(n::Node) = n.name

function Node(root::JLD2.JLDFile)
    attrs = parse_attributes(root, "")
    return Node{typeof(root)}(root, "root", attrs)
end

function Node(parent, node_name::String)
    attrs = parse_attributes(parent, node_name)
    node = parent[node_name]
    return Node{typeof(node)}(parent[node_name], node_name, attrs)
end

Node(parent::Node, node_name::String) = Node(get_wrapped_node(parent), node_name)

function parse_attributes(parent, node_name)
    raw_attrs = JLD2.load_attributes(parent, node_name)
    attrs = Dict{Symbol, String}()
    attrs_keys = (:name, :label, :type)
    for (key, val) in raw_attrs
        if key in attrs_keys
            attrs[key] = first(split(val, "\0"))
        end
    end
    return attrs
end

"""
We could make this function type-stable by converting the "type" attribute to a type-parameter of `Node`
"""
function get_value(n::Node)
    data_type = get_data_type(n)
    data = get_wrapped_node(n)[" data"]
    if data_type == "C1"
        return String(UInt8.(data))
    elseif data_type in ("I4", "R8")
        return data
    else
        error("Datatype '$(data_type)' not handled")
    end
end

function read_zone(zone, topo_dim, space_dim)
    # Preliminary check
    zoneType = get_value(get_child(zone; type = "ZoneType_t"))
    @assert zoneType == "Unstructured" "Only unstructured zone are supported"

    # Number of elements
    nvertices, ncells, nbnd = get_value(zone)

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

    # Read elements
    volumicEltsNodes = filter(
        child -> is_volumic_shape(child, topo_dim),
        get_children(zone; type = "Elements_t"),
    )

    list_c2t_c2n = map(read_connectivity, volumicEltsNodes)

    c2t = mapreduce(x -> x[1], vcat, list_c2t_c2n)
    c2n = mapreduce(x -> x[2], vcat, list_c2t_c2n)

    @show coords
    @show c2t
    @show c2n
end

function get_child(parent; name = "", type = "")
    for child_name in keys(get_wrapped_node(parent))
        child = Node(parent, child_name)
        child_match(child, name, type) && (return child)
    end
end

function get_children(parent; name = "", type = "")
    filtered = filter(
        child_name -> child_match(Node(parent, child_name), name, type),
        keys(get_wrapped_node(parent)),
    )
    map(child_name -> Node(parent, child_name), filtered)
end

function child_match(child, name, type)
    if get_name(child) == name
        if length(name) > 0 && length(type) > 0
            (get_cgns_type(child) == type) && (return true)
        elseif length(name) > 0
            return true
        end
    end

    if get_cgns_type(child) == type
        if length(name) > 0 && length(type) > 0
            (get_name(child) == name) && (return true)
        elseif length(type) > 0
            return true
        end
    end

    return false
end

function is_volumic_shape(node, topo_dim)
    @assert get_cgns_type(node) == "Elements_t"
    code, _ = get_value(node)

    if code in (3,) # 3 = Bar2
        return topo_dim == 1
    elseif code in (7,) # 7 = Quad4
        return topo_dim == 2
    else
        error("Unknown element code : $code")
    end
end

function read_connectivity(node)
    @assert get_cgns_type(node) == "Elements_t"

    # Build cell to (cgns) type
    code, _ = get_value(node)
    erange = get_value(get_child(node; name = "ElementRange"))
    nelts = erange[2] - erange[1] + 1
    c2t = fill(code, nelts)

    # Build cell to node
    c2n = get_value(get_child(node; name = "ElementConnectivity"))

    return c2t, c2n
end

end