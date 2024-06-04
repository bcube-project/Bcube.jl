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

    @show keys(file)
    x = file["domainLine_o1.cgns"]
    @show attributes(x)
    @show haskey(attributes(x), "label")
    attr = attributes(x)["label"]
    @show attr[]

    # data = read(x)
    data = x[" data"]
    @show read(data)

    # Find the list of CGNSBase_t
    cgnsBases = get_children(root; type = "CGNSBase_t")
    if length(cgnsBases) == 0
        error("Could not find any CGNSBase_t node in the file")
    elseif length(cgnsBases) > 1
        error("The file contains several CGNSBase_t nodes, only one base is supported")
    end
    cgnsBase = first(cgnsBases)
    @show cgnsBase

    # Read base dimensions (topo and space)
    dims = get_value(cgnsBase)
    topodim = topodim > 0 ? topodim : dims[1]
    spacedim = spacedim > 0 ? spacedim : dims[2]
    verbose && println("topodim = $topodim, spacedim = $spacedim")

    # Close the file
    close(file)
end

function get_child(parent; name = "", type = "")
    for child_name in keys(parent)
        child = parent[child_name]
        child_match(child, name, type) && (return child)
    end
end

function get_children(parent; name = "", type = "")
    filtered =
        filter(child_name -> child_match(parent[child_name], name, type), keys(parent))
    map(child_name -> parent[child_name], filtered)
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

get_name(obj) = HDF5.name(obj)

get_cgns_type(obj) = haskey(attributes(obj), "label") ? attributes(obj)["label"][] : nothing

get_value(obj) = read(obj[" data"])