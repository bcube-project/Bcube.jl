abstract type AbstractIoHandler end
struct GMSHIoHandler <: AbstractIoHandler end
struct HDF5IoHandler <: AbstractIoHandler end
struct JLD2IoHandler <: AbstractIoHandler end

"""
    read_file([handler::AbstractIoHandler,] filepath::String; domainNames = nothing, varnames = nothing, topodim = 0, kwargs...,)

Read the mesh and associated data in the given file.

Returns a NamedTuple with the following keys:
* mesh -> the Bcube mesh
* data -> dictionnary of FlowSolutionName => (dictionnary of VariableName => MeshData)
* to be defined : stuff related to subdomains

If `domainNames` is an empty list/array, all the domains found will be read and merged. Otherwise, `domainNames` can be
a filtered list/array of the domain names to retain.

If `varnames` is set to `nothing`, no variables will be read, which is the behavior of `read_mesh`. To read all the
variables, `varnames` must be set to `"*"`.

The argument `topodim` can be used to force and/or select the elements of this topological dimension to be interpreted as
"volumic". Leave it to `0` to let the reader determines the topological dimension automatically. The same goes for `spacedim`.

# Dev
Possible names:
* read_file
* read_from_file
* same with "load" instead of "read"

Questions:
"""
function read_file(
    handler::AbstractIoHandler,
    filepath::String;
    domainNames = String[],
    varnames = nothing,
    topodim = 0,
    spacedim = 0,
    kwargs...,
)
    error(
        "'read_file' is not implemented for $(typeof(handler)). Have you loaded the extension?",
    )
end

function read_file(filepath::String; domainNames = String[], varnames = nothing, kwargs...)
    read_file(_filename_to_handler(filepath), filepath; domainNames, varnames, kwargs...)
end

"""
Similar as `read_file`, but return only the mesh.
"""
function read_mesh(
    handler::AbstractIoHandler,
    filepath::String,
    domainNames = String[],
    kwargs...,
)
    res = read_file(handler, filepath; domainNames, kwargs...)
    return res.mesh
end

function read_mesh(filepath::String; domainNames = String[], kwargs...)
    read_mesh(_filename_to_handler(filepath), filepath; domainNames, kwargs...)
end

"""
Possible names:
* write_file
* write_to_file
* write_data
* same with "save" instead of "write"

Questions:
* should we define a defaut value for `U_export` ?
"""
function write_file(
    handler::AbstractIoHandler,
    basename::String,
    mesh::AbstractMesh,
    vars::Dict{String, F} = Dict{String, AbstractLazy}(),
    U_export::AbstractFESpace = SingleFESpace(FunctionSpace(:Lagrange, 1), mesh),
    it::Integer = -1,
    time::Real = 0.0;
    collection_append = false,
    kwargs...,
) where {F <: AbstractLazy}
    error("'write_file' is not implemented for $(typeof(handler))")
end

function write_file(
    basename::String,
    mesh::AbstractMesh,
    vars::Dict{String, F} = Dict{String, AbstractLazy}(),
    U_export::AbstractFESpace = SingleFESpace(FunctionSpace(:Lagrange, 1), mesh),
    it::Integer = -1,
    time::Real = 0.0;
    collection_append = false,
    kwargs...,
) where {F <: AbstractLazy}
    write_file(
        _filename_to_handler(basename),
        basename,
        mesh,
        vars,
        U_export;
        it,
        time,
        collection_append,
        kwargs...,
    )
end

"""
    check_input_file([handler::AbstractIoHandler,] filepath::String)

Check that the input file is compatible with the Bcube reader, and print warnings and/or errors if it's not.
"""
function check_input_file(handler::AbstractIoHandler, filepath::String) end

check_input_file(filename::String) = check_input_file(_filename_to_handler(filename))

function _filename_to_handler(filename::String)
    ext = last(splitext(filename))
    if ext in [".msh"]
        return GMSHIoHandler()
    elseif ext in [".cgns", ".hdf", ".hdf5"]
        return HDF5IoHandler()
    end
    error("Could not find a handler for the filename $filename")
end