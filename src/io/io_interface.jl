abstract type AbstractIoHandler end
struct GMSHIoHandler <: AbstractIoHandler end
struct HDF5IoHandler <: AbstractIoHandler end
struct JLD2IoHandler <: AbstractIoHandler end
struct VTKIoHandler <: AbstractIoHandler end

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

# Example
```julia
result = read_file("file.cgns"; varnames = ["Temperature", "Density"], verbose = true)
@show ncells(result.mesh)
@show keys(result.data)
```

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

function read_mesh(filepath::String; kwargs...)
    read_mesh(_filename_to_handler(filepath), filepath; kwargs...)
end

"""
    write_file(
        [handler::AbstractIoHandler,]
        basename::String,
        mesh::AbstractMesh,
        data = nothing,
        it::Integer = -1,
        time::Real = 0.0;
        mesh_degree::Integer = 1,
        functionSpaceType::AbstractFunctionSpaceType = Lagrange(),
        discontinuous::Bool = true,
        collection_append::Bool = false,
        kwargs...,
    )

Write a set of `AbstractLazy` to a file.

`data` can be provided as a `Dict{String, AbstractLazy}` if they share the same
"container" (~FlowSolution), or as a `Dict{String, T}` where T is the previous
Dict type described.

To write cell-centered data, wrapped your input into a `MeshCellData` (for instance
using Bcube.cell_mean).


# Implementation
To specialize this method, please specialize:
write_file(
    handler::AbstractIoHandler,
    basename::String,
    mesh::AbstractMesh,
    U_export::AbstractFESpace,
    data = nothing,
    it::Integer = -1,
    time::Real = 0.0;
    collection_append::Bool = false,
    kwargs...,
)

Possible alternative names:
* write_file
* write_to_file
* write_data
* same with "save" instead of "write"

"""
function write_file(
    handler::AbstractIoHandler,
    basename::String,
    mesh::AbstractMesh,
    data = nothing,
    it::Integer = -1,
    time::Real = 0.0;
    mesh_degree::Integer = 1,
    functionSpaceType::AbstractFunctionSpaceType = Lagrange(),
    discontinuous::Bool = true,
    collection_append::Bool = false,
    kwargs...,
)

    # Build FESpace corresponding to asked degree, func space and discontinuous
    U_export = TrialFESpace(
        FunctionSpace(functionSpaceType, mesh_degree),
        mesh;
        isContinuous = !discontinuous,
    )

    # Call version with U_export
    write_file(
        handler,
        basename,
        mesh,
        U_export,
        data,
        it,
        time;
        collection_append,
        kwargs...,
    )
end

function write_file(
    handler::AbstractIoHandler,
    basename::String,
    mesh::AbstractMesh,
    U_export::AbstractFESpace,
    data = nothing,
    it::Integer = -1,
    time::Real = 0.0;
    collection_append::Bool = false,
    kwargs...,
)
    error("'write_file' is not implemented for $(typeof(handler))")
end

function write_file(basename::String, args...; kwargs...)
    write_file(_filename_to_handler(basename), basename, args...; kwargs...)
end

"""
    check_input_file([handler::AbstractIoHandler,] filepath::String)

Check that the input file is compatible with the Bcube reader, and print warnings and/or errors if it's not.
"""
function check_input_file(handler::AbstractIoHandler, filepath::String) end

check_input_file(filename::String) = check_input_file(_filename_to_handler(filename))

function _filename_to_handler(filename::String)
    ext = last(splitext(filename))
    if ext in (".msh",)
        return GMSHIoHandler()
    elseif ext in (".pvd", ".vtk", ".vtu")
        return VTKIoHandler()
    elseif ext in (".cgns", ".hdf", ".hdf5")
        return HDF5IoHandler()
    end
    error("Could not find a handler for the filename $filename")
end
