abstract type AbstractIoHandler end
struct HDF5IoHandler <: AbstractIoHandler end # to be removed
struct JLD2IoHandler <: AbstractIoHandler end # to be removed

"""
    read_file(
        [handler::AbstractIoHandler,]
        filepath::String;
        domainNames = String[],
        varnames = nothing,
        topodim = 0,
        spacedim = 0,
        kwargs...,
    )

Read the mesh and associated data in the given file.

Returns a NamedTuple with the following keys:
* mesh -> the Bcube mesh
* data -> dictionnary of FlowSolutionName => (dictionnary of VariableName => MeshData)

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
        filepath::String,
        mesh::AbstractMesh,
        data = nothing,
        it::Integer = -1,
        time::Real = 0.0;
        mesh_degree::Integer = 1,
        functionSpaceType::AbstractFunctionSpaceType = Lagrange(),
        discontinuous::Bool = false,
        collection_append::Bool = false,
        kwargs...,
    )

Write a set of `AbstractLazy` to a file.

`data` can be provided as a `Dict{String, AbstractLazy}` if they share the same
"container" (~FlowSolution), or as a `Dict{String, T}` where T is the previous
Dict type described.

To write cell-centered data, wrapped your input into a `MeshCellData` (for instance
using `cell_mean` or `MeshCellData ∘ var_on_centers`).

# Example
```julia
mesh = rectangle_mesh(6, 7; xmin = -1, xmax = 1.0, ymin = -1, ymax = 1.0)
f_u = PhysicalFunction(x -> x[1]^2 + x[2]^2)
u = FEFunction(TrialFESpace(FunctionSpace(:Lagrange, 4), mesh))
projection_l2!(u, f_u, mesh)

vars = Dict("f_u" => f_u, "u" => u, "grad_u" => ∇(u))

for mesh_degree in 1:5
    write_file(
        joinpath(@__DIR__, "output"),
        mesh,
        vars;
        mesh_degree,
        discontinuous = false
    )
end
```

# Remarks:
* If `mesh` is of degree `d`, the solution will be written on a mesh of degree `mesh_degree`, even if this
number is different from `d`.
* The degree of the input FEFunction (P1, P2, P3, ...) is not used to define the nodes where the solution is
written, only `mesh` and `mesh_degree` matter. The FEFunction is simply evaluated on the aforementionned nodes.

# Dev notes
To specialize this method, please specialize:
```julia
write_file(
    handler::AbstractIoHandler,
    filepath::String,
    mesh::AbstractMesh,
    U_export::AbstractFESpace,
    data = nothing,
    it::Integer = -1,
    time::Real = 0.0;
    collection_append::Bool = false,
    kwargs...,
)
```
"""
function write_file(
    handler::AbstractIoHandler,
    filepath::String,
    mesh::AbstractMesh,
    data = nothing,
    it::Integer = -1,
    time::Real = 0.0;
    mesh_degree::Integer = 1,
    functionSpaceType::AbstractFunctionSpaceType = Lagrange(),
    discontinuous::Bool = false,
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
        filepath,
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
    filepath::String,
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

function write_file(filepath::String, args...; kwargs...)
    write_file(_filename_to_handler(filepath), filepath, args...; kwargs...)
end

"""
    check_input_file([handler::AbstractIoHandler,] filepath::String)

Check that the input file is compatible with the Bcube reader, and print warnings and/or errors if it's not.
"""
function check_input_file(handler::AbstractIoHandler, filepath::String) end

check_input_file(filename::String) = check_input_file(_filename_to_handler(filename))

function _filename_to_handler(filename::String)
    ext = last(splitext(filename))
    _filename_to_handler(Val(Symbol(ext[2:end]))) # remove the "dot"
end

function _filename_to_handler(extension)
    error("Could not find a handler for the extension $extension")
end

# to be removed :
_filename_to_handler(::Union{Val{:cgns}, Val{:hdf}, Val{:hdf5}}) = HDF5IoHandler()
