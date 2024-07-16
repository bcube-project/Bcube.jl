function write_file(
    ::Bcube.VTKIoHandler,
    basename::String,
    mesh::AbstractMesh,
    data = nothing,
    it::Integer = -1,
    time::Real = 0.0;
    collection_append = false,
    kwargs...,
)

    # Remove extension from filename
    _basename = first(splitext(basename))

    # Just write the mesh if `data` is `nothing`
    if isnothing(data)
        write_vtk(_basename, mesh)
        return
    end

    # We don't use FlowSolution names in VTK, so we flatten everything
    _data = data
    if valtype(data) <: Dict
        _keys = map(d -> collect(keys(d)), values(data))
        _keys = vcat(_keys...)
        _values = map(d -> collect(values(d)), values(data))
        _values = vcat(_values...)
        _data = Dict(_keys .=> _values)
    end

    # First, we check that the export space is the same for all variables
    # -> a bit ugly, but ok for now
    vars = values(_data)
    U_export = first(vars)[2]
    @assert all(x -> x[2] == U_export, vars) "Export FESpace must be identical for all variables"

    # Reshape "data" to remove export space
    _data = Dict(keys(_data) .=> first.(values(_data)))

    # Write !
    write_vtk_lagrange(_basename, _data, mesh, U_export, it, time; collection_append)
end