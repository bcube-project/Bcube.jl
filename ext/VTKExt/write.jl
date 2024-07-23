function write_file(
    ::Bcube.VTKIoHandler,
    basename::String,
    mesh::AbstractMesh,
    U_export,
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

    # Write !
    write_vtk_lagrange(_basename, _data, mesh, U_export, it, time; collection_append)
end