function Bcube.write_file(
    ::Bcube.JLD2IoHandler,
    basename::String,
    mesh::Bcube.AbstractMesh,
    vars::Dict{String, F} = Dict{String, Bcube.AbstractLazy}(),
    U_export::Bcube.AbstractFESpace = SingleFESpace(FunctionSpace(:Lagrange, 1), mesh),
    it::Integer = -1,
    time::Real = 0.0;
    collection_append = false,
    kwargs...,
) where {F <: Bcube.AbstractLazy}
    error("not implemented yet")

    mode = collection_append ? "a+" : "w"
    jldopen(basename, mode) do file
    end
end
