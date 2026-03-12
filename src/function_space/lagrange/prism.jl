#_scalar_shape_functions
# Lagrange prismatic elements are built from the cartesian product of the Triangle and the Line
function _scalar_shape_functions(fs::FunctionSpace{<:Lagrange, N}, ::Prism, ξηζ) where {N}
    ξ, η, ζ = ξηζ
    λ_tri = _scalar_shape_functions(fs, Triangle(), SA[ξ, η])
    λ_line = _scalar_shape_functions(fs, Line(), SA[ζ])
    return SVector{length(λ_tri) * length(λ_line)}(λt * λl for λt in λ_tri, λl in λ_line)
end

# get_ndofs
function get_ndofs(fs::FunctionSpace{<:Lagrange, N}, ::Prism) where {N}
    get_ndofs(fs, Triangle()) * get_ndofs(fs, Line())
end

# idof_by_vertex
function idof_by_vertex(::FunctionSpace{<:Lagrange, 0}, shape::Prism)
    ntuple(i -> SA[], nvertices(shape))
end

function idof_by_vertex(::FunctionSpace{<:Lagrange, degree}, shape::Prism) where {degree}
    ntuple(i -> SA[i], nvertices(shape))
end

# idof_by_edge
function idof_by_edge(::FunctionSpace{<:Lagrange, 0}, shape::Prism)
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_edge(fs::FunctionSpace{<:Lagrange, N}, ::Prism) where {N}
    idof_edge_tri = idof_by_edge(fs, Triangle())
    ndofs_tri = get_ndofs(fs, Triangle())
    ndofs_line = get_ndofs(fs, Line())

    return (
        idof_edge_tri[1],
        idof_edge_tri[2],
        idof_edge_tri[3],
        SVector{ndofs_line - 2}(1 + (i - 1) * ndofs_tri for i in 2:(ndofs_line - 1)),
        SVector{ndofs_line - 2}(2 + (i - 1) * ndofs_tri for i in 2:(ndofs_line - 1)),
        SVector{ndofs_line - 2}(3 + (i - 1) * ndofs_tri for i in 2:(ndofs_line - 1)),
        idof_edge_tri[1] .+ (ndofs_line - 1) * ndofs_tri,
        idof_edge_tri[2] .+ (ndofs_line - 1) * ndofs_tri,
        idof_edge_tri[3] .+ (ndofs_line - 1) * ndofs_tri,
    )
end

# idof_by_edge_with_bounds
function idof_by_edge_with_bounds(::FunctionSpace{<:Lagrange, 0}, shape::Prism)
    ntuple(i -> SA[], nedges(shape))
end

function idof_by_edge_with_bounds(fs::FunctionSpace{<:Lagrange, N}, ::Prism) where {N}
    idof_edge_tri = idof_by_edge_with_bounds(fs, Triangle())
    ndofs_tri = get_ndofs(fs, Triangle())
    ndofs_line = get_ndofs(fs, Line())

    return (
        idof_edge_tri[1],
        idof_edge_tri[2],
        idof_edge_tri[3],
        SVector{ndofs_line}(1 + (i - 1) * ndofs_tri for i in 1:ndofs_line),
        SVector{ndofs_line}(2 + (i - 1) * ndofs_tri for i in 1:ndofs_line),
        SVector{ndofs_line}(3 + (i - 1) * ndofs_tri for i in 1:ndofs_line),
        idof_edge_tri[1] .+ (ndofs_line - 1) * ndofs_tri,
        idof_edge_tri[2] .+ (ndofs_line - 1) * ndofs_tri,
        idof_edge_tri[3] .+ (ndofs_line - 1) * ndofs_tri,
    )
end

# idof_by_face
function idof_by_face(::FunctionSpace{<:Lagrange, 0}, shape::Prism)
    ntuple(i -> SA[], nfaces(shape))
end

# Note : recall that the Prism is obtained by cartesian product between Triangle and Line.
# Every inner dof on the "edge" of a Triangle becomes a dof of a side-face of the Prism, excluding the
# one lying on the bottom and top edges.
# WARNING : only Triangle WITHOUT INSIDE NODE are supported
# TODO : use `idof_by_volume`
function idof_by_face(fs::FunctionSpace{<:Lagrange, N}, ::Prism) where {N}
    idof_edge_tri = idof_by_edge(fs, Triangle())
    ndofs_tri = get_ndofs(fs, Triangle())
    ndofs_line = get_ndofs(fs, Line()) # >=3 (ensured by multi-dispatch on N)
    ndofs_line_inner = ndofs_line - 2 # exclude bottom and top edges

    #  Re "(i+1)" because "i" starts at "1" whereas the first element is "2"
    # Ideally, we would write vcat(ntuple(i -> idof_edge_tri[1] .+ (i - 1) * ndofs_tri, 2:ndofs_line_inner-1)...)
    return (
        vcat(ntuple(i -> idof_edge_tri[1] .+ (i + 1 - 1) * ndofs_tri, ndofs_line_inner)...),
        vcat(ntuple(i -> idof_edge_tri[2] .+ (i + 1 - 1) * ndofs_tri, ndofs_line_inner)...),
        vcat(ntuple(i -> idof_edge_tri[3] .+ (i + 1 - 1) * ndofs_tri, ndofs_line_inner)...),
        SA[], # bottom face (z=zmin)
        SA[], # top face (z=zmax)
    )
end

# idof_by_face_with_bounds
function idof_by_face_with_bounds(::FunctionSpace{<:Lagrange, 0}, shape::Prism)
    ntuple(i -> SA[], nfaces(shape))
end

# Note : recall that the Prism is obtained by cartesian product between Triangle and Line.
# Every dof on the "edge" of a Triangle becomes a dof of a side-face of the Prism. Additionnaly,
# every dof of the Triangle becomes a dof of the bottom and top faces of the Prism.
# TODO : use `idof_by_volume`
function idof_by_face_with_bounds(fs::FunctionSpace{<:Lagrange, N}, ::Prism) where {N}
    idof_edge_tri = idof_by_edge_with_bounds(fs, Triangle())
    ndofs_tri = get_ndofs(fs, Triangle())
    ndofs_line = get_ndofs(fs, Line())

    return (
        vcat(ntuple(i -> idof_edge_tri[1] .+ (i - 1) * ndofs_tri, ndofs_line)...),
        vcat(ntuple(i -> idof_edge_tri[2] .+ (i - 1) * ndofs_tri, ndofs_line)...),
        vcat(ntuple(i -> idof_edge_tri[3] .+ (i - 1) * ndofs_tri, ndofs_line)...),
        SVector{ndofs_tri}(i for i in 1:ndofs_tri), # bottom face (z=zmin)
        SVector{ndofs_tri}(
            i for i in ((ndofs_tri * (ndofs_line - 1)) + 1):(ndofs_tri * ndofs_line)
        ), # top face (z=zmax)
    )
end

# get_coords
get_coords(::FunctionSpace{<:Lagrange, 0}, shape::Prism) = (center(shape),)

# Rq: this implementation triggers a small allocation due to the call to
# `get_coords(fs, Bcube.Triangle())`. The rest of the function does not
# allocate
function get_coords(fs::FunctionSpace{<:Bcube.Lagrange}, ::Bcube.Prism)
    x_tri = get_coords(fs, Bcube.Triangle())
    x_line = get_coords(fs, Bcube.Line())

    return mapreduce(
        xl -> map(xt -> SA[xt..., xl], x_tri),
        (a, b) -> (a..., b...),
        first.(x_line),
    )
end