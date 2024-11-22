# This file is only for debug purpose, its content is not intended to be used for production

struct Gmsh22IoHandler <: AbstractIoHandler end

_filename_to_handler(::Val{:msh22}) = GmshIoHandler()

function read_mesh(::Gmsh22IoHandler, filepath::String, domainNames = String[], kwargs...)
    @assert length(domainNames) == 0

    io = open(filepath)

    # MeshFormat
    read_check(io, "\$MeshFormat")

    line = readline(io)
    format = first(split(line, ' '))
    @assert format == "2.2"

    read_check(io, "\$EndMeshFormat")

    # Physical Names
    read_check(io, "\$PhysicalNames")
    n = parse(Int, readline(io))
    lines = [split(readline(io), ' ') for i in 1:n]
    physical_names = Dict(
        parse(Int, line[2]) => (parse(Int, line[1]), replace(line[3], "\"" => "")) for
        line in lines
    ) # tag -> (dim, name)
    read_check(io, "\$EndPhysicalNames")

    # Nodes
    read_check(io, "\$Nodes")
    n = parse(Int, readline(io))
    xyz = zeros(n, 3)
    nodes_indices = zeros(Int, n)
    for i in 1:n
        line = readline(io)
        elts = split(line, ' ')
        nodes_indices[i] = parse(Int, first(elts))
        xyz[i, :] .= parse.(Float64, elts[2, :])
    end
    read_check(io, "\$EndNodes")

    # Elements
    read_check(io, "\$Elements")
    n = parse(Int, readline(io))
    c2n = []
    elts_indices = zeros(Int, n)
    elts_types = zeros(Int, n)
    elts_tags = []
    for i in 1:n
        line = readline(io)
        elts = split(line, ' ')
        elts_indices[i] = parse(Int, elts[1])
        elts_types[i] = parse(Int, elts[2])
        ntags = parse(Int, elts[3])
        if ntags > 0
            push!(elts_tags, parse.(Int, elts[4:(4 + ntags - 1)]))
        end
        push!(c2n, parse.(Int, elts[(4 + ntags):end]))
    end
    read_check(io, "\$EndElements")

    close(io)

    # Now, build Bcube mesh from the collected infos

    # topodim and spacedim
    topodim = maximum(((k, v),) -> first(v), physical_names)
    extr = extrema.(eachcol(xyz))
    lx = extr[1][2] - extr[1][1]
    ly = extr[2][2] - extr[2][1]
    lz = extr[3][2] - extr[3][1]
    spacedim = _compute_space_dim(topodim, lx, ly, lz, 1e-15, true)

    # nodes
    nodes = [Node(_xyz[1:spacedim]) for _xyz in eachrow(xyz)]
end

const GMSHTYPE = Dict(
    1 => Bar2_t(),
    2 => Tri3_t(),
    3 => Quad4_t(),
    4 => Tetra4_t(),
    5 => Hexa8_t(),
    6 => Penta6_t(),
    7 => Pyra5_t(),
    9 => Tri6_t(),
    10 => Quad9_t(),
    21 => Tri9_t(),
    21 => Tri10_t(),
    36 => Quad16_t(),
)

"""
    nodes_gmsh2cgns(entity::AbstractEntityType, nodes::AbstractArray)

Reorder `nodes` of a given `entity` from the Gmsh format to CGNS format.

See https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering

# Implementation
By default, same numbering between CGNS and Gmsh is applied. Specialized the function
`nodes_gmsh2cgns(e::Type{<:T}) where {T <: Bcube.AbstractEntityType}` to secify a
different numbering
"""
function nodes_gmsh2cgns(entity::Bcube.AbstractEntityType, nodes::AbstractArray)
    map(i -> nodes[i], nodes_gmsh2cgns(entity))
end

nodes_gmsh2cgns(e) = nodes_gmsh2cgns(typeof(e))

function nodes_gmsh2cgns(e::Type{<:T}) where {T <: Bcube.AbstractEntityType}
    nodes(e)
end
function nodes_gmsh2cgns(::Type{Bcube.Hexa27_t})
    SA[
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        17,
        18,
        19,
        20,
        13,
        14,
        15,
        16,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
    ]
end

"""
Convert a cell->node connectivity with gmsh numbering convention to a cell->node connectivity
with CGNs numbering convention.
"""
function _c2n_gmsh2cgns(celltypes, c2n_gmsh)
    n = Int[]
    indices = Int[]
    for (ct, c2nᵢ) in zip(celltypes, c2n_gmsh)
        append!(n, length(c2nᵢ))
        append!(indices, nodes_gmsh2cgns(ct, c2nᵢ))
    end
    return Connectivity(n, indices)
end

function read_check(io, ref)
    line = readline(io)
    @assert line == ref
end