"""
# Implementation
All subtypes should implement the following functions:
* `Base.parent(AbstractMesh)` (default should be `Base.parent(m::MyMesh) = m`)
"""
abstract type AbstractMesh{topoDim, spaceDim} end

topodim(::AbstractMesh{topoDim, spaceDim}) where {topoDim, spaceDim} = topoDim
spacedim(::AbstractMesh{topoDim, spaceDim}) where {topoDim, spaceDim} = spaceDim

function topodim(::AbstractMesh{topoDim}, label::Symbol) where {topoDim}
    if label == :node || label == :vertex
        return 0
    elseif label == :edge
        return 1
    elseif label == :face
        return topoDim - 1
    elseif label == :cell
        return topoDim
    else
        throw(ArgumentError("Expected label :node,:vertex,:edge,:face or :cell"))
    end
end

@inline boundary_tag(mesh::AbstractMesh, name) = boundary_tag(parent(mesh), name)

abstract type AbstractMeshConnectivity end

"""
Type parameters details:
* `C` is the indices type
* `F` is a symbol standing for "from"
* `T` is a symbol standing for "to"
* `B::Union{Symbol,Nothing}` is a symbol (or Nothing) standing for "by"
* `L::Union{Int,Nothing}` is nLayers type

Dev notes: `from`, `to`, `by` were previously structure properties (of type Symbol) but
were moved to type parameter for GPU compatibility
"""
struct MeshConnectivity{C, F, T, B, L} <: AbstractMeshConnectivity
    nLayers::L
    indices::C
end

function MeshConnectivity(
    from::Symbol,
    to::Symbol,
    by::Union{Symbol, Nothing},
    nLayers::Union{Int, Nothing},
    connectivity::AbstractConnectivity,
)
    MeshConnectivity{typeof(connectivity), from, to, by, typeof(nLayers)}(
        nLayers,
        connectivity,
    )
end
function MeshConnectivity(from::Symbol, to::Symbol, by::Symbol, c::AbstractConnectivity)
    MeshConnectivity(from, to, by, nothing, c)
end
function MeshConnectivity(from::Symbol, to::Symbol, c::AbstractConnectivity)
    MeshConnectivity(from, to, nothing, nothing, c)
end

@inline from(::MeshConnectivity{C, F}) where {C, F} = F
@inline to(::MeshConnectivity{C, F, T}) where {C, F, T} = T
@inline by(::MeshConnectivity{C, F, T, B}) where {C, F, T, B} = B
@inline nlayers(c::MeshConnectivity) = c.nLayers
@inline indices(c::MeshConnectivity) = c.indices

Base.eltype(::MeshConnectivity{C}) where {C <: AbstractConnectivity} = eltype{C}
function Base.iterate(c::MeshConnectivity{C}, i = 1) where {C <: AbstractConnectivity}
    return iterate(indices(c); i = 1)
end

function Base.show(io::IO, c::MeshConnectivity)
    println("--------------------")
    println("MeshConnectivity    ")
    println("type: ", typeof(c))
    println("from,to,by,nlayers = (", from(c), " ", to(c), " ", by(c), " ", nlayers(c), ")")
    println("indices :")
    show(indices(c))
    println("--------------------")
end

"""
Mesh metadata intended to store informations about "zones" (or "domains") originally
stored in the mesh file.

# Devs notes
All subtypes should implement the following functions:
* `get_zone_names(::AbstractMeshMetaData, ::AbstractMesh)`
* `get_zone_element_indices(::AbstractMeshMetaData, ::AbstractMesh, name)`
"""
abstract type AbstractMeshMetaData end

"""
    get_zone_names(::AbstractMeshMetaData)

Return the zone names
"""
get_zone_names(mesh::AbstractMesh) = get_zone_names(get_metadata(mesh), mesh)
get_zone_names(::AbstractMeshMetaData, ::AbstractMesh) = error("not implemented")

"""
    get_zone_indices(::AbstractMesh, name)

Return the cell indices of a given zone, designated by its name
"""
function get_zone_element_indices(mesh::AbstractMesh, name)
    get_zone_element_indices(get_metadata(mesh), mesh, name)
end
function get_zone_element_indices(::AbstractMeshMetaData, ::AbstractMesh, name)
    error("not implemented")
end

struct DefaultMeshMetaData <: AbstractMeshMetaData end
get_zone_names(::DefaultMeshMetaData, ::AbstractMesh) = ("Zone",)
get_zone_element_indices(::DefaultMeshMetaData, mesh::AbstractMesh, name) = 1:ncells(mesh)

"""
`bc_names` : <boundary tag> => <boundary names>
`bc_nodes` : <boundary tag> => <boundary nodes tags>
`bc_faces` : <boundary tag> => <boundary faces tags>
"""
struct Mesh{topoDim, spaceDim, N, E, C, BCn, BCf, M} <: AbstractMesh{topoDim, spaceDim}
    nodes::N
    entities::E
    connectivities::C

    bc_nodes::BCn #Vector{Vector{Int}}
    bc_faces::BCf #Vector{Vector{Int}}

    metadata::M
end

function Mesh(
    topoDim::Int,
    nodes::AbstractVector{<:Node{spaceDim, T}},
    celltypes::Vector{E},
    cell2node::C,
    buildfaces::Bool,
    buildboundaryfaces::Bool,
    bc_nodes::NamedTuple,
    bc_faces::NamedTuple,
    metadata::AbstractMeshMetaData,
) where {T <: Real, spaceDim, E <: AbstractEntityType, C <: AbstractConnectivity}
    @assert topoDim ≤ spaceDim

    # to improve type-stability (through union-splitting if needed)
    celltypes = convert_to_vector_of_union(celltypes)

    nodetypes = Node_t[Node_t() for i in 1:length(nodes)]
    entities = (cell = celltypes, node = nodetypes)

    connectivities = Dict{Symbol, MeshConnectivity}()
    c2n = MeshConnectivity(:cell, :node, cell2node)
    connectivities = (c2n = c2n,)

    if buildfaces
        facetypes, c2f, f2c, f2n = _build_faces!(c2n, celltypes)
        facetypes = convert_to_vector_of_union(facetypes)  # to improve type-stability (through union-splitting if needed)
        connectivities = (connectivities..., c2f = c2f, f2c = f2c, f2n = f2n)
        entities = (entities..., face = facetypes)
        _bc_faces = bc_faces
    end
    if buildboundaryfaces
        @assert buildfaces "Faces must be built before boundary faces"
        f2bc, _bc_faces = _build_boundary_faces!(f2n, f2c, bc_nodes)
        connectivities = (connectivities..., f2bc = f2bc)
    end

    Mesh{
        topoDim,
        spaceDim,
        typeof(nodes),
        typeof(entities),
        typeof(connectivities),
        typeof(bc_nodes),
        typeof(bc_faces),
        typeof(metadata),
    }(
        nodes,
        entities,
        connectivities,
        bc_nodes,
        _bc_faces,
        metadata,
    )
end

function Mesh(
    nodes,
    celltypes::E,
    cell2nodes::C;
    buildfaces::Bool = true,
    buildboundaryfaces::Bool = false,
    bc_names::AbstractDict = Dict{Int, String}(),
    bc_nodes::AbstractDict = Dict{Int, Vector{Int}}(),
    bc_faces::AbstractDict = Dict{Int, Vector{Int}}(),
    metadata::AbstractMeshMetaData = DefaultMeshMetaData(),
) where {E <: AbstractVector{<:AbstractEntityType}, C <: AbstractConnectivity}
    topoDim = topodim(valtype(celltypes))
    if buildboundaryfaces
        @assert (!isempty(bc_names) && !isempty(bc_nodes)) "bc_names and bc_nodes must be provided to build boundary faces"
        @assert isempty(bc_faces) "`bc_faces` is not supported yet"
    else
        (!isempty(bc_names) && !isempty(bc_nodes)) ? buildboundaryfaces = true : nothing
    end

    # Determine integer type. Warning : in the case where both `bc_nodes` and `bc_faces`
    # are empty and are set by default (think of a closed surface for instance), we cannot
    # properly determiner T_int : the user has to input empty dicts of the correct type
    T_int = length(bc_nodes) > 0 ? eltype(valtype(bc_nodes)) : eltype(valtype(bc_faces))

    _names = (Symbol(bc_names[i]) for i in keys(bc_names))
    _nodes = (get(bc_nodes, i, T_int[]) for i in keys(bc_names))
    _faces = (get(bc_faces, i, T_int[]) for i in keys(bc_names))
    nt_bc_nodes = (; zip(_names, _nodes)...)
    nt_bc_faces = (; zip(_names, _faces)...)

    Mesh(
        topoDim,
        nodes,
        celltypes,
        cell2nodes,
        buildfaces,
        buildboundaryfaces,
        nt_bc_nodes,
        nt_bc_faces,
        metadata,
    )
end

Base.parent(mesh::Mesh) = mesh

@inline get_metadata(mesh::Mesh) = mesh.metadata

@inline get_nodes(mesh::Mesh) = mesh.nodes
@inline get_nodes(mesh::Mesh, i::SVector) = mesh.nodes[i]
@inline get_nodes(mesh::Mesh, i) = view(mesh.nodes, i)
@inline get_nodes(mesh::Mesh, i::Int) = mesh.nodes[i]

@inline set_nodes!(mesh::Mesh, nodes) = mesh.nodes .= nodes

@inline entities(mesh::Mesh) = mesh.entities
@inline entities(mesh::Mesh, e::Symbol) = entities(mesh)[e]
@inline has_entities(mesh::Mesh, e::Symbol) = haskey(entities(mesh), e)
@inline has_vertices(mesh::Mesh) = has_entities(mesh, :vertex)
@inline has_nodes(mesh::Mesh) = has_entities(mesh, :node)
@inline has_edges(mesh::Mesh) = has_entities(mesh, :edge)
@inline has_faces(mesh::Mesh) = has_entities(mesh, :face)
@inline has_cells(mesh::Mesh) = has_entities(mesh, :cell)

@inline function n_entities(mesh::Mesh, e::Symbol)
    has_entities(mesh, e) ? length(entities(mesh, e)) : 0
end
@inline nvertices(mesh::Mesh) = n_entities(mesh, :vertex)
@inline nnodes(mesh::Mesh) = n_entities(mesh, :node)
@inline nedges(mesh::Mesh) = n_entities(mesh, :edge)
@inline nfaces(mesh::Mesh) = n_entities(mesh, :face)
@inline ncells(mesh::Mesh) = n_entities(mesh, :cell)

@inline nodes(mesh::Mesh) = entities(mesh, :node)
@inline vertex(mesh::Mesh) = entities(mesh, :vertex)
@inline edges(mesh::Mesh) = entities(mesh, :edges)
@inline faces(mesh::Mesh) = entities(mesh, :face)
@inline cells(mesh::Mesh) = entities(mesh, :cell)

@inline connectivities(mesh::Mesh) = mesh.connectivities
@inline has_connectivities(mesh::Mesh, c::Symbol) = haskey(connectivities(mesh), c)
@inline connectivities(mesh::Mesh, c::Symbol) = connectivities(mesh)[c]
@inline connectivities_indices(mesh::Mesh, c::Symbol) = indices(connectivities(mesh, c))

@inline boundary_names(mesh::Mesh) = keys(mesh.bc_nodes)
@inline boundary_names(mesh::Mesh, tag) = keys(mesh.bc_nodes)[tag]
@inline boundary_nodes(mesh::Mesh) = mesh.bc_nodes
@inline boundary_nodes(mesh::Mesh, tag) = mesh.bc_nodes[tag]
@inline boundary_faces(mesh::Mesh) = mesh.bc_faces
@inline boundary_faces(mesh::Mesh, tag) = mesh.bc_faces[tag]
@inline nboundaries(mesh::Mesh) = length(keys(boundary_names(mesh)))
@inline function boundary_tag(mesh::Mesh, name::Union{String, Symbol})
    findfirst(i -> i == Symbol(name), boundary_names(mesh))
end
@inline function boundary_faces(mesh::Mesh, name::Union{String, Symbol})
    mesh.bc_faces[Symbol(name)]
end

function _build_faces!(c2n::MeshConnectivity, celltypes)
    c2n_indices = indices(c2n)
    T = eltype(c2n_indices)
    #build face->cell connectivity
    dict_faces = Dict{Set{T}, T}()
    c2f_indices = [T[] for i in 1:length(celltypes)]
    f2c_indices = Vector{SVector{2, T}}()
    f2n_indices = Vector{Vector{T}}()
    face_types = Vector{AbstractEntityType}()
    sizehint!(f2c_indices, sum(nfaces, celltypes))
    nface = zero(T)
    for (i, (_c, _c2n)) in enumerate(zip(celltypes, c2n_indices))
        for (j, _f2n) in enumerate(f2n_from_c2n(_c, _c2n))
            if !haskey(dict_faces, Set(_f2n))
                #create a new face
                push!(f2c_indices, SVector{2, T}(T(i)::T, zero(T)))
                push!(face_types, facetypes(_c)[j])
                push!(f2n_indices, _f2n)
                nface += 1
                dict_faces[Set(_f2n)] = nface
                push!(c2f_indices[i], nface)
            else
                iface = get(dict_faces, Set(_f2n), 0)
                f2c_indices[iface] = [f2c_indices[iface][1], i]
                push!(c2f_indices[i], iface)
            end
        end
    end
    c2f = MeshConnectivity(
        :cell,
        :face,
        Connectivity(T[length(a) for a in c2f_indices], reduce(vcat, c2f_indices)),
    )
    numIndices = T[sum(i -> i > zero(eltype(a)) ? 1 : 0, a) for a in f2c_indices]
    _indices = [f2c_indices[i][j] for i in eachindex(f2c_indices) for j in 1:numIndices[i]]
    f2c = MeshConnectivity(:face, :cell, Connectivity(numIndices, _indices))
    f2n = MeshConnectivity(
        :face,
        :node,
        Connectivity(T[length(a) for a in f2n_indices], reduce(vcat, f2n_indices)),
    )

    # try to convert `face_types` to a vector of concrete type when it is possible
    _face_types = identity.(face_types)

    return _face_types, c2f, f2c, f2n
end

function _build_boundary_faces!(f2n::MeshConnectivity, f2c::MeshConnectivity, bc_nodes)
    @assert (bc_nodes ≠ nothing) "bc_names and bc_nodes must be defined"
    @assert length(indices(f2n)) == length(indices(f2c)) "invalid f2n and/or f2c"
    @assert length(indices(f2n)) ≠ 0 "invalid f2n and/or f2c"

    # find boundary faces (all faces with only 1 adjacent cell)
    T_int = eltype(valtype(bc_nodes))
    f2bc_numindices = zeros(T_int, length(indices(f2n)))
    f2bc_indices = Vector{T_int}()
    sizehint!(f2bc_indices, length(f2bc_numindices))

    f2c = indices(f2c)
    for (iface, f2n) in enumerate(indices(f2n))
        if length(f2c[iface]) == 1
            # Search for associated BC (if any)
            # Rq: there might not be any BC, for instance if the face if a ghost-cell face
            for (tag, nodes) in enumerate(bc_nodes)
                if f2n ⊆ nodes
                    push!(f2bc_indices, tag)
                    f2bc_numindices[iface] = 1
                    break
                end
            end
        end
    end

    # add face->bc connectivities in mesh struct
    f2bc = MeshConnectivity(:face, :face, Connectivity(f2bc_numindices, f2bc_indices))

    # create bc_faces
    if length(bc_nodes) > 0
        f2bc_indices = indices(f2bc)
        #bc2f_indices = [[i for (i,bc) in enumerate(indices(f2bc)) if length(bc)>0 && bc[1]==bctag] for bctag in keys(bc_names)]
        #bc_faces = (;zip(Symbol.(keys(bc_names)), bc2f_indices)...)
        bc_faces = Dict{Int, Vector{Int}}()
        bc_faces = (;
            zip(
                keys(bc_nodes),
                [
                    [
                        i for (i, bc) in enumerate(indices(f2bc)) if
                        length(bc) > 0 && bc[1] == bctag
                    ] for bctag in 1:length(bc_nodes)
                ],
            )...,
        )
    end

    return f2bc, bc_faces
end

function build_boundary_faces!(mesh::Mesh)
    @assert (mesh.bc_names ≠ nothing && mesh.bc_nodes ≠ nothing) "bc_names and bc_nodes must be defined"
    @assert (mesh.bc_names ≠ nothing && mesh.bc_nodes ≠ nothing) "bc_names and bc_nodes must be defined"
    @assert has_faces(mesh) "face entities must be created first"
    @assert has_connectivities(mesh, :f2n) "face->node connectivity must be defined"

    # find boundary faces (all faces with only 1 adjacent cell)

    f2bc_numindices = zeros(Int, nfaces(mesh))
    f2bc_indices = Vector{Int}()
    sizehint!(f2bc_indices, length(f2bc_numindices))

    f2c = connectivities_indices(mesh, :f2c)
    for (iface, f2n) in enumerate(connectivities_indices(mesh, :f2n))
        if length(f2c[iface]) == 1
            f2bc_numindices[iface] = 1
            for (tag, nodes) in mesh.bc_nodes
                f2n ⊆ nodes ? (push!(f2bc_indices, tag); break) : nothing
            end
        end
    end

    # add face->bc connectivities in mesh struct
    mesh.connectivities[:f2bc] =
        MeshConnectivity(:face, :face, Connectivity(f2bc_numindices, f2bc_indices))

    # create bc_faces
    if nboundaries(mesh) > 0
        f2bc = connectivities_indices(mesh, :f2bc)
        for bctag in keys(mesh.bc_names)
            mesh.bc_faces[bctag] =
                [i for (i, bc) in enumerate(f2bc) if length(bc) > 0 && bc[1] == bctag]
        end
    end

    return nothing
end

"""
    connectivity_cell2cell_by_faces(mesh)

Build the cell -> cell connectivity by looking at neighbors by faces.
"""
function connectivity_cell2cell_by_faces(mesh)
    f2c = connectivities_indices(mesh, :f2c)
    T = eltype(f2c)
    c2c = [T[] for _ in 1:ncells(mesh)]
    for k in 1:nfaces(mesh)
        _f2c = f2c[k]
        if length(_f2c) > 1
            i = _f2c[1]
            j = _f2c[2]
            push!(c2c[i], j)
            push!(c2c[j], i)
        end
    end
    return Connectivity(length.(c2c), rawcat(c2c))
end

"""
    connectivity_cell2cell_by_nodes(mesh)

Build the cell -> cell connectivity by looking at neighbors by nodes.
"""
function connectivity_cell2cell_by_nodes(mesh)
    c2n = connectivities_indices(mesh, :c2n)
    n2c, _ = inverse_connectivity(c2n)
    T = eltype(n2c)
    c2c = [T[] for _ in 1:ncells(mesh)]

    # Loop over node->cells arrays
    for _n2c in n2c
        # For each cell, append the others as neighbors
        for ci in _n2c, cj in _n2c
            (cj ≠ ci) && push!(c2c[ci], cj)
        end
    end

    # Eliminate duplicates
    c2c = unique.(c2c)

    return Connectivity(length.(c2c), rawcat(c2c))
end

"""
    oriented_cell_side(mesh::Mesh,icell::Int,iface::Int)

Return the side number to which the face 'iface' belongs
in the cell 'icell' of the mesh. A negative side number is
returned if the face is inverted. Returns '0' if the face
does not belongs the cell.
"""
function oriented_cell_side(mesh::Mesh, icell::Int, iface::Int)
    c = cells(mesh)[icell]
    c2n = indices(connectivities(mesh, :c2n))[icell]
    f2n = indices(connectivities(mesh, :f2n))[iface]
    oriented_cell_side(c, c2n, f2n)
end

"""
    inner_faces(mesh)

Return the indices of the inner faces.
"""
function inner_faces(mesh)
    f2c = indices(connectivities(mesh, :f2c))
    return findall([length(f2c[i]) > 1 for i in 1:nfaces(mesh)])
end

"""
    outer_faces(mesh)

Return the indices of the outer (=boundary) faces.
"""
function outer_faces(mesh)
    f2c = indices(connectivities(mesh, :f2c))
    return findall([length(f2c[i]) == 1 for i in 1:nfaces(mesh)])
end
