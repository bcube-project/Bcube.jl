"""
An `AbstractDomain` designates any set of entities from a mesh. For instance a set of
cells, a set of faces etc.
"""
abstract type AbstractDomain{M <: AbstractMesh} end

@inline get_mesh(domain::AbstractDomain) = domain.mesh
@inline indices(domain::AbstractDomain) = domain.indices # with ghislainb's help, `AbstractDomain` could be parametered by `I`
@inline topodim(::AbstractDomain) = error("undefined")
@inline codimension(d::AbstractDomain) = topodim(get_mesh(d)) - topodim(d)
LazyOperators.pretty_name(domain::AbstractDomain) = "AbstractDomain"

abstract type AbstractCellDomain{M, I} <: AbstractDomain{M} end

"""
A `CellDomain` is a representation of the cells of a mesh. It's primary
purpose is to represent a domain to integrate over.

# Examples
```julia-repl
julia> mesh = rectangle_mesh(10, 10)
julia> Ω_all = CellDomain(mesh)
julia> selectedCells = [1,3,5,6]
julia> Ω_selected = CellDomain(mesh, selectedCells)
```
"""
struct CellDomain{M, I} <: AbstractCellDomain{M, I}
    mesh::M
    indices::I
    CellDomain(mesh, indices) = new{typeof(mesh), typeof(indices)}(mesh, indices)
end
@inline topodim(d::CellDomain) = topodim(get_mesh(d))
CellDomain(mesh::AbstractMesh) = CellDomain(parent(mesh))
CellDomain(mesh::Mesh) = CellDomain(mesh, 1:ncells(mesh))
LazyOperators.pretty_name(domain::CellDomain) = "CellDomain"

abstract type AbstractFaceDomain{M} <: AbstractDomain{M} end

struct InteriorFaceDomain{M, I} <: AbstractFaceDomain{M}
    mesh::M
    indices::I
    InteriorFaceDomain(mesh, indices) = new{typeof(mesh), typeof(indices)}(mesh, indices)
end
@inline topodim(d::InteriorFaceDomain) = topodim(get_mesh(d)) - 1
InteriorFaceDomain(mesh::AbstractMesh) = InteriorFaceDomain(parent(mesh))
InteriorFaceDomain(mesh::Mesh) = InteriorFaceDomain(mesh, inner_faces(mesh))
LazyOperators.pretty_name(domain::InteriorFaceDomain) = "InteriorFaceDomain"

struct AllFaceDomain{M, I} <: AbstractFaceDomain{M}
    mesh::M
    indices::I
    AllFaceDomain(mesh, indices) = new{typeof(mesh), typeof(indices)}(mesh, indices)
end
@inline topodim(d::AllFaceDomain) = topodim(get_mesh(d)) - 1
AllFaceDomain(mesh::AbstractMesh) = AllFaceDomain(mesh, 1:nfaces(mesh))
LazyOperators.pretty_name(domain::AllFaceDomain) = "AllFaceDomain"

struct BoundaryFaceDomain{M, BC, L, C} <: AbstractFaceDomain{M}
    mesh::M
    bc::BC
    labels::L
    cache::C
end
@inline get_mesh(d::BoundaryFaceDomain) = d.mesh
@inline topodim(d::BoundaryFaceDomain) = topodim(get_mesh(d)) - 1
@inline bctype(::BoundaryFaceDomain{M, BC}) where {M, BC} = BC
@inline get_bc(d::BoundaryFaceDomain) = d.bc
@inline get_cache(d::BoundaryFaceDomain) = d.cache
LazyOperators.pretty_name(domain::BoundaryFaceDomain) = "BoundaryFaceDomain"

indices(d::BoundaryFaceDomain) = get_cache(d)

function BoundaryFaceDomain(mesh::Mesh, bc::PeriodicBCType)
    cache =
        _compute_periodicity(mesh, labels_master(bc), labels_slave(bc), transformation(bc))
    labels = unique(vcat(labels_master(bc)..., labels_slave(bc)...))
    BoundaryFaceDomain{typeof(mesh), typeof(bc), typeof(labels), typeof(cache)}(
        mesh,
        bc,
        labels,
        cache,
    )
end

function indices(d::BoundaryFaceDomain{M, <:PeriodicBCType}) where {M}
    _, _, _, _, bnd_ftypes, = get_cache(d)
    return 1:length(bnd_ftypes)
end

"""
    Find periodic face connectivities sush as :
    (faces of `labels2`) = A(faces of `labels1`)
"""
function _compute_periodicity(mesh, labels1, labels2, A, tol = 1e-9)

    # Get cell -> node connectivity
    c2n = connectivities_indices(mesh, :c2n)

    # Get face -> node connectivity
    f2n = connectivities_indices(mesh, :f2n)

    # Get face -> cell connectivity
    f2c = connectivities_indices(mesh, :f2c)

    # Cell and face types
    #celltypes = cells(mesh)

    tags1 = map(label -> boundary_tag(mesh, label), labels1)
    tags2 = map(label -> boundary_tag(mesh, label), labels2)

    # Boundary faces
    bndfaces1 = vcat(map(tag -> boundary_faces(mesh, tag), tags1)...)
    bndfaces2 = vcat(map(tag -> boundary_faces(mesh, tag), tags2)...)
    #nbnd1 = length(bndfaces1)
    nbnd2 = length(bndfaces2)

    # Allocate result
    bnd_f2n1 = [zero(f2n[iface]) for iface in bndfaces1]
    bnd_f2n2 = [zero(f2n[iface]) for iface in bndfaces2]
    bnd_f2c = zeros(Int, nbnd2, 2) # Adjacent cells for each bnd face
    bnd_ftypes = Array{AbstractEntityType}(undef, nbnd2)

    # Loop over bnd faces
    for (i, iface) in enumerate(bndfaces2)
        icell = f2c[iface][1]
        fnodesᵢ = get_nodes(mesh, f2n[iface])
        #sideᵢ = cell_side(celltypes[icell], c2n[icell], f2n[iface])
        Mᵢ = center(fnodesᵢ)
        # compute a characteristic length :
        Δxᵢ = distance(center(get_nodes(mesh, c2n[icell])), Mᵢ)
        isfind = false

        for (j, jface) in enumerate(bndfaces1)
            jcell = f2c[jface][1]
            fnodesⱼ = get_nodes(mesh, f2n[jface])
            #sideⱼ = cell_side(celltypes[jcell], c2n[jcell], f2n[jface])
            Mⱼ = center(fnodesⱼ)

            # compute image point
            Mⱼ_bis = Node(A(get_coords(Mⱼ)))

            # Centers must be identical
            if isapprox(Mᵢ, Mⱼ_bis; atol = tol * Δxᵢ)
                bnd_f2n1[i] = f2n[iface]
                bnd_f2n2[i] = f2n[jface]

                bnd_ftypes[i] = faces(mesh)[iface]

                bnd_f2c[i, 1] = icell
                bnd_f2c[i, 2] = jcell

                # Stop looking for face in relation
                isfind = true
                break
            end
        end
        if isfind === false
            error("Face i=", i, " ", iface, " not found")
        end
    end

    bnd_f2f = Dict{Int, Int}()
    for (i, iface) in enumerate(bndfaces2)
        bnd_f2f[iface] = i
    end

    # node to node relation
    bnd_n2n = Dict{Int, Int}()
    for (_f2c, _f2n1, _f2n2) in zip(bnd_f2c, bnd_f2n1, bnd_f2n2)
        # distance between face center and adjacent cell center
        Δx = distance(center(get_nodes(mesh, c2n[_f2c[1]])), center(get_nodes(mesh, _f2n1)))
        isfind = false
        for i in _f2n1
            Mᵢ = get_nodes(mesh, i)
            for j in _f2n2
                Mⱼ = get_nodes(mesh, j)
                # compute image point
                Mⱼ_bis = Node(A(get_coords(Mⱼ)))
                if isapprox(Mᵢ, Mⱼ_bis; atol = tol * Δx)
                    bnd_n2n[i] = j
                    bnd_n2n[j] = i
                    # Stop looking for face in relation
                    isfind = true
                    break
                end
            end
            if isfind === false
                error("Node i=", i, " not found (2)")
            end
        end
    end

    # Retain only relevant faces
    return bnd_f2f, bnd_f2n1, bnd_f2n2, bnd_f2c, bnd_ftypes, bnd_n2n
end

"""
    BoundaryFaceDomain(mesh)
    BoundaryFaceDomain(mesh, label::String)
    BoundaryFaceDomain(mesh, labels::Tuple{String, Vararg{String}})

Build a `BoundaryFaceDomain` corresponding to the boundaries designated by one
or several labels (=names).

If no label is provided, all the `BoundaryFaceDomain` corresponds
to all the boundary faces.
"""
function BoundaryFaceDomain(mesh::Mesh, labels::Tuple{String, Vararg{String}})
    tags = map(label -> boundary_tag(mesh, label), labels)

    # Check all tags have been found
    for tag in tags
        (tag isa Nothing) && error(
            "Failed to build a `BoundaryFaceDomain` with labels '$(labels...)' -> double-check that your mesh contains these labels",
        )
    end

    bndfaces = vcat(map(tag -> boundary_faces(mesh, tag), tags)...)
    cache = bndfaces
    bc = nothing
    BoundaryFaceDomain{typeof(mesh), typeof(bc), typeof(labels), typeof(cache)}(
        mesh,
        bc,
        labels,
        cache,
    )
end
BoundaryFaceDomain(mesh::AbstractMesh, label::String) = BoundaryFaceDomain(mesh, (label,))
function BoundaryFaceDomain(mesh::AbstractMesh)
    BoundaryFaceDomain(mesh, Tuple(values(boundary_names(mesh))))
end
function BoundaryFaceDomain(mesh::AbstractMesh, args...; kwargs...)
    BoundaryFaceDomain(parent(mesh), args...; kwargs...)
end

"""
    abstract type AbstractDomainIndex

# Devs notes
All subtypes should implement the following functions:
- get_element_type(::AbstractDomainIndex)
"""
abstract type AbstractDomainIndex end

abstract type AbstractCellInfo <: AbstractDomainIndex end

struct CellInfo{C, N, CN} <: AbstractCellInfo
    icell::Int
    ctype::C
    nodes::N # List/array of the cell `Node`s
    c2n::CN # Global indices of the nodes composing the cell
end
function CellInfo(icell, ctype, nodes, c2n)
    CellInfo{typeof(ctype), typeof(nodes), typeof(c2n)}(icell, ctype, nodes, c2n)
end
@inline cellindex(c::CellInfo) = c.icell
@inline celltype(c::CellInfo) = c.ctype
@inline nodes(c::CellInfo) = c.nodes
@inline get_nodes_index(c::CellInfo) = c.c2n
get_element_type(c::CellInfo) = celltype(c)

""" Legacy constructor for CellInfo : no information about node indices """
CellInfo(icell, ctype, nodes) = CellInfo(icell, ctype, nodes, nothing)

"""
    CellInfo(mesh, icell)

DEBUG constructor for `icell`-th cell of `mesh`. For performance issues, don't use this version
in production.
"""
function CellInfo(mesh, icell)
    c2n = connectivities_indices(mesh, :c2n)
    _c2n = c2n[icell]
    celltypes = cells(mesh)

    return CellInfo(icell, celltypes[icell], get_nodes(mesh, _c2n), _c2n)
end

abstract type AbstractCellSide <: AbstractDomainIndex end

struct CellSide{C, N, CN} <: AbstractCellSide
    icell::Int
    iside::Int
    ctype::C
    nodes::N
    c2n::CN
end
function CellSide(icell, iside, ctype, nodes, c2n)
    CellSide{typeof(ctype), typeof(nodes), typeof(c2n)}(icell, iside, ctype, nodes, c2n)
end
@inline cellindex(c::CellSide) = c.icell
@inline cellside(c::CellSide) = c.iside
@inline celltype(c::CellSide) = c.ctype
@inline nodes(c::CellSide) = c.nodes
@inline cell2nodes(c::CellSide) = c.c2n
get_element_type(c::CellSide) = celltype(c)

abstract type AbstractFaceInfo <: AbstractDomainIndex end

"""
    FaceInfo{CN<:CellInfo,CP<:CellInfo,FT,FN,F2N,I}

Type describing a face as the common side of two adjacent cells.
`CellInfo` of cells from both sides is stored with
the local side index of the face relative to each adjacent cell.

`iface` is the mesh-face-index (and not the domain-face-index).

# Remark:
- For boundary face with no periodic condition, positive cell side info
are duplicate from the negative ones.
- For performance reason (type stability), nodes and type of the face
is stored explicitely in `FaceInfo` even if it could have been
computed by collecting info from the side of the negative or positive cells.
"""
struct FaceInfo{CN <: CellInfo, CP <: CellInfo, FT, FN, F2N, I} <: AbstractFaceInfo
    cellinfo_n::CN
    cellinfo_p::CP
    cellside_n::Int
    cellside_p::Int
    faceType::FT
    faceNodes::FN
    f2n::F2N
    iface::I
end

"""
FaceInfo constructor

Cell sides are computed automatically.
"""
function FaceInfo(
    cellinfo_n::CellInfo,
    cellinfo_p::CellInfo,
    faceType,
    faceNodes,
    f2n::AbstractVector,
    iface,
)
    cellside_n = cell_side(celltype(cellinfo_n), get_nodes_index(cellinfo_n), f2n)
    cellside_p = cell_side(celltype(cellinfo_p), get_nodes_index(cellinfo_p), f2n)
    if cellside_n === nothing || cellside_p === nothing
        printstyled("\n=== ERROR in FaceInfo ===\n"; color = :red)
        @show cellside_n
        @show cellside_p
        @show celltype(cellinfo_n)
        @show get_nodes_index(cellinfo_n)
        @show celltype(cellinfo_p)
        @show get_nodes_index(cellinfo_p)
        @show faceType
        @show faceNodes
        @show f2n
        error("Invalid cellside in `FaceInfo`")
    end
    FaceInfo(
        cellinfo_n,
        cellinfo_p,
        cellside_n,
        cellside_p,
        faceType,
        faceNodes,
        f2n,
        iface,
    )
end

"""
DEBUG `FaceInfo` constructor for `kface`-th cell of `mesh`. For performance issues, don't use this
version in production.
"""
function FaceInfo(mesh::Mesh, kface::Int)
    f2n = connectivities_indices(mesh, :f2n)
    f2c = connectivities_indices(mesh, :f2c)

    _f2n = f2n[kface]
    fnodes = get_nodes(mesh, _f2n)
    ftype = faces(mesh)[kface]

    cellinfo_n = CellInfo(mesh, f2c[kface][1])
    cellinfo_p = length(f2c[kface]) > 1 ? CellInfo(mesh, f2c[kface][2]) : cellinfo_n

    return FaceInfo(cellinfo_n, cellinfo_p, ftype, fnodes, _f2n, kface)
end

nodes(faceInfo::FaceInfo) = faceInfo.faceNodes
facetype(faceInfo::FaceInfo) = faceInfo.faceType
faceindex(faceInfo::FaceInfo) = faceInfo.iface
get_cellinfo_n(faceInfo::FaceInfo) = faceInfo.cellinfo_n
get_cellinfo_p(faceInfo::FaceInfo) = faceInfo.cellinfo_p
@inline get_nodes_index(faceInfo::FaceInfo) = faceInfo.f2n
get_cell_side_n(faceInfo::FaceInfo) = faceInfo.cellside_n
get_cell_side_p(faceInfo::FaceInfo) = faceInfo.cellside_p
get_element_type(c::FaceInfo) = facetype(c)

"""
Return the opposite side of the `FaceInfo` : cellside "n" because cellside "p"
"""
function opposite_side(fInfo::FaceInfo)
    return FaceInfo(
        get_cellinfo_p(fInfo),
        get_cellinfo_n(fInfo),
        get_cell_side_p(fInfo),
        get_cell_side_n(fInfo),
        facetype(fInfo),
        nodes(fInfo),
        get_nodes_index(fInfo),
        faceindex(fInfo),
    )
end

"""
    get_face_normals(::AbstractFaceDomain)
    get_face_normals(::Measure{<:AbstractFaceDomain})

Return a LazyOperator representing a face normal
"""
get_face_normals(::AbstractFaceDomain) = FaceNormal()

"""
    get_cell_normals(::AbstractCellDomain)
    get_cell_normals(::Measure{<:AbstractCellDomain})

Return a LazyOperator representing a cell normal in the context of hypersurfaces (see [`cell_normal`]@ref for more details)

"""
function get_cell_normals(domain::AbstractCellDomain)
    mesh = get_mesh(domain)
    @assert topodim(mesh) < spacedim(mesh) "get_cell_normals on a CellDomain has only sense when dealing with hypersurface, maybe you confused it with get_face_normals?"
    return CellNormal()
end

abstract type AbstractDomainIterator{D <: AbstractDomain} end
get_domain(iter::AbstractDomainIterator) = iter.domain
Base.iterate(::AbstractDomainIterator) = error("to be defined")
Base.iterate(::AbstractDomainIterator, state) = error("to be defined")
Base.eltype(::AbstractDomainIterator) = error("to be defined")
Base.length(iter::AbstractDomainIterator) = length(indices(get_domain(iter)))
Base.firstindex(::AbstractDomainIterator) = 1
Base.getindex(::AbstractDomainIterator, i) = error("to be defined")

struct DomainIterator{D <: AbstractDomain} <: AbstractDomainIterator{D}
    domain::D
end

Base.eltype(::DomainIterator{<:AbstractCellDomain}) = CellInfo
Base.eltype(::DomainIterator{<:AbstractFaceDomain}) = FaceInfo

function Base.iterate(iter::DomainIterator, i::Integer = 1)
    if i > length(indices(get_domain(iter)))
        return nothing
    else
        return _get_index(get_domain(iter), i), i + 1
    end
end

Base.getindex(iter::DomainIterator, i) = _get_index(get_domain(iter), i)

function _get_index(domain::AbstractCellDomain, i::Integer)
    icell = indices(domain)[i]
    mesh = get_mesh(domain)
    _get_cellinfo(mesh, icell)
end
function _get_cellinfo(mesh, icell)
    c2n = connectivities_indices(mesh, :c2n)
    celltypes = cells(mesh)
    ctype = celltypes[icell]
    n_nodes = Val(nnodes(ctype))
    _c2n = c2n[icell, n_nodes]
    cnodes = get_nodes(mesh, _c2n)
    CellInfo(icell, ctype, cnodes, _c2n)
end

function _get_index(domain::AbstractFaceDomain, i::Integer)
    iface = indices(domain)[i]
    mesh = get_mesh(domain)
    f2n = connectivities_indices(mesh, :f2n)

    cellinfo1, cellinfo2 = _get_face_cellinfo(domain, i)

    ftype = faces(mesh)[iface]
    n_fnodes = Val(nnodes(ftype))
    _f2n = f2n[iface, n_fnodes]
    fnodes = get_nodes(mesh, _f2n)
    FaceInfo(cellinfo1, cellinfo2, ftype, fnodes, _f2n, iface)
end

function _get_face_cellinfo(domain::InteriorFaceDomain, i)
    mesh = get_mesh(domain)
    f2c = connectivities_indices(mesh, :f2c)
    iface = indices(domain)[i]
    icell1, icell2 = f2c[iface]
    cellinfo1 = _get_cellinfo(mesh, icell1)
    cellinfo2 = _get_cellinfo(mesh, icell2)
    return cellinfo1, cellinfo2
end

function _get_face_cellinfo(domain::AllFaceDomain, i)
    mesh = get_mesh(domain)
    f2c = connectivities_indices(mesh, :f2c)
    iface = indices(domain)[i]
    if length(f2c[iface]) > 1
        icell1, icell2 = f2c[iface]
        cellinfo1 = _get_cellinfo(mesh, icell1)
        cellinfo2 = _get_cellinfo(mesh, icell2)
        return cellinfo1, cellinfo2
    else
        icell1, = f2c[iface]
        cellinfo1 = _get_cellinfo(mesh, icell1)
        return cellinfo1, cellinfo1
    end
end

function _get_face_cellinfo(domain::BoundaryFaceDomain, i)
    mesh = get_mesh(domain)
    f2c = connectivities_indices(mesh, :f2c)
    iface = indices(domain)[i]
    icell1, = f2c[iface]
    cellinfo1 = _get_cellinfo(mesh, icell1)
    return cellinfo1, cellinfo1
end

function _get_index(domain::BoundaryFaceDomain{M, <:PeriodicBCType}, i::Integer) where {M}
    mesh = get_mesh(domain)
    c2n = connectivities_indices(mesh, :c2n)
    iface = indices(domain)[i]

    # TODO : add a specific API for the domain cache:
    perio_cache = get_cache(domain)
    perio_trans = transformation(get_bc(domain))
    _, bnd_f2n1, bnd_f2n2, bnd_f2c, bnd_ftypes, bnd_n2n = perio_cache

    icell1, icell2 = bnd_f2c[i, :]

    cellinfo1 = _get_cellinfo(mesh, icell1)

    ctype_j = cells(mesh)[icell2]
    nnodes_j = Val(nnodes(ctype_j))
    _c2n_j = c2n[icell2, nnodes_j]
    cnodes_j = get_nodes(mesh, _c2n_j) # to be removed when function barrier is used
    cnodes_j_perio = map(node -> Node(perio_trans(get_coords(node))), cnodes_j)
    # Build c2n for cell j with nodes indices taken from cell i
    # for those that are shared on the periodic BC:
    _c2n_j_perio = map(k -> get(bnd_n2n, k, k), _c2n_j)
    cellinfo2 = CellInfo(icell2, ctype_j, cnodes_j_perio, _c2n_j_perio)

    # Face info
    ftype = bnd_ftypes[i]
    fnodes = get_nodes(mesh, bnd_f2n1[i])
    cside_i = cell_side(celltype(cellinfo1), get_nodes_index(cellinfo1), bnd_f2n1[i])
    cside_j = cell_side(celltype(cellinfo2), _c2n_j, bnd_f2n2[i])
    _f2n = bnd_f2n1[i]

    return FaceInfo(cellinfo1, cellinfo2, cside_i, cside_j, ftype, fnodes, _f2n, iface)
end
