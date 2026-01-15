# Axis-Aligned Bounding Box
struct AABB{Dim, T <: Real}
    min::SVector{Dim, T}
    max::SVector{Dim, T}
end

function AABB(min::SVector{N, T}, max::SVector{N, T}) where {N, T}
    AABB{N, T}(min, max)
end

# Boxed cell
struct BCell{CI, B}
    cellindex::CI
    bbox::B
end

center(bcell::BCell, axis::Integer) = 0.5 * (bcell.bbox.min[axis] + bcell.bbox.max[axis])

# Nœud BVH
struct BVHNode{B, C}
    bbox::B
    left::Union{BVHNode{B, C}, Nothing}
    right::Union{BVHNode{B, C}, Nothing}
    bcells::Vector{C}   # non vide uniquement pour les feuilles
end

# Bounding box d'un ensemble de points
function compute_bbox(points::AbstractVector{<:SVector{N}}) where {N}
    xmin = ntuple(i -> minimum(p[i] for p in points), Val(N))
    xmax = ntuple(i -> maximum(p[i] for p in points), Val(N))
    AABB(SVector{N}(xmin), SVector{N}(xmax))
end

# Union de deux bounding boxes
function union_bbox(a::AABB, b::AABB)
    AABB(min.(a.min, b.min), max.(a.max, b.max))
end

# Bounding box of a list of boxed-cells
function compute_node_bbox(bcells)
    bbox = bcells[1].bbox
    for c in bcells[2:end]
        bbox = union_bbox(bbox, c.bbox)
    end
    bbox
end

function make_bcell(cellinfo::Bcube.CellInfo)
    cnodes = map(get_coords, Bcube.nodes(cellinfo))
    bbox = compute_bbox(cnodes)
    BCell(Bcube.get_element_index(cellinfo), bbox)
end

longest_axis(bbox::AABB) = argmax(bbox.max - bbox.min)

function build_bvh(domain::CellDomain; max_leaf_size = 4)
    bcells = BCell[]
    @assert Threads.nthreads() == 1 "build_bvh is not working with nthreads>1"
    Bcube.foreach_element(domain) do element, _, _
        push!(bcells, make_bcell(element))
    end
    build_bvh(bcells; max_leaf_size = max_leaf_size)
end

function build_bvh(bcells::Vector{BCell}; max_leaf_size = 4)
    node_bbox = compute_node_bbox(bcells)

    # stop condition
    if length(bcells) ≤ max_leaf_size
        return BVHNode(node_bbox, nothing, nothing, bcells)
    end

    # select axis
    axis = longest_axis(node_bbox)

    # sorting according centers
    sorted_cells = sort(bcells; by = c -> center(c, axis))

    mid         = length(sorted_cells) ÷ 2
    left_cells  = sorted_cells[1:mid]
    right_cells = sorted_cells[(mid + 1):end]

    # backup if splitting fails
    if isempty(left_cells) || isempty(right_cells)
        return BVHNode(node_bbox, nothing, nothing, cells)
    end

    left_node  = build_bvh(left_cells; max_leaf_size)
    right_node = build_bvh(right_cells; max_leaf_size)

    BVHNode(node_bbox, left_node, right_node, BCell[])
end

function point_in_bbox(p, bbox::AABB{Dim}; eps = 1e-12) where {Dim}
    all(i -> (bbox.min[i] - eps ≤ p[i] ≤ bbox.max[i] + eps), 1:Dim)
end

# "exact" test
function point_inside_cell(domain, cellindex, p)
    cellinfo = domain[cellindex]
    cpoint = Bcube.CellPoint(p, cellinfo, Bcube.PhysicalDomain())
    Bcube.is_point_in_cell(cellinfo, cpoint)
end

function find_bcell(node::BVHNode, domain, point)
    if !point_in_bbox(point, node.bbox)
        return nothing
    end

    if node.left === nothing && node.right === nothing
        for bcell in node.bcells
            point_inside_cell(domain, bcell.cellindex, point) && (return bcell.cellindex)
        end
        return nothing
    end

    found = (node.left !== nothing) ? find_bcell(node.left, domain, point) : nothing
    (found !== nothing) && (return found)

    return (node.right !== nothing) ? find_bcell(node.right, domain, point) : nothing
end

function test_bvh()
    mesh = Bcube.rectangle_mesh(21, 21)
    domain = Bcube.CellDomain(mesh)
    bcells = BCell[]
    Bcube.foreach_element(domain) do element, i, _
        push!(bcells, make_bcell(element))
    end

    bvh = build_bvh(bcells)

    p = SA[0.13, 0.23]
    cell = find_bcell(bvh, Bcube.DomainIterator(domain), p)
    println()
    @show cell
end
