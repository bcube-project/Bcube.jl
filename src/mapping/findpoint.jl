abstract type AbstractPointFinderStrategy end
struct StrictPointFinderStrategy <: AbstractPointFinderStrategy end
struct ClosestPointFinderStrategy <: AbstractPointFinderStrategy end

struct PointFinder{T, M, S}
    tree::T
    mesh::M
    strategy::S
end

function PointFinder(
    mesh::AbstractMesh;
    strategy::AbstractPointFinderStrategy = ClosestPointFinderStrategy(),
)
    xc0 = get_cell_centers(mesh)
    xc = reshape(
        [xc0[n][idim] for n in 1:length(xc0) for idim in 1:spacedim(mesh)],
        spacedim(mesh),
        length(xc0),
    )
    tree = KDTree(xc; leafsize = 10)
    PointFinder{typeof(tree), typeof(mesh), typeof(strategy)}(tree, mesh, strategy)
end

function find_cell_index(pf::PointFinder, x)
    idxs, dists = knn(pf.tree, x, 1)
    c2c_n = connectivity_cell2cell_by_nodes(pf.mesh)
    cellids = [idxs[1], c2c_n[idxs[1]]...]
    for icell in cellids
        cinfo = CellInfo(pf.mesh, icell)
        cpoint = CellPoint(x, cinfo, PhysicalDomain())
        is_point_in_cell(cinfo, cpoint) && (return icell)
    end
    isa(pf.strategy, ClosestPointFinderStrategy) && (return idxs[1])
    return missing
end

function find_cell_point(pf::PointFinder, x)
    icell = find_cell_index(pf, x)
    ismissing(icell) && (return missing)
    CellPoint(x, CellInfo(pf.mesh, icell), Bcube.PhysicalDomain())
end

function interpolate_at_point(pf::PointFinder, x, u)
    cpoint = find_cell_point(pf, x)
    ismissing(cpoint) && (return missing)
    uᵢ = materialize(u, get_cellinfo(cpoint))
    materialize(uᵢ, cpoint)
end

function is_point_in_cell(cinfo, cpoint)
    cpoint_ref = change_domain(cpoint, ReferenceDomain())
    is_point_in_shape(shape(celltype(cinfo)), get_coords(cpoint_ref))
end
