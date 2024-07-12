struct PointFinder{T, M}
    tree::T
    mesh::M
end

function PointFinder(mesh::AbstractMesh)
    xc0 = get_cell_centers(mesh)
    xc = reshape(
        [xc0[n][idim] for n in 1:length(xc0) for idim in 1:spacedim(mesh)],
        spacedim(mesh),
        length(xc0),
    )
    tree = KDTree(xc; leafsize = 10)
    PointFinder{typeof(tree), typeof(mesh)}(tree, mesh)
end

function find_cell(pf::PointFinder, point)
    idxs, dists = knn(pf.tree, point, 1)
    c2c_n = connectivity_cell2cell_by_nodes(pf.mesh)
    cellids = [idxs[1], c2c_n[idxs[1]]...]

    for icell in cellids
        cinfo = CellInfo(pf.mesh, icell)
        cpoint = CellPoint(point, cinfo, PhysicalDomain())
        isIn = point_in_cell(cinfo, cpoint)
        isIn && return icell
    end
    return nothing
end

function point_in_cell(cinfo, cpoint)
    cpoint_ref = change_domain(cpoint, ReferenceDomain())
    point_in_shape(shape(celltype(cinfo)), get_coords(cpoint_ref))
end

function point_in_shape(s::Square, x)
    get_coords(s)[1][1] ≤ x[1] ≤ get_coords(s)[3][1] &&
        get_coords(s)[1][2] ≤ x[2] ≤ get_coords(s)[3][2]
end

function point_in_shape(shape::AbstractShape, x)
    for (normal, f2n) in zip(normals(shape), faces2nodes(shape))
        dx = (x - get_coords(shape)[first(f2n)])
        (dx ⋅ normal > 0) && (return false)
    end
    return true
end