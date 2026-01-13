abstract type AbstractPointFinderStrategy end
struct StrictPointFinderStrategy <: AbstractPointFinderStrategy end
struct ClosestPointFinderStrategy <: AbstractPointFinderStrategy end

struct PointFinder{T, M, S, C}
    tree::T
    mesh::M
    strategy::S
    c2c_n::C
end

"""
    PointFinder(mesh::AbstractMesh; strategy::AbstractPointFinderStrategy = ClosestPointFinderStrategy())

A `PointerFinder` helps locating some arbitrary coordinates on a `Mesh`.
Given some coordinates `x`, the cell containing this point can be identified,
and a corresponding `CellPoint` can be built to evaluate `Bcube` lazy operators.

# Example
```julia
mesh = rectangle_mesh(10,10)
pf = PointFinder(mesh)
x = rand(2)

icell = find_cell_index(pf, x) # find the index of the cell index containing x
cpoint = find_cell_point(pf, x) # build a `CellPoint` corresponding to x

u = FEFunction(TrialFESpace(FunctionSpace(:Lagrange, 1), mesh))
interpolate_at_point(pf, x, u) # interpolate u on x
```
"""
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
    c2c_n = connectivity_cell2cell_by_nodes(mesh)
    PointFinder{typeof(tree), typeof(mesh), typeof(strategy), typeof(c2c_n)}(
        tree,
        mesh,
        strategy,
        c2c_n,
    )
end

function find_cell_index(pf::PointFinder, x)
    idxs, dists = knn(pf.tree, x, 1)
    cellids = [idxs[1], pf.c2c_n[idxs[1]]...]
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

function interpolate_at_point(
    pf::PointFinder,
    x::AbstractArray{<:Number},
    u::Vararg{AbstractLazy, N},
) where {N}
    cpoint = find_cell_point(pf, x)
    ismissing(cpoint) && (return missing)
    uᵢ = materialize(u, get_cellinfo(cpoint))
    return materialize(uᵢ, cpoint)
end

"""
    is_point_in_cell(cinfo, cpoint)

Indicate if `cpoint` lies inside the cell `cinfo`.

# Dev notes
This function triggers a call to `mapping_inv`. But `mapping_inv` is undefined
for points outside the cell, and a `DomainError` is thrown when the situation occurs.
Hence if a `DomainError` is catched during the evaluation of `change_domain`, it only means
that the point is outside the cell. If it's an other error type, an error is thrown.
"""
function is_point_in_cell(cinfo, cpoint)
    result = false
    try
        cpoint_ref = change_domain(cpoint, ReferenceDomain())
        result = is_point_in_shape(shape(celltype(cinfo)), get_coords(cpoint_ref))
    catch err
        @assert err isa DomainError
    end
    return result
end
