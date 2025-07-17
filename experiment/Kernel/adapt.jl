using Bcube
using StaticArrays
using KernelAbstractions
using GPUArrays
using Adapt
import AcceleratedKernels as AK
using BenchmarkTools
using Profile
using InteractiveUtils
using CUDA
import Bcube:
    AbstractCellDomain,
    AbstractFaceDomain,
    AllFaceDomain,
    CellInfo,
    CellPoint,
    Connectivity,
    DofHandler,
    DomainIterator,
    FaceInfo,
    FacePoint,
    FaceSidePair,
    LazyMapOver,
    Mesh,
    MeshConnectivity,
    NullOperator,
    PhysicalDomain,
    ReferenceDomain,
    Side⁻,
    Side⁺,
    SingleFESpace,
    SingleFieldFEFunction,
    boundary_faces,
    boundary_nodes,
    connectivities,
    entities,
    get_args,
    get_cellinfo_n,
    get_cellinfo_p,
    get_cell_shape_functions,
    get_cell_side_n,
    get_cell_side_p,
    get_coords,
    get_dof,
    get_dirichlet_boundary_tags,
    get_element_index,
    get_element_type,
    get_function_space,
    get_metadata,
    get_quadrature,
    idof_by_face_with_bounds,
    indices,
    integrate_on_ref_element,
    is_continuous,
    nfaces,
    nlayers,
    shape,
    _get_dhl,
    _get_index,
    _scalar_shape_functions

const WORKGROUP_SIZE = 256

#>>>>>>>> Adapt some structures
Adapt.@adapt_structure Connectivity

function Adapt.adapt_structure(to, conn::MeshConnectivity{C, F, T, B}) where {C, F, T, B}
    layers = adapt(to, nlayers(conn))
    ind = adapt(to, indices(conn))
    MeshConnectivity{typeof(ind), F, T, B, typeof(layers)}(layers, ind)
end

function Adapt.adapt_structure(to, mesh::Mesh)
    nodes_gpu = adapt(to, get_nodes(mesh))
    entities_gpu = adapt(to, entities(mesh))
    connectivities_gpu = adapt(to, connectivities(mesh))
    bc_nodes_gpu = adapt(to, boundary_nodes(mesh))
    bc_faces_gpu = adapt(to, boundary_faces(mesh))
    metadata_gpu = adapt(to, get_metadata(mesh))

    Mesh{
        topodim(mesh),
        spacedim(mesh),
        typeof(nodes_gpu),
        typeof(entities_gpu),
        typeof(connectivities_gpu),
        typeof(bc_nodes_gpu),
        typeof(bc_faces_gpu),
        typeof(metadata_gpu),
    }(
        nodes_gpu,
        entities_gpu,
        connectivities_gpu,
        bc_nodes_gpu,
        bc_faces_gpu,
        metadata_gpu,
    )
end

function Adapt.adapt_structure(to, cinfo::CellInfo)
    cellindex_gpu = adapt(to, cellindex(cinfo))
    celltype_gpu = adapt(to, celltype(cinfo))
    nodes_gpu = adapt(to, nodes(cinfo))
    nodes_index_gpu = adapt(to, get_nodes_index(cinfo))

    CellInfo{typeof(celltype_gpu), typeof(nodes_gpu), typeof(nodes_index_gpu)}(
        cellindex_gpu,
        celltype_gpu,
        nodes_gpu,
        nodes_index_gpu,
    )
end

Adapt.@adapt_structure CellDomain
Adapt.@adapt_structure InteriorFaceDomain

function Adapt.adapt_structure(to, b::BoundaryFaceDomain)
    #println("Running adapt on BoundaryFaceDomain")
    mesh = adapt(to, Bcube.get_mesh(b))
    bc = adapt(to, Bcube.get_bc(b))
    labels = adapt(to, b.labels)
    cache = adapt(to, Bcube.get_cache(b))

    # if !isbits(bc)
    #     @show typeof(bc)
    #     error("bc")
    # end
    # if !isbits(labels)
    #     @show typeof(labels)
    #     error("labels")
    # end
    # if !isbits(cache)
    #     @show typeof(cache)
    #     error("cache")
    # end

    BoundaryFaceDomain{typeof(mesh), typeof(bc), typeof(labels), typeof(cache)}(
        mesh,
        bc,
        labels,
        cache,
    )
end

function Adapt.adapt_structure(
    to,
    b::BoundaryFaceDomain{M, BC},
) where {M, BC <: Bcube.PeriodicBCType}
    error("not implemented yet")
end

Adapt.@adapt_structure Measure

Adapt.@adapt_structure DofHandler

function Adapt.adapt_structure(to, feSpace::SingleFESpace{S, FS}) where {S, FS}
    dhl = adapt(to, _get_dhl(feSpace))
    tags = adapt(to, get_dirichlet_boundary_tags(feSpace))
    SingleFESpace{S, FS, typeof(dhl), typeof(tags)}(
        get_function_space(feSpace),
        dhl,
        is_continuous(feSpace),
        tags,
    )
end

Adapt.@adapt_structure TestFESpace
Adapt.@adapt_structure TrialFESpace
Adapt.@adapt_structure SingleFieldFEFunction

#<<<<<<<< Adapt some structures

@kernel function inner_faces_kernel!(n_neighbors, @Const(f2c))
    iface = @index(Global)
    n_neighbors[iface] = length(f2c[iface])
end

function Bcube.inner_faces(mesh::Mesh{T, S, N}) where {T, S, N <: AbstractGPUArray}
    # TODO : recall why we can't just use `n_neighbors = AK.map(length, f2c)` ?
    # (maybe I haven't tried)
    f2c = indices(connectivities(mesh, :f2c))
    backend = get_backend(get_nodes(mesh))
    n_neighbors = KernelAbstractions.zeros(backend, Int, nfaces(mesh))
    inner_faces_kernel!(backend, WORKGROUP_SIZE)(
        n_neighbors,
        f2c;
        ndrange = size(n_neighbors),
    )
    return findall(n_neighbors .> 1)
end
