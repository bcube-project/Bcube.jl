module RunCPU
using KernelAbstractions
using AcceleratedKernels

include(joinpath(@__DIR__, "BcubeGPU.jl"))
using .BcubeGPU
using CUDA
using Cthulhu
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

function run()
    #  CUDA.versioninfo()
    backend = get_backend(CUDA.ones(2))

    BcubeGPU.run(backend, 1000, 1000)
end

# --- NVIDIA A30 ---
# FP64 	 5,2 TFlops
# FP32 	10,3 TFlops

# --- CPU JUNO (queue GPU)
# FP64   1,6  TFlops (48 cores)
#        0,033 TFlops (1 core) --> speedup GPU = 150

# --- NVIDIA A100 ---
# FP64 	 9,7 TFlops
# FP32 	19,5 TFlops

#--- CPU TURPAN
# F64  1.3     (80 cores)
#      0.02375   (1 core)  --> speedup GPU A100 = 821

run()
end
