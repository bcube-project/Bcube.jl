module Bcube

using Base
using Base: @propagate_inbounds
using StaticArrays
using SparseArrays
using FEMQuad
using FastGaussQuadrature
using ForwardDiff
using LinearAlgebra
using WriteVTK
using Printf # just for tmp vtk, to be removed
# import LinearSolve: solve, solve!, LinearProblem
import LinearSolve
using Symbolics # used for generation of Lagrange shape functions

const MAX_LENGTH_STATICARRAY = (10^6)

include("LazyOperators/LazyOperators.jl")
using .LazyOperators
import .LazyOperators:
    materialize, materialize_args, AbstractLazyOperator, get_args, get_operator, unwrap

include("utils.jl")
export densify!, densify, myrand, rawcat

include("./mesh/transformation.jl")
export Translation, get_transformation

include("./mesh/boundary_condition.jl")
export BoundaryCondition, apply, type, PeriodicBCType

include("./mesh/entity.jl")
export AbstractEntityType
export Node_t, Bar2_t, Bar3_t, Tri3_t, Quad4_t, Quad9_t, Tetra4_t, Hexa8_t, Poly2_t, Poly3_t
export Node
export nnodes,
    nodes,
    nedges,
    edges2nodes,
    nfaces,
    faces2nodes,
    facetypes,
    f2n_from_c2n,
    coords,
    cell_side,
    oriented_cell_side,
    topology_style

include("./mesh/shape.jl")
export Line,
    Triangle,
    Square,
    Prism,
    Tetra,
    Cube,
    shape,
    nvertices,
    nedges,
    nfaces,
    face_area,
    faces2nodes,
    face_shapes,
    coords,
    normals

include("./mesh/connectivity.jl")
export AbstractConnectivity, Connectivity, minsize, maxsize, inverse_connectivity

include("./mesh/mesh.jl")
export AbstractMesh,
    topodim,
    spacedim,
    MeshConnectivity,
    from,
    to,
    by,
    nlayers,
    indices,
    Mesh,
    get_nodes,
    set_nodes,
    n_entities,
    nnodes,
    nvertices,
    nedges,
    nfaces,
    ncells, #nodes, <-useless ?
    cells,
    faces,
    has_entities,
    has_cells,
    has_nodes,
    has_vertices,
    has_edges,
    has_faces,
    entities,
    connectivities,
    connectivities_indices,
    has_connectivities,
    boundary_names,
    boundary_nodes,
    boundary_faces,
    nboundaries,
    boundary_tag,
    build_boundary_faces!,
    absolute_indices,
    local_indices,
    add_absolute_indices!,
    inner_faces,
    outer_faces,
    connectivity_cell2cell_by_faces,
    connectivity_cell2cell_by_nodes

include("./mesh/gmsh_utils.jl")
export read_msh,
    read_msh_with_cell_names,
    gen_line_mesh,
    gen_rectangle_mesh,
    gen_hexa_mesh,
    gen_disk_mesh,
    gen_star_disk_mesh,
    gen_cylinder_mesh,
    read_partitions,
    gen_rectangle_mesh_with_tri_and_quad

include("./mesh/mesh_generator.jl")
export basic_mesh,
    one_cell_mesh,
    line_mesh,
    rectangle_mesh,
    ncube_mesh,
    circle_mesh,
    scale,
    scale!,
    transform,
    transform!,
    translate,
    translate!

include("./mesh/domain.jl")
export AbstractDomain,
    CellDomain,
    InteriorFaceDomain,
    BoundaryFaceDomain,
    CellInfo,
    FaceInfo,
    get_mesh,
    get_face_normals

include("./quadrature/quadrature.jl")
export QuadratureLobatto, QuadratureLegendre, QuadratureUniform, Quadrature, QuadratureRule

include("./function_space/function_space.jl")
export FunctionSpace,
    shape_functions,
    grad_shape_functions,
    idof_by_vertex,
    idof_by_edge,
    idof_by_edge_with_bounds,
    idof_by_face,
    idof_by_face_with_bounds,
    ndofs,
    get_degree,
    coords,
    get_type

include("./function_space/lagrange.jl")
include("./function_space/taylor.jl")

include("./mapping/mapping.jl")
export mapping,
    mapping_jacobian,
    mapping_det_jacobian,
    mapping_inv,
    mapping_inv_jacobian,
    mapping_jacobian_inv,
    mapping_face

include("./mapping/ref2phys.jl")
export normal, center, grad_shape_functions, interpolate, cell_normal, get_cell_centers

include("./cellfunction/eval_point.jl")

include("./cellfunction/cellfunction.jl")
export PhysicalFunction, ReferenceFunction, side_p, side_n, side⁺, side⁻, jump

include("./cellfunction/meshdata.jl")
export MeshCellData, MeshPointData, get_values, set_values!

include("./fespace/dofhandler.jl")
export dof, max_ndofs

include("./fespace/fespace.jl")
export TestFESpace, TrialFESpace, MultiplierFESpace, MultiFESpace, get_ndofs, get_fespace

include("./fespace/fefunction.jl")
export FEFunction, set_dof_values!, get_dof_values, get_fe_functions

include("./fespace/eval_shape_function.jl")

include("./integration/measure.jl")
export AbstractMeasure, Measure, get_domain

include("./integration/integration.jl")
export ∫,
    integrate,
    integrate_ref,
    integrate_n_ref,
    integrate_n,
    getcache_∫,
    InvMassMatrix,
    sparse,
    IntegralResult,
    assemble,
    result

include("./assembler/assembler.jl")
export assemble_bilinear, assemble_linear, assemble_linear!

include("./assembler/dirichlet_condition.jl")
export assemble_dirichlet_vector,
    apply_dirichlet_to_matrix!,
    apply_dirichlet_to_vector!,
    apply_homogeneous_dirichlet_to_vector!

include("./assembler/affine_fe_system.jl")
export AffineFESystem

include("./algebra/gradient.jl")
export ∇

include("./algebra/algebra.jl")
export FaceNormal, otimes, ⊗, dcontract, ⊡

include("./feoperator/projection_newapi.jl")
export projection_l2!

include("./feoperator/projection.jl")
export var_on_centers,
    var_on_vertices, var_on_nodes_discontinuous, var_on_bnd_nodes_discontinuous

include("./feoperator/limiter.jl")
export linear_scaling_limiter

include("./writers/vtk.jl")
export write_vtk

end
