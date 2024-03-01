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

include("./mesh/transformation.jl")
export Translation

include("./mesh/boundary_condition.jl")
export BoundaryCondition, PeriodicBCType

include("./mesh/entity.jl")
export Node_t, Bar2_t, Bar3_t, Tri3_t, Quad4_t, Quad9_t, Tetra4_t, Hexa8_t, Poly2_t, Poly3_t
export Node

include("./mesh/shape.jl")

include("./mesh/connectivity.jl")

include("./mesh/mesh.jl")
export ncells, nnodes, boundary_names, nboundaries, boundary_tag, get_nodes

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
    CellDomain, InteriorFaceDomain, BoundaryFaceDomain, get_mesh, get_face_normals

include("./quadrature/quadrature.jl")
export QuadratureLobatto, QuadratureLegendre, QuadratureUniform, Quadrature, QuadratureRule

include("./function_space/function_space.jl")
export FunctionSpace, get_degree

include("./function_space/lagrange.jl")
include("./function_space/taylor.jl")

include("./mapping/mapping.jl")

include("./mapping/ref2phys.jl")
export get_cell_centers

include("./cellfunction/eval_point.jl")

include("./cellfunction/cellfunction.jl")
export PhysicalFunction, ReferenceFunction, side_p, side_n, side⁺, side⁻, jump

include("./cellfunction/meshdata.jl")
export MeshCellData, MeshPointData, get_values, set_values!

include("./fespace/dofhandler.jl")

include("./fespace/fespace.jl")
export TestFESpace, TrialFESpace, MultiplierFESpace, MultiFESpace, get_ndofs, get_fespace

include("./fespace/fefunction.jl")
export FEFunction, set_dof_values!, get_dof_values, get_fe_functions

include("./fespace/eval_shape_function.jl")

include("./integration/measure.jl")
export AbstractMeasure, Measure, get_domain

include("./integration/integration.jl")
export ∫

include("./algebra/gradient.jl")
export ∇, ∇ₛ

include("./algebra/algebra.jl")
export FaceNormal, otimes, ⊗, dcontract, ⊡

include("./assembler/assembler.jl")
export assemble_bilinear, assemble_linear, assemble_linear!

include("./assembler/dirichlet_condition.jl")
export assemble_dirichlet_vector,
    apply_dirichlet_to_matrix!,
    apply_dirichlet_to_vector!,
    apply_homogeneous_dirichlet_to_vector!

include("./assembler/affine_fe_system.jl")
export AffineFESystem

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
