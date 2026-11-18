module BenchDirichlet
using Bcube, BenchmarkTools, SparseArrays
using Bcube: assemble_dirichlet_sparse_vector, apply_homogeneous_dirichlet_to_matrix!

# One fixed 41 × 21 vertex mesh. Continuous Q2 velocity (two components)
# and Q1 pressure exercise shared boundary dofs and mixed global mappings.
const mesh = rectangle_mesh(41, 21)
const velocity = TrialFESpace(
    FunctionSpace(:Lagrange, 2),
    mesh,
    Dict("xmin" => [1.0, 2.0], "ymin" => [1.0, 2.0]);
    size = 2,
)
const pressure =
    TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, Dict("xmin" => 3.0, "ymin" => 3.0))
const suite = BenchmarkGroup()
for (label, U) in (("single", velocity), ("multi", MultiFESpace(velocity, pressure)))
    V = if U isa Bcube.AbstractMultiFESpace
        MultiFESpace(map(TestFESpace, Tuple(U))...)
    else
        TestFESpace(U)
    end
    n = Bcube.get_ndofs(U)
    # Assemble the complete reference outside measurement, including zero entries.
    prescribed = assemble_dirichlet_sparse_vector(U, V, mesh)
    indices = findnz(prescribed)[1]
    group = suite[label] = BenchmarkGroup()
    group["assemble"] =
        @benchmarkable assemble_dirichlet_sparse_vector($U, $V, $mesh) evals=1
    for (storage, reference) in (
        ("dense", ones(n)),
        ("sparse_missing", spzeros(n)),
        ("sparse_present", sparsevec(indices, ones(length(indices)), n)),
    )
        group["vector_" * storage] =
            @benchmarkable apply_dirichlet_to_vector!(b, $U, $V, $mesh) setup=(
                b=copy($reference)
            ) evals=1
    end
    for (storage, reference) in (
        ("missing", spdiagm(1 => ones(n-1))),
        ("present", spdiagm(-1 => ones(n-1), 0 => ones(n), 1 => ones(n-1))),
    )
        group["matrix_" * storage] =
            @benchmarkable apply_dirichlet_to_matrix!(A, $U, $V, $mesh) setup=(
                A=copy($reference)
            ) evals=1
        group["homogeneous_" * storage] =
            @benchmarkable apply_homogeneous_dirichlet_to_matrix!(A, $U, $V, $mesh) setup=(
                A=copy($reference)
            ) evals=1
    end
end
end
BenchDirichlet.suite
