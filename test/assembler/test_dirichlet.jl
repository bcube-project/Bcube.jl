@testset "Dirichlet" begin
    mesh = rectangle_mesh(3, 2)
    dΩ = Measure(CellDomain(mesh), 1)
    U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, Dict("xmin" => 0.0))
    V = TestFESpace(U)
    A = assemble_bilinear((u, v) -> ∫(u ⋅ v)dΩ, U, V)
    B = copy(A)
    Bcube.apply_dirichlet_to_matrix!((A, B), U, V, mesh; diag_values = (2.0, 0.0))
    IA, JA, VA = findnz(A)
    @test IA == [
        1,
        2,
        3,
        4,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        1,
        2,
        3,
        4,
        5,
        6,
        2,
        4,
        5,
        6,
        2,
        4,
        5,
        6,
    ]
    @test JA == [
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
    ]
    @test VA == [
        2.0,
        0.03125,
        0.0,
        0.03125,
        0.0,
        0.0625,
        0.0,
        0.0625,
        0.03125,
        0.03125,
        0.0,
        0.03125,
        2.0,
        0.03125,
        0.0,
        0.0625,
        0.0,
        0.0625,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
    ]
    IB, JB, VB = findnz(B)
    @test IB == IA
    @test JB == JA
    @test VB == [
        0.0,
        0.03125,
        0.0,
        0.03125,
        0.0,
        0.0625,
        0.0,
        0.0625,
        0.03125,
        0.03125,
        0.0,
        0.03125,
        0.0,
        0.03125,
        0.0,
        0.0625,
        0.0,
        0.0625,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
        0.03125,
    ]
end

@testset "Dirichlet with FEFunction" begin
    mesh = rectangle_mesh(4, 3)
    fSpace = FunctionSpace(:Lagrange, 1)
    u = FEFunction(TrialFESpace(fSpace, mesh), mesh, PhysicalFunction(x -> x[2] + 1))
    U = TrialFESpace(fSpace, mesh, Dict("xmin" => t -> u))
    V = TestFESpace(U)
    x = Bcube.assemble_dirichlet_sparse_vector(U, V, mesh, 0.0)
    @test x == sparsevec([1, 3, 9], [1.0, 1.5, 2.0], 12)
end

@testset "Sparse dirichlet" begin
    mesh = rectangle_mesh(4, 3)
    fSpace = FunctionSpace(:Lagrange, 1)
    U = TrialFESpace(fSpace, mesh, Dict("xmin" => PhysicalFunction(x -> x[2])))
    V = TestFESpace(U)
    x = Bcube.assemble_dirichlet_sparse_vector(U, V, mesh)
    @test x == sparsevec([3, 9], [0.5, 1.0], 12)
end

@testset "Dirichlet storage contracts" begin
    DIRICHLET_BACKENDS = ((mesh, U, V) -> (mesh, U, V),)
    @testset "Prepared Dirichlet storage" begin
        mesh_cpu = rectangle_mesh(4, 3)
        scalar = TrialFESpace(
            FunctionSpace(:Lagrange, 1),
            mesh_cpu,
            Dict(
                "xmin" => PhysicalFunction(x -> 2 + x[1] + 3*x[2]),
                "ymin" => PhysicalFunction(x -> 2 + x[1] + 3*x[2]),
            ),
        )
        vector = TrialFESpace(
            FunctionSpace(:Lagrange, 2),
            mesh_cpu,
            Dict(
                "xmin" => PhysicalFunction(x -> [3 + x[1], 4 + 2*x[2]]),
                "ymin" => PhysicalFunction(x -> [3 + x[1], 4 + 2*x[2]]),
            );
            size = 2,
        )
        for trial_cpu in (scalar, MultiFESpace(vector, scalar))
            test_cpu = if trial_cpu isa Bcube.AbstractMultiFESpace
                MultiFESpace(map(TestFESpace, Tuple(trial_cpu))...)
            else
                TestFESpace(trial_cpu)
            end
            # The original sequential boundary evaluator supplies an independent oracle
            # for topology, FE evaluation and mixed mappings.
            expected_values = zeros(Bcube.get_ndofs(trial_cpu))
            expected_mask = fill(false, length(expected_values))
            Bcube.foreach_dirichlet_value!(
                (expected_values, expected_mask),
                ((a, i, c, v) -> (a[i] = v[c]), (a, i, c, v) -> (a[i] = true)),
                trial_cpu,
                test_cpu,
                Bcube.prepare_dirichlet_boundaries(trial_cpu, mesh_cpu),
                0.0,
            )
            indices = findall(expected_mask)
            n = length(expected_values)
            unconstrained = findall(.!expected_mask)
            partial_sparse = sparsevec(unconstrained, fill(7.0, length(unconstrained)), n)

            for accelerated in DIRICHLET_BACKENDS
                mesh, U, V = accelerated(mesh_cpu, trial_cpu, test_cpu)
                boundaries = Bcube.prepare_dirichlet_boundaries(U, mesh)
                backend = Bcube.get_bcube_backend(mesh)
                I_diri, position_by_dof = Bcube.dirichlet_data(U, V, mesh, boundaries)
                @test Set(I_diri) == Set(indices)
                @test length(I_diri) == length(indices)
                @test position_by_dof[I_diri] == collect(eachindex(I_diri))
                @test all(iszero, position_by_dof[unconstrained])
                compact_values = Bcube.allocate_dirichlet_values(
                    I_diri,
                    Float64,
                )
                @test length(compact_values) == length(I_diri)
                Bcube.evaluate_dirichlet_values!(
                    compact_values,
                    position_by_dof,
                    U,
                    V,
                    boundaries,
                    backend,
                    0.0,
                )
                @test compact_values == expected_values[I_diri]
                @test Bcube.assemble_dirichlet_vector(U, V, mesh) == expected_values
                for reference in (
                    fill(7.0, n),
                    spzeros(n),
                    partial_sparse,
                    sparsevec(1:n, fill(7.0, n), n),
                )
                    b = copy(reference)
                    expected = Vector(reference)
                    expected[indices] = expected_values[indices]
                    Bcube.apply_dirichlet_to_vector!(b, U, V, mesh)
                    @test b == expected
                    Bcube.apply_homogeneous_dirichlet_to_vector!(b, U, V, mesh)
                    expected[indices] .= 0
                    @test b == expected
                end
                for reference in (
                    spdiagm(1 => fill(5.0, n-1)),
                    spdiagm(-1 => fill(3.0, n-1), 0 => fill(4.0, n), 1 => fill(5.0, n-1)),
                )
                    for homogeneous in (false, true)
                        A, B = copy(reference), copy(reference)
                        expected_A, expected_B = Matrix(reference), Matrix(reference)
                        for (expected, diagonal) in ((expected_A, 2.0), (expected_B, 0.0))
                            expected[indices, :] .= 0
                            homogeneous && (expected[:, indices] .= 0)
                            for i in indices
                                expected[i, i] = diagonal
                            end
                        end
                        apply! = if homogeneous
                            Bcube.apply_homogeneous_dirichlet_to_matrix!
                        else
                            Bcube.apply_dirichlet_to_matrix!
                        end
                        apply!((A, B), U, V, mesh; diag_values = (2.0, 0.0))
                        @test A == expected_A
                        @test B == expected_B
                    end
                end
                @test_throws AssertionError Bcube.apply_dirichlet_to_vector!(
                    zeros(n),
                    U,
                    V,
                    mesh;
                    dt_derivative_order = 1,
                )
            end
        end

        U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh_cpu)
        V = TestFESpace(U)
        n = Bcube.get_ndofs(U)
        for accelerated in DIRICHLET_BACKENDS
            mesh, trial, test = accelerated(mesh_cpu, U, V)
            b, A = ones(n), spdiagm(0 => ones(n))
            Bcube.apply_dirichlet_to_vector!(b, trial, test, mesh)
            Bcube.apply_homogeneous_dirichlet!(A, b, trial, test, mesh)
            @test b == ones(n)
            @test A == spdiagm(0 => ones(n))
            boundaries = Bcube.prepare_dirichlet_boundaries(trial, mesh)
            I_diri, position_by_dof = Bcube.dirichlet_data(trial, test, mesh, boundaries)
            @test isempty(I_diri)
            @test position_by_dof == zeros(Int, n)
        end
    end
end

@testset "Dirichlet matrices must be square" begin
    mesh = rectangle_mesh(3, 3)
    U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, Dict("xmin" => 1.0))
    V = TestFESpace(U)
    n = Bcube.get_ndofs(U)
    rectangular_matrix = spzeros(n, n + 1)
    square_matrix = spzeros(n, n)

    @test_throws AssertionError Bcube.apply_dirichlet_to_matrix!(
        rectangular_matrix, U, V, mesh,
    )
    @test_throws AssertionError Bcube.apply_homogeneous_dirichlet_to_matrix!(
        (square_matrix, rectangular_matrix), U, V, mesh,
    )
    @test nnz(square_matrix) == 0
end

@testset "Topology and homogeneous constraints do not evaluate values" begin
    mesh = rectangle_mesh(3, 3)
    U = TrialFESpace(
        FunctionSpace(:Lagrange, 1),
        mesh,
        Dict("xmin" => t -> error("Boundary values must not be evaluated")),
    )
    V = TestFESpace(U)
    boundaries = Bcube.prepare_dirichlet_boundaries(U, mesh)
    backend = Bcube.get_bcube_backend(mesh)
    indices, position_by_dof = Bcube.dirichlet_data(U, V, mesh, boundaries)
    next_indices, next_positions = Bcube.dirichlet_data(U, V, mesh, boundaries)
    @test Set(indices) == Set([1, 3, 7])
    occurrences = Bcube.collect_dirichlet_indices(U, V, backend, boundaries)
    @test indices == unique(occurrences)
    @test indices !== next_indices
    @test position_by_dof !== next_positions
    n = Bcube.get_ndofs(U)
    A, b = spzeros(n, n), ones(n)
    Bcube.apply_dirichlet_to_matrix!(A, U, V, mesh)
    @test diag(A)[indices] == ones(length(indices))
    Bcube.apply_homogeneous_dirichlet!(A, b, U, V, mesh)
    @test b[indices] == zeros(length(indices))
end

@testset "Unsorted compact Dirichlet positions" begin
    # Compact encounter order deliberately differs from canonical sparse storage.
    I_diri = [6, 2, 5]
    position_by_dof = zeros(Int, 8)
    position_by_dof[I_diri] = collect(eachindex(I_diri))
    V_diri = [60.0, 20.0, 50.0]
    backend = Bcube.BcubeBackendCPUSerial()
    for reference in (fill(7.0, 8), sparsevec([1, 3, 5], [-0.0, 7.0, 9.0], 8))
        b = copy(reference)
        positions = Bcube.prepare_dirichlet_vector!(b, I_diri, position_by_dof, backend)
        @test b == reference
        Bcube.set_dirichlet_values!(b, positions, V_diri, backend)
        expected = Vector(reference)
        expected[I_diri] = V_diri
        @test b == expected
        if b isa SparseVector
            @test SparseArrays.nonzeroinds(b)[positions] == I_diri
            @test signbit(b[1])
            stored_indices, stored_values = SparseArrays.nonzeroinds(b), nonzeros(b)
            @test Bcube.prepare_dirichlet_vector!(b, I_diri, position_by_dof, backend) ==
                  positions
            @test SparseArrays.nonzeroinds(b) === stored_indices
            @test nonzeros(b) === stored_values
        end
    end
    mesh = rectangle_mesh(3, 3)
    U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
    b = Bcube.allocate_dirichlet_sparse_vector(U, I_diri, Float64)
    @test nnz(b) == length(I_diri)
    @test Set(SparseArrays.nonzeroinds(b)) == Set(I_diri)
    @test all(iszero, nonzeros(b))

    A = spdiagm(1 => ones(7))
    positions = Bcube.prepare_dirichlet_matrix!(A, I_diri, backend)
    @test rowvals(A)[positions] == I_diri
    Bcube.set_dirichlet_diagonal!(A, positions, 3.0, backend)
    @test diag(A)[I_diri] == fill(3.0, length(I_diri))
    @test diag(A, 1) == ones(7)
end

@testset "Prepared Dirichlet boundary domains" begin
    mesh = rectangle_mesh(4, 3)
    value_calls = Ref(0)
    boundary_value = t -> begin
        value_calls[] += 1
        PhysicalFunction(x -> 2*t + x[1] + 3*x[2])
    end
    scalar = TrialFESpace(
        FunctionSpace(:Lagrange, 1),
        mesh,
        Dict("xmin" => boundary_value, "ymin" => boundary_value),
    )
    velocity = TrialFESpace(
        FunctionSpace(:Lagrange, 2),
        mesh,
        Dict("xmax" => PhysicalFunction(x -> [x[1], x[2]]));
        size = 2,
    )
    for U in (scalar, MultiFESpace(scalar, velocity))
        V = if U isa Bcube.AbstractMultiFESpace
            MultiFESpace(map(TestFESpace, Tuple(U))...)
        else
            TestFESpace(U)
        end
        boundaries = Bcube.prepare_dirichlet_boundaries(U, mesh)
        domains = [domain for domains in boundaries for domain in domains]
        @test all(domain isa BoundaryFaceDomain for domain in domains)
        trial_fields = U isa Bcube.AbstractMultiFESpace ? Tuple(U) : (U,)
        for (trial, field_domains) in zip(trial_fields, boundaries)
            @test length(field_domains) == length(Bcube.get_dirichlet_boundary_tags(trial))
        end
        cache_arrays = [Bcube.get_cache(domain) for domain in domains]
        backend = Bcube.get_bcube_backend(mesh)
        count_before = value_calls[]
        occurrences = Bcube.collect_dirichlet_indices(U, V, backend, boundaries)
        second_occurrences = Bcube.collect_dirichlet_indices(U, V, backend, boundaries)
        @test occurrences == second_occurrences
        @test occurrences !== second_occurrences
        @test value_calls[] == count_before
        I_diri, inverse = Bcube.dirichlet_data(U, V, mesh, boundaries)
        @test I_diri == unique(occurrences)
        @test Bcube.dirichlet_data(U, V, backend, boundaries) ==
              (I_diri, inverse)
        for t in (0.0, 2.0)
            values = zeros(length(I_diri))
            Bcube.evaluate_dirichlet_values!(
                values,
                inverse,
                U,
                V,
                boundaries,
                Bcube.BcubeBackendCPUSerial(),
                t,
            )
            @test values == Bcube.assemble_dirichlet_vector(U, V, mesh, t)[I_diri]
        end
        @test [domain for domains in boundaries for domain in domains] == domains
        @test all(
            Bcube.get_cache(domain) === cache for
            (domain, cache) in zip(domains, cache_arrays)
        )
    end
end

# This CSC implementation deliberately exposes no .colptr field. The algorithms
# must use the abstract storage interface rather than the concrete representation.
struct ReviewCSC{Tv, Ti <: Integer} <: SparseArrays.AbstractSparseMatrixCSC{Tv, Ti}
    storage::SparseMatrixCSC{Tv, Ti}
end
Base.size(A::ReviewCSC) = size(A.storage)
Base.getindex(A::ReviewCSC, row::Int, column::Int) = A.storage[row, column]
SparseArrays.rowvals(A::ReviewCSC) = rowvals(A.storage)
SparseArrays.nonzeros(A::ReviewCSC) = nonzeros(A.storage)
SparseArrays.nzrange(A::ReviewCSC, column::Integer) = nzrange(A.storage, column)
SparseArrays.getcolptr(A::ReviewCSC) = SparseArrays.getcolptr(A.storage)
function Base.copyto!(A::ReviewCSC, B::SparseArrays.AbstractSparseMatrixCSC)
    copyto!(A.storage, B)
    return A
end

@testset "Abstract CSC and typed Dirichlet allocation" begin
    mesh = rectangle_mesh(3, 2)
    U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, Dict("xmin" => 1.5))
    V = TestFESpace(U)
    for T in (Float32, Float64)
        result = Bcube.assemble_dirichlet_sparse_vector(U, V, mesh; T)
        @test eltype(result) == T
        @test nonzeros(result) == fill(T(1.5), nnz(result))
    end
    n = get_ndofs(U)
    boundaries = Bcube.prepare_dirichlet_boundaries(U, mesh)
    indices, _ = Bcube.dirichlet_data(U, V, mesh, boundaries)
    for apply! in
        (Bcube.apply_dirichlet_to_matrix!, Bcube.apply_homogeneous_dirichlet_to_matrix!)
        reference = spdiagm(1 => ones(n-1))
        abstract_csc = ReviewCSC(copy(reference))
        apply!(reference, U, V, mesh; diag_value = 2.5)
        apply!(abstract_csc, U, V, mesh; diag_value = 2.5)
        @test abstract_csc.storage == reference
        @test diag(abstract_csc.storage)[indices] == fill(2.5, length(indices))
    end
end

@testset "Domain order determines the correct boundary function" begin
    mesh = rectangle_mesh(4, 3)
    fs = FunctionSpace(:Lagrange, 1)
    # These traces agree at the corner but differ away from their own boundaries.
    U = TrialFESpace(
        fs,
        mesh,
        Dict(
            "xmin" => t -> PhysicalFunction(x -> 2*t + 3*x[2]),
            "ymin" => t -> PhysicalFunction(x -> 2*t + x[1]),
        ),
    )
    V = TestFESpace(U)
    boundaries = Bcube.prepare_dirichlet_boundaries(U, mesh)
    indices, inverse = Bcube.dirichlet_data(U, V, mesh, boundaries)
    for t in (0.0, 2.0)
        reference = FEFunction(
            TrialFESpace(fs, mesh),
            mesh,
            PhysicalFunction(x -> 2*t + x[1] + 3*x[2]),
        )
        values = zeros(length(indices))
        Bcube.evaluate_dirichlet_values!(
            values,
            inverse,
            U,
            V,
            boundaries,
            Bcube.BcubeBackendCPUSerial(),
            t,
        )
        @test values == get_dof_values(reference)[indices]
    end
end

@testset "Dirichlet occurrences on mixed face types" begin
    # One pyramid has a quadrilateral face and four triangular faces.
    mesh = Bcube.Mesh(
        Bcube.get_nodes(one_cell_mesh(:pyra)),
        [Bcube.Pyra5_t()],
        Bcube.Connectivity([5], collect(1:5));
        bc_names = Dict(1 => "boundary"),
        bc_nodes = Dict(1 => collect(1:5)),
    )
    U = TrialFESpace(
        FunctionSpace(:Lagrange, 1),
        mesh,
        Dict("boundary" => [1.0, 2.0]);
        size = 2,
    )
    V = TestFESpace(U)
    boundaries = Bcube.prepare_dirichlet_boundaries(U, mesh)
    backend = Bcube.get_bcube_backend(mesh)
    dhl = Bcube.get_dhl(V)

    # Face incidences in geometric order give an independent topology oracle.
    face_nodes = [1, 4, 3, 2, 1, 2, 5, 2, 3, 5, 3, 4, 5, 4, 1, 5]
    expected = [
        Bcube.get_dof(dhl, 1, component, node) for
        node in face_nodes for component in 1:2
    ]
    occurrences = Bcube.collect_dirichlet_indices(U, V, backend, boundaries)
    @test length(Bcube.get_subdomains(only(only(boundaries)))) == 2
    @test length(occurrences) == 32
    @test occurrences == expected
end
