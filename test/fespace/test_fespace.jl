@testset "FESpace" begin
    @testset "Misc." begin
        mesh = one_cell_mesh(:line)
        U = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, Dict("xmin" => 3.0); size = 2)
        @test Bcube.is_continuous(U)
        @test !Bcube.is_discontinuous(U)
        @test Bcube.get_ncomponents(U) == 2
        bctag = first(Bcube.get_dirichlet_boundary_tags(U))
        cInfo = Bcube.CellInfo(mesh, 1)
        cPoint = Bcube.CellPoint([0.0], cInfo, Bcube.ReferenceDomain())
        f_diri_t = Bcube.get_dirichlet_values(U, bctag)(0.0)
        @test Bcube.materialize(f_diri_t, cPoint) == 3.0
    end

    @testset "Sparsity pattern" begin
        # The test consists in assembling the ~most complex bilinear form possible
        # and to check that the non-zeros elements are all in the "built" sparsity
        # pattern.
        mesh = rectangle_mesh(9, 4)
        dΩ = Measure(CellDomain(mesh), 3)
        U1 = TrialFESpace(FunctionSpace(:Lagrange, 2), mesh)
        V1 = TestFESpace(U1)
        U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
        V2 = TestFESpace(U2)
        U = MultiFESpace(U1, U2)
        V = MultiFESpace(V1, V2)
        a1((u1, u2), (v1, v2)) = ∫(u1 * v1 + u1 * v2 + u2 * v1 + u2 * v2)dΩ
        A = assemble_bilinear(a1, U, V)
        A.nzval .= 1.0
        J = Bcube.build_jacobian_sparsity_pattern(U, mesh)
        D = J - A
        @test all(D.nzval .> 0.0) # test that non-zeros elements of "A" are included in "J"

        function f(q, ∇q, v, ∇v)
            u1, u2 = q
            ∇u1, ∇u2 = ∇q
            v1, v2 = v
            ∇v1, ∇v2 = ∇v
            return u1 * v1 + u2 ⋅ v2 + u2 ⋅ ∇v1 + ∇u1 ⋅ v2
        end
        mesh = rectangle_mesh(3, 4)
        dΩ = Measure(CellDomain(mesh), 3)
        U1 = TrialFESpace(FunctionSpace(:Lagrange, 2), mesh)
        V1 = TestFESpace(U1)
        U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh; size = 2)
        V2 = TestFESpace(U2)
        U = MultiFESpace(U1, U2)
        V = MultiFESpace(V1, V2)
        a2(q, v) = ∫(f ∘ (q, map(∇, q), v, map(∇, v)))dΩ
        A = assemble_bilinear(a2, U, V)
        A.nzval .= 1.0
        J = Bcube.build_jacobian_sparsity_pattern(U, mesh)
        D = J - A
        @test all(D.nzval .> 0.0) # test that non-zeros elements of "A" are included in "J"
    end

    @testset "Dof numbering" begin
        # Geometrically check that, for a mesh of two hexa side by side, the dof numbering
        # is correct (up to degree 5). Note that for degree ≤ 2, the numbering is built topologically
        # only, and for degree ≥ 3 it is built geometrically
        mesh = hexa_mesh(3, 2, 2; xmax = 2, zmax = 0.5)
        max_degree = VERSION ≥ v"1.12" ? 5 : 2 # TODO on Julia 1.10, degrees ≥ 3 takes forever to build...
        @test all(
            degree -> begin
                fs = FunctionSpace(:Lagrange, degree)
                U = TrialFESpace(fs, mesh)
                space = parent(U)
                res = Bcube.check_numbering(space, mesh; exit_on_error = false)
                return res.n_errors == 0
            end,
            0:max_degree,
        )
        # Same but in 2D for the "basic mesh"
        mesh = Bcube.basic_mesh()
        @test all(
            degree -> begin
                fs = FunctionSpace(:Lagrange, degree)
                U = TrialFESpace(fs, mesh)
                space = parent(U)
                res = Bcube.check_numbering(space, mesh; exit_on_error = false)
                return res.n_errors == 0
            end,
            0:3,
        )
    end
    @testset "Sparsity pattern with CellDomain" begin
        # Feature: build_jacobian_sparsity_pattern accepts a Tuple of CellDomain
        # (one per FESpace). Coupling on a cell pair (ic, ic2) requires connectivity
        # AND ic ∈ CD_i AND ic2 ∈ CD_j. Adjacency is computed on the FULL mesh.

        # Helper used throughout: a sparse pattern "covers" an assembled matrix iff
        # every assembled nonzero is present in the pattern (J - A).nzval .> 0.
        covers(J, A) = all((J - A).nzval .> 0.0)

        # ---------------------------------------------------------------------
        # 1. Backward compatibility: mesh form == tuple-of-full-CellDomain form
        # ---------------------------------------------------------------------
        @testset "backward compat equivalence" begin
            mesh = rectangle_mesh(5, 3)
            U1 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
            U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh; size = 2)
            U = MultiFESpace(U1, U2)
            Jmesh = Bcube.build_jacobian_sparsity_pattern(U, mesh)
            Jdom = Bcube.build_jacobian_sparsity_pattern(
                U,
                (CellDomain(mesh), CellDomain(mesh)),
            )
            # Same nonzero structure (positions), values may differ by dup-sum but
            # both are >0 on the same sparsity set.
            @test size(Jmesh) == size(Jdom)
            @test nnz(Jmesh) == nnz(Jdom)
            @test findall(!iszero, Jmesh) == findall(!iszero, Jdom)
            @test findall(!iszero, Jdom) == findall(!iszero, Jmesh)
        end

        # ---------------------------------------------------------------------
        # 2. Length mismatch must error with an ArgumentError (a deliberate
        #    argument check, not an incidental dispatch/crash failure).
        # ---------------------------------------------------------------------
        @testset "length mismatch errors" begin
            mesh = rectangle_mesh(3, 3)
            U1 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
            U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
            U = MultiFESpace(U1, U2)
            # Too few domains
            @test_throws ArgumentError Bcube.build_jacobian_sparsity_pattern(
                U,
                (CellDomain(mesh),),
            )
            # Too many domains
            @test_throws ArgumentError Bcube.build_jacobian_sparsity_pattern(
                U,
                (CellDomain(mesh), CellDomain(mesh), CellDomain(mesh)),
            )
            # Single FESpace with wrong number of domains
            @test_throws ArgumentError Bcube.build_jacobian_sparsity_pattern(
                U1,
                (CellDomain(mesh), CellDomain(mesh)),
            )
        end

        # ---------------------------------------------------------------------
        # 3. Disjoint, non-touching domains => block-diagonal, no cross-coupling
        # ---------------------------------------------------------------------
        @testset "disjoint non-touching domains" begin
            # Two quads far apart, sharing neither node nor face.
            nodes = [
                Node(SA[0.0, 0.0]),
                Node(SA[1.0, 0.0]),
                Node(SA[1.0, 1.0]),
                Node(SA[0.0, 1.0]),
                Node(SA[10.0, 0.0]),
                Node(SA[11.0, 0.0]),
                Node(SA[11.0, 1.0]),
                Node(SA[10.0, 1.0]),
            ]
            celltypes = fill(Quad4_t(), 2)
            c2n = Connectivity([4, 4], [1, 2, 3, 4, 5, 6, 7, 8])
            mesh = Mesh(nodes, celltypes, c2n)
            # Sanity: cells are fully disconnected (no node neighbors)
            c2c = Bcube.connectivity_cell2cell_by_nodes(mesh)
            @test isempty(c2c[1]) && isempty(c2c[2])

            U1 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
            U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
            U = MultiFESpace(U1, U2)
            V = MultiFESpace(TestFESpace(U1), TestFESpace(U2))
            cdL = CellDomain(mesh, [1])
            cdR = CellDomain(mesh, [2])
            @test indices(cdL) == [1]
            @test indices(cdR) == [2]

            J = Bcube.build_jacobian_sparsity_pattern(U, (cdL, cdR))
            nd1 = Bcube.get_ndofs(U1)
            nd2 = Bcube.get_ndofs(U2)
            nd = Bcube.get_ndofs(U)

            # Mapping is AoS: variable 1 occupies global dofs 1:nd1, variable 2
            # occupies global dofs (nd1+1):nd. With disjoint domains and no
            # adjacency, there must be NO off-diagonal cross block coupling.
            # Cross blocks: rows (var1) vs cols (var2) and vice-versa.
            cross12 = J[1:nd1, (nd1 + 1):nd]
            cross21 = J[(nd1 + 1):nd, 1:nd1]
            @test nnz(cross12) == 0
            @test nnz(cross21) == 0
            # Each diagonal block must be non-empty (on-cell coupling present
            # within each variable on its own cell).
            @test nnz(J[1:nd1, 1:nd1]) > 0
            @test nnz(J[(nd1 + 1):nd, (nd1 + 1):nd]) > 0

            # Strictly sparser than the (over-conservative) full-mesh pattern,
            # which today couples cell1-var1 with cell2-var2 across variables.
            Jfull = Bcube.build_jacobian_sparsity_pattern(U, mesh)
            @test nnz(J) < nnz(Jfull)

            # Coverage: a form coupling each variable to itself, integrated ONLY where
            # the variable lives, must be fully contained in J.
            dΩL = Measure(cdL, 2)
            dΩR = Measure(cdR, 2)
            a((u1, u2), (v1, v2)) = ∫(u1 * v1)dΩL + ∫(u2 * v2)dΩR
            A = assemble_bilinear(a, U, V)
            A.nzval .= 1.0
            @test covers(J, A)
        end

        # ---------------------------------------------------------------------
        # 4. Adjacent domains sharing a node column => interface cross-coupling
        # ---------------------------------------------------------------------
        @testset "adjacent domains share interface vertices" begin
            # 5x3-node quad mesh => 4x2 = 8 cells. Split into left {1,2,5,6} and
            # right {3,4,7,8}, which share the node column (nodes 3,8,13).
            mesh = rectangle_mesh(5, 3)
            ncells_mesh = Bcube.ncells(mesh)
            @test ncells_mesh == 8
            left = [1, 2, 5, 6]
            right = [3, 4, 7, 8]
            cdL = CellDomain(mesh, left)
            cdR = CellDomain(mesh, right)
            @test indices(cdL) == left
            @test indices(cdR) == right
            # Disjoint cell sets
            @test isempty(intersect(Set(left), Set(right)))
            # But node-adjacency reaches across the interface (cells share nodes
            # 3,8,13): e.g. cell 2 neighbors include cell 3 (right side).
            c2c = Bcube.connectivity_cell2cell_by_nodes(mesh)
            @test 3 ∈ c2c[2]
            @test 7 ∈ c2c[2]
            @test 2 ∈ c2c[3]

            U1 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
            U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
            U = MultiFESpace(U1, U2)
            V = MultiFESpace(TestFESpace(U1), TestFESpace(U2))

            J = Bcube.build_jacobian_sparsity_pattern(U, (cdL, cdR))
            nd1 = Bcube.get_ndofs(U1)
            nd2 = Bcube.get_ndofs(U2)
            nd = Bcube.get_ndofs(U)

            # Cross coupling MUST exist at the interface (shared vertices honored).
            @test nnz(J[1:nd1, (nd1 + 1):nd]) > 0
            @test nnz(J[(nd1 + 1):nd, 1:nd1]) > 0

            # Strictly sparser than the full-mesh pattern.
            Jfull = Bcube.build_jacobian_sparsity_pattern(U, mesh)
            @test nnz(J) < nnz(Jfull)

            # Strictly denser than the disjoint-domains pattern would be: i.e. the
            # cross block is non-empty, whereas for truly disjoint non-touching
            # domains it would be empty (asserted explicitly in case 3).

            # Coverage: self-coupling of each variable, integrated only where the
            # variable lives, must be fully contained in J. (The cross-block
            # entries produced by the shared-vertex neighbor reach are
            # conservative padding for continuous FESpaces; they are a superset of
            # any real on-cell assembly, so we do not try to "cover" them with a
            # cross term. Their existence is asserted above.)
            dΩL = Measure(cdL, 2)
            dΩR = Measure(cdR, 2)
            a((u1, u2), (v1, v2)) = ∫(u1 * v1)dΩL + ∫(u2 * v2)dΩR
            A = assemble_bilinear(a, U, V)
            A.nzval .= 1.0
            @test covers(J, A)
        end

        # ---------------------------------------------------------------------
        # 5. Same-domain multi-FESpace == full-mesh pattern
        # ---------------------------------------------------------------------
        @testset "same domain equivalence" begin
            mesh = rectangle_mesh(4, 3)
            U1 = TrialFESpace(FunctionSpace(:Lagrange, 2), mesh)
            U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh; size = 2)
            U = MultiFESpace(U1, U2)
            full = CellDomain(mesh)
            Jdom = Bcube.build_jacobian_sparsity_pattern(U, (full, full))
            Jmesh = Bcube.build_jacobian_sparsity_pattern(U, mesh)
            @test size(Jdom) == size(Jmesh)
            @test findall(!iszero, Jdom) == findall(!iszero, Jmesh)

            # Also for a single TrialFESpace (single CellDomain).
            Jsingle = Bcube.build_jacobian_sparsity_pattern(U1, (full,))
            JsingleMesh = Bcube.build_jacobian_sparsity_pattern(U1, mesh)
            @test findall(!iszero, Jsingle) == findall(!iszero, JsingleMesh)
        end

        # ---------------------------------------------------------------------
        # 6. All-discontinuous FESpaces on adjacent domains (exercises the
        #    by-faces neighbor branch, which no earlier case reaches since all
        #    prior FESpaces are continuous => by-nodes branch).
        # ---------------------------------------------------------------------
        @testset "all-discontinuous adjacent domains" begin
            mesh = rectangle_mesh(5, 3)
            left = [1, 2, 5, 6]
            right = [3, 4, 7, 8]
            cdL = CellDomain(mesh, left)
            cdR = CellDomain(mesh, right)

            U1 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U = MultiFESpace(U1, U2)
            V = MultiFESpace(TestFESpace(U1), TestFESpace(U2))
            @test all(Bcube.is_discontinuous.(U)) # sanity: by-faces branch will run

            J = Bcube.build_jacobian_sparsity_pattern(U, (cdL, cdR))
            nd1 = Bcube.get_ndofs(U1)
            nd = Bcube.get_ndofs(U)

            # Adjacent domains share an interface face, so cross coupling exists.
            @test nnz(J[1:nd1, (nd1 + 1):nd]) > 0
            @test nnz(J[(nd1 + 1):nd, 1:nd1]) > 0
            # Pattern is symmetric.
            @test J == transpose(J)
            # Strictly sparser than full-mesh.
            @test nnz(J) < nnz(Bcube.build_jacobian_sparsity_pattern(U, mesh))
            # Coverage: self-coupling on each own domain.
            dΩL = Measure(cdL, 2)
            dΩR = Measure(cdR, 2)
            a((u1, u2), (v1, v2)) = ∫(u1 * v1)dΩL + ∫(u2 * v2)dΩR
            A = assemble_bilinear(a, U, V)
            A.nzval .= 1.0
            @test covers(J, A)
        end

        # ---------------------------------------------------------------------
        # 7. Mixed full-mesh + sub-domain FESpace (one variable everywhere,
        #    one restricted to a sub-part). Also covers a continuous +
        #    discontinuous mix in one MultiFESpace.
        # ---------------------------------------------------------------------
        @testset "full-mesh and sub-domain mix" begin
            mesh = rectangle_mesh(5, 3)
            left = [1, 2, 5, 6]
            cdL = CellDomain(mesh, left)
            full = CellDomain(mesh)

            # U1 continuous on the whole mesh, U2 discontinuous on left only.
            U1 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
            U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh, :discontinuous)
            U = MultiFESpace(U1, U2)
            V = MultiFESpace(TestFESpace(U1), TestFESpace(U2))
            J = Bcube.build_jacobian_sparsity_pattern(U, (full, cdL))
            nd1 = Bcube.get_ndofs(U1)
            nd = Bcube.get_ndofs(U)

            # U1 is everywhere, U2 only on the left. Cross coupling exists
            # wherever U2 lives (left cells) via both on-cell and neighbor reach.
            @test nnz(J[1:nd1, (nd1 + 1):nd]) > 0
            @test nnz(J[(nd1 + 1):nd, 1:nd1]) > 0
            # Pattern is symmetric.
            @test J == transpose(J)
            # Strictly sparser than full-mesh (U2 restricted).
            @test nnz(J) < nnz(Bcube.build_jacobian_sparsity_pattern(U, mesh))
            # Coverage: U1 self-coupling on the whole mesh + U2 self-coupling on
            # the left must be contained in J.
            dΩ = Measure(full, 2)
            dΩL = Measure(cdL, 2)
            a((u1, u2), (v1, v2)) = ∫(u1 * v1)dΩ + ∫(u2 * v2)dΩL
            A = assemble_bilinear(a, U, V)
            A.nzval .= 1.0
            @test covers(J, A)
        end

        # ---------------------------------------------------------------------
        # 8. Empty CellDomain produces a sane (empty) pattern rather than
        #    crashing on a pre-existing indices() empty-collection bug.
        # ---------------------------------------------------------------------
        @testset "empty CellDomain" begin
            # The AoS dof layout interleaves variables per cell, so "variable 2"
            # dofs are NOT a contiguous global range. Build a layout-independent
            # boolean mask of the global dofs belonging to a given FESpace.
            function var_dof_mask(mesh, u, ifes)
                nd = Bcube.get_ndofs(u)
                mask = falses(nd)
                ui = Bcube.get_fespace(u, ifes)
                for icell in 1:Bcube.ncells(mesh)
                    for d in Bcube.get_mapping(u, ifes)[Bcube.get_dofs(ui, icell)]
                        mask[d] = true
                    end
                end
                return mask
            end

            mesh = rectangle_mesh(3, 3)
            U1 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
            U2 = TrialFESpace(FunctionSpace(:Lagrange, 1), mesh)
            U = MultiFESpace(U1, U2)
            cdEmpty = CellDomain(mesh, Int[])
            cdFull = CellDomain(mesh)
            # Variable 2 on no cells: only variable 1 should couple to itself.
            J = Bcube.build_jacobian_sparsity_pattern(U, (cdFull, cdEmpty))
            mask2 = var_dof_mask(mesh, U, 2)
            # No coupling involving variable 2 at all (rows or columns).
            @test nnz(J[mask2, :]) == 0
            @test nnz(J[:, mask2]) == 0
            # Variable 1 still couples to itself (full mesh).
            mask1 = var_dof_mask(mesh, U, 1)
            @test nnz(J[mask1, mask1]) > 0
        end
    end
end
