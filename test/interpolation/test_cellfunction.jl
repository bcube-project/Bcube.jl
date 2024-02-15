import Bcube: Mesh

function test_CellPoint(cellinfo, x_ref, x_phys)
    p_ref = Bcube.CellPoint(x_ref, cellinfo, Bcube.ReferenceDomain())
    @test Bcube.change_domain(p_ref, Bcube.ReferenceDomain()) == p_ref

    p_phys = Bcube.change_domain(p_ref, Bcube.PhysicalDomain())
    @test p_phys == Bcube.CellPoint(x_phys, cellinfo, Bcube.PhysicalDomain())

    @test Bcube.change_domain(p_phys, Bcube.PhysicalDomain()) == p_phys

    p_ref_bis = Bcube.change_domain(p_phys, Bcube.ReferenceDomain())
    @test p_ref_bis == p_ref

    nothing
end

@testset "CellFunction" begin
    lx, ly = 20.0, 10.0
    mesh = one_cell_mesh(:quad; xmin = 0.0, xmax = lx, ymin = 0.0, ymax = ly)
    x = [lx / 2, ly / 2]
    cellinfo = CellInfo(mesh, 1)

    x1_ref  = SA[0.0, 0.0]
    x1_phys = SA[lx / 2, ly / 2]

    x2_ref  = SA[-3 / 4, 1.0]
    x2_phys = SA[lx / 8, ly]

    @testset "CellPoint" begin
        @testset "One point" begin
            test_CellPoint(cellinfo, x1_ref, x1_phys)
            test_CellPoint(cellinfo, x2_ref, x2_phys)
        end
        @testset "Tuple of points" begin
            test_CellPoint(cellinfo, (x1_ref, x2_ref), (x1_phys, x2_phys))
        end
        @testset "Vector of points" begin
            test_CellPoint(cellinfo, SA[x1_ref, x2_ref], SA[x1_phys, x2_phys])
        end
    end

    @testset "PhysicalFunction" begin
        a = Bcube.CellPoint(SA[x1_ref, x2_ref], cellinfo, Bcube.ReferenceDomain())
        b = Bcube.CellPoint(SA[x1_phys, x2_phys], cellinfo, Bcube.PhysicalDomain())
        _f = x -> x[1] + x[2]
        f = PhysicalFunction(_f)
        @test f(a) == _f.(SA[x1_phys, x2_phys])
        @test f(b) == _f.(SA[x1_phys, x2_phys])

        a = Bcube.CellPoint((x1_ref, x2_ref), cellinfo, Bcube.ReferenceDomain())
        b = Bcube.CellPoint((x1_phys, x2_phys), cellinfo, Bcube.PhysicalDomain())
        @test f(a) == _f.((x1_phys, x2_phys))
        @test f(b) == _f.((x1_phys, x2_phys))
    end

    @testset "CellFunction" begin
        a = Bcube.CellPoint((x1_ref, x2_ref), cellinfo, Bcube.ReferenceDomain())
        b = Bcube.CellPoint((x1_phys, x2_phys), cellinfo, Bcube.PhysicalDomain())
        _f = x -> x[1] + x[2]

        λref = Bcube.CellFunction(_f, Bcube.ReferenceDomain(), Val(1))
        @test λref(a) == _f.((x1_ref, x2_ref))

        λphys = Bcube.CellFunction(_f, Bcube.PhysicalDomain(), Val(1))
        @test λphys(a) == _f.((x1_phys, x2_phys))
    end
end

@testset "CoplanarRotation" begin
    @testset "Topodim = 1" begin
        cross_2D(a, b) = a[1] * b[2] - a[2] * b[1]

        #--- 1
        A = Node([0, 0])
        B = Node([1, 1])
        C = Node([2, 0])
        _nodes = [A, B, C]
        ctypes = [Bar2_t(), Bar2_t()]
        cell2nodes = [1, 2, 2, 3]
        cell2nnodes = [2, 2]

        mesh = Mesh(_nodes, ctypes, Connectivity(cell2nnodes, cell2nodes))

        fInfo = Bcube.FaceInfo(mesh, 2)
        fPoint = Bcube.FacePoint([0.0], fInfo, Bcube.ReferenceDomain())

        R = Bcube.CoplanarRotation()

        _R = Bcube.materialize(side_n(R), fPoint)
        u1 = normalize(Bcube.coords(B) - Bcube.coords(A))
        v2 = normalize(Bcube.coords(C) - Bcube.coords(B)) * 2
        v2_in_1 = _R * v2
        @test abs(cross_2D(u1, v2_in_1)) < 1e-15
        @test norm(v2_in_1) ≈ norm(v2)

        _R = Bcube.materialize(side_p(R), fPoint)
        u2 = normalize(Bcube.coords(C) - Bcube.coords(B))
        v1 = normalize(Bcube.coords(B) - Bcube.coords(A)) * 2
        v1_in_2 = _R * v1
        @test abs(cross_2D(u2, v1_in_2)) < 1e-15
        @test norm(v1_in_2) ≈ norm(v1)

        #--- 2
        A = Node([0, 0])
        B = Node([1, 1])
        C = Node([2, 1])
        _nodes = [A, B, C]
        ctypes = [Bar2_t(), Bar2_t()]
        cell2nodes = [1, 2, 2, 3]
        cell2nnodes = [2, 2]

        mesh = Mesh(_nodes, ctypes, Connectivity(cell2nnodes, cell2nodes))

        fInfo = Bcube.FaceInfo(mesh, 2)
        fPoint = Bcube.FacePoint([0.0], fInfo, Bcube.ReferenceDomain())

        R = Bcube.CoplanarRotation()

        _R = Bcube.materialize(side_n(R), fPoint)
        u1 = normalize(Bcube.coords(B) - Bcube.coords(A))
        v2 = normalize(Bcube.coords(C) - Bcube.coords(B)) * 2
        v2_in_1 = _R * v2
        @test abs(cross_2D(u1, v2_in_1)) < 1e-15
        @test norm(v2_in_1) ≈ norm(v2)

        _R = Bcube.materialize(side_p(R), fPoint)
        u2 = normalize(Bcube.coords(C) - Bcube.coords(B))
        v1 = normalize(Bcube.coords(B) - Bcube.coords(A)) * 2
        v1_in_2 = _R * v1
        @test abs(cross_2D(u2, v1_in_2)) < 1e-15
        @test norm(v1_in_2) ≈ norm(v1)

        #--- 3
        A = Node([0, 0])
        B = Node([1, 1])
        C = Node([1, 2])
        _nodes = [A, B, C]
        ctypes = [Bar2_t(), Bar2_t()]
        cell2nodes = [1, 2, 2, 3]
        cell2nnodes = [2, 2]

        mesh = Mesh(_nodes, ctypes, Connectivity(cell2nnodes, cell2nodes))

        fInfo = Bcube.FaceInfo(mesh, 2)
        fPoint = Bcube.FacePoint([0.0], fInfo, Bcube.ReferenceDomain())

        R = Bcube.CoplanarRotation()

        _R = Bcube.materialize(side_n(R), fPoint)
        u1 = normalize(Bcube.coords(B) - Bcube.coords(A))
        v2 = -normalize(Bcube.coords(C) - Bcube.coords(B)) * 2
        v2_in_1 = _R * v2
        @test abs(cross_2D(u1, v2_in_1)) < 1e-15
        @test norm(v2_in_1) ≈ norm(v2)

        _R = Bcube.materialize(side_p(R), fPoint)
        u2 = normalize(Bcube.coords(C) - Bcube.coords(B))
        v1 = -normalize(Bcube.coords(B) - Bcube.coords(A)) * 2
        v1_in_2 = _R * v1
        @test abs(cross_2D(u2, v1_in_2)) < 1e-15
        @test norm(v1_in_2) ≈ norm(v1)

        #--- 4 (full planar)
        A = Node([0, 1])
        B = Node([1, 1])
        C = Node([2, 1])
        _nodes = [A, B, C]
        ctypes = [Bar2_t(), Bar2_t()]
        cell2nodes = [1, 2, 2, 3]
        cell2nnodes = [2, 2]

        mesh = Mesh(_nodes, ctypes, Connectivity(cell2nnodes, cell2nodes))

        fInfo = Bcube.FaceInfo(mesh, 2)
        fPoint = Bcube.FacePoint([0.0], fInfo, Bcube.ReferenceDomain())

        R = Bcube.CoplanarRotation()

        _R = Bcube.materialize(side_n(R), fPoint)
        u1 = normalize(Bcube.coords(B) - Bcube.coords(A))
        v2 = -normalize(Bcube.coords(C) - Bcube.coords(B)) * 2
        v2_in_1 = _R * v2
        @test abs(cross_2D(u1, v2_in_1)) < 1e-15
        @test norm(v2_in_1) ≈ norm(v2)

        _R = Bcube.materialize(side_p(R), fPoint)
        u2 = normalize(Bcube.coords(C) - Bcube.coords(B))
        v1 = -normalize(Bcube.coords(B) - Bcube.coords(A)) * 2
        v1_in_2 = _R * v1
        @test abs(cross_2D(u2, v1_in_2)) < 1e-15
        @test norm(v1_in_2) ≈ norm(v1)
    end

    @testset "Topodim = 2" begin
        #--- 1
        A = Node([0, 0, 0])
        B = Node([1, 0, 0])
        C = Node([1, 1, 0])
        D = Node([0, 1, 0])
        E = Node([3, 0, 2])
        F = Node([3, 1, 2])

        _nodes = [A, B, C, D, E, F]
        ctypes = [Quad4_t(), Quad4_t()]
        cell2nodes = [1, 2, 3, 4, 2, 5, 6, 3]
        cell2nnodes = [4, 4]

        mesh = Mesh(_nodes, ctypes, Connectivity(cell2nnodes, cell2nodes))

        fInfo = Bcube.FaceInfo(mesh, 2) # 2 is the good one, cf `Bcube.get_nodes_index(fInfo)`
        fPoint = Bcube.FacePoint([0.0, 0.0], fInfo, Bcube.ReferenceDomain())
        cInfo_n = Bcube.get_cellinfo_n(fInfo)
        cnodes_n = Bcube.nodes(cInfo_n)
        ctype_n = Bcube.celltype(cInfo_n)
        ξ_n = Bcube.get_coord(side_n(fPoint))
        cInfo_p = Bcube.get_cellinfo_p(fInfo)
        cnodes_p = Bcube.nodes(cInfo_p)
        ctype_p = Bcube.celltype(cInfo_p)
        ξ_p = Bcube.get_coord(side_p(fPoint))

        R = Bcube.CoplanarRotation()

        _R = Bcube.materialize(side_n(R), fPoint)
        v2 = normalize(Bcube.coords(F) - Bcube.coords(E)) * 2
        u = normalize(Bcube.coords(C) - Bcube.coords(B))
        v2_in_1 = _R * v2
        ν1 = Bcube.cell_normal(ctype_n, cnodes_n, ξ_n)
        @test v2 ⋅ u ≈ v2_in_1 ⋅ u
        @test abs(ν1 ⋅ v2_in_1) < 1e-16

        _R = Bcube.materialize(side_p(R), fPoint)
        v1 = normalize(Bcube.coords(D) - Bcube.coords(B)) * 2
        u = normalize(Bcube.coords(C) - Bcube.coords(B))
        v1_in_2 = _R * v1
        ν2 = Bcube.cell_normal(ctype_p, cnodes_p, ξ_p)
        @test v1 ⋅ u ≈ v1_in_2 ⋅ u
        @test abs(ν2 ⋅ v1_in_2) < 1e-16

        #--- 2
        A = Node([0.0, 0.0, 0.0])
        B = Node([1.0, 0.0, 0.0])
        C = Node([1.0, 1.0, 0.0])
        D = Node([0.0, 1.0, 0.0])
        E = Node([3.0, 0.0, 2.0])
        F = Node([3.0, 1.0, 2.0])

        _nodes = [A, B, C, D, E, F]
        ctypes = [Quad4_t(), Quad4_t()]
        cell2nodes = [1, 2, 3, 4, 2, 5, 6, 3]
        cell2nnodes = [4, 4]

        mesh = Mesh(_nodes, ctypes, Connectivity(cell2nnodes, cell2nodes))
        axis = [1, 2, 3]
        θ = π / 3
        rot = (
            cos(θ) * I +
            sin(θ) * [0 (-axis[3]) axis[2]; axis[3] 0 (-axis[1]); -axis[2] axis[1] 0] +
            (1 - cos(θ)) * (axis ⊗ axis)
        )
        transform!(mesh, x -> rot * x)

        fInfo = Bcube.FaceInfo(mesh, 2) # 2 is the good one, cf `Bcube.get_nodes_index(fInfo)`
        fPoint = Bcube.FacePoint([0.0, 0.0], fInfo, Bcube.ReferenceDomain())
        cInfo_n = Bcube.get_cellinfo_n(fInfo)
        cnodes_n = Bcube.nodes(cInfo_n)
        ctype_n = Bcube.celltype(cInfo_n)
        ξ_n = Bcube.get_coord(side_n(fPoint))
        cInfo_p = Bcube.get_cellinfo_p(fInfo)
        cnodes_p = Bcube.nodes(cInfo_p)
        ctype_p = Bcube.celltype(cInfo_p)
        ξ_p = Bcube.get_coord(side_p(fPoint))

        R = Bcube.CoplanarRotation()

        _R = Bcube.materialize(side_n(R), fPoint)
        v2 = rot * normalize(Bcube.coords(F) - Bcube.coords(E)) * 2
        u = rot * normalize(Bcube.coords(C) - Bcube.coords(B))
        v2_in_1 = _R * v2
        ν1 = Bcube.cell_normal(ctype_n, cnodes_n, ξ_n)
        @test v2 ⋅ u ≈ v2_in_1 ⋅ u
        @test abs(ν1 ⋅ v2_in_1) < 2e-15

        _R = Bcube.materialize(side_p(R), fPoint)
        v1 = rot * normalize(Bcube.coords(D) - Bcube.coords(B)) * 2
        u = rot * normalize(Bcube.coords(C) - Bcube.coords(B))
        v1_in_2 = _R * v1
        ν2 = Bcube.cell_normal(ctype_p, cnodes_p, ξ_p)
        @test v1 ⋅ u ≈ v1_in_2 ⋅ u
        @test abs(ν2 ⋅ v1_in_2) < 1e-16
    end
end
