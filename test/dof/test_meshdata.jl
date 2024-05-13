@testset "MeshData" begin
    @testset "CellData" begin
        #--- Test 1
        n = 3
        mesh = line_mesh(n)
        dΩ = Measure(CellDomain(mesh), 2)

        # Scalar data
        data = collect(1:ncells(mesh))
        d = Bcube.MeshCellData(data)
        integ = ∫(d)dΩ
        @test compute(integ) == [0.5, 1.0]

        # tensor data
        # data = [[1,2,3,4], [5,6,7,8], ...]
        data = collect(reinterpret(SVector{4, Int}, collect(1:(4 * ncells(mesh)))))
        d = Bcube.MeshCellData(data)
        @test compute(∫(d ⋅ [0, 1, 0, 0])dΩ) == [1.0, 3.0]

        #--- Test 2
        mesh = line_mesh(3)

        array = [1.0, 2.0]
        funcs = [PhysicalFunction(x -> x), PhysicalFunction(x -> 2x)]
        cellArray = MeshCellData(array)
        cellFuncs = MeshCellData(funcs)
        for cInfo in DomainIterator(CellDomain(mesh))
            i = cellindex(cInfo)
            cPointRef = CellPoint([0.0], cInfo, ReferenceDomain())
            cPointPhy = change_domain(cPointRef, PhysicalDomain())

            _cellArray = Bcube.materialize(cellArray, cInfo)
            @test Bcube.materialize(_cellArray, cPointRef) == array[i]

            _cellFuncs = Bcube.materialize(cellFuncs, cInfo)
            @test Bcube.materialize(_cellFuncs, cPointRef) == funcs[i](cPointPhy)
        end

        #--- Test 3, MeshCellData in face integration
        mesh = rectangle_mesh(2, 2; xmin = 0, xmax = 3, ymin = -1, ymax = 1)
        data = MeshCellData(ones(ncells(mesh)))
        dΓ = Measure(BoundaryFaceDomain(mesh), 1)
        values = nonzeros(Bcube.compute(∫(side_n(data))dΓ))
        @test isapprox_arrays(values, [3.0, 2.0, 3.0, 2.0])
    end

    @testset "FaceData" begin
        nodes = [Node([0.0, 0.0]), Node([2.0, 0.0]), Node([3.0, 1.0]), Node([1.0, 2.0])]
        celltypes = [Quad4_t()]
        cell2node = Connectivity([4], [1, 2, 3, 4])
        bnd_name = "BORDER"
        tag2name = Dict(1 => bnd_name)
        tag2nodes = Dict(1 => collect(1:4))

        mesh = Mesh(nodes, celltypes, cell2node; bc_names = tag2name, bc_nodes = tag2nodes)

        f(fnodes) = PhysicalFunction(x -> begin
            norm(x - get_coords(get_nodes(mesh, fnodes[1])))
        end)
        f2n = connectivities_indices(mesh, :f2n)
        D = MeshFaceData([f(fnodes) for fnodes in f2n])

        ∂Ω = Measure(BoundaryFaceDomain(mesh), 3)

        a = compute(∫(side⁻(D))∂Ω)
        l = compute(∫(side⁻(PhysicalFunction(x -> 1.0)))∂Ω)
        @test isapprox_arrays(a, l .^ 2 ./ 2; rtol = 1e-15)
    end
end
