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
        vals = compute(integ)
        @test vals[1] == 0.5
        @test vals[2] == 1.0

        # tensor data
        # data = [[1,2,3,4], [5,6,7,8], ...]
        data = collect(reinterpret(SVector{4, Int}, collect(1:(4 * ncells(mesh)))))
        d = Bcube.MeshCellData(data)
        integ = ∫(d ⋅ [0, 1, 0, 0])dΩ
        @test Bcube.compute(integ) == [1.0, 3.0]
        vals = compute(integ)
        @test vals[1] == 1.0
        @test vals[2] == 3.0

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
    end
end
