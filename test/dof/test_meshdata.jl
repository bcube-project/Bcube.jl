@testset "MeshData" begin
    @testset "CellData" begin
        n = 3
        mesh = line_mesh(n)
        dΩ = Measure(CellDomain(mesh), 2)

        # Scalar data
        data = collect(1:ncells(mesh))
        d = Bcube.MeshCellData(data)
        integ = ∫(d)dΩ
        @test Bcube.compute(integ) == [0.5, 1.0]

        # tensor data
        # data = [[1,2,3,4], [5,6,7,8], ...]
        data = collect(reinterpret(SVector{4, Int}, collect(1:(4 * ncells(mesh)))))
        d = Bcube.MeshCellData(data)
        integ = ∫(d ⋅ [0, 1, 0, 0])dΩ
        @test Bcube.compute(integ) == [1.0, 3.0]
    end
end
