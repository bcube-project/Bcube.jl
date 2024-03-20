@testset "MeshData" begin
    @testset "MeshCellData" begin
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
        vals = compute(integ)
        @test vals[1] == 1.0
        @test vals[2] == 3.0
    end
end
