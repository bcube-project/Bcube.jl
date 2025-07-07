@testset "issue #101" begin
    function run()
        mesh = line_mesh(4)
        Ω    = CellDomain(mesh)
        dΩ   = Measure(Ω, 2)

        f  = PhysicalFunction(x -> 1.0)
        int1 = ∫(f)dΩ
        int2 = 2.0 * int1

        res  = compute(int1 + int2)
        vols = compute(int1)

        @test maximum(abs.(res .- 3 .* vols)) < 1e-12
    end

    run()
end