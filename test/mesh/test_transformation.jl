@testset "Transformation" begin
    @testset "Translation" begin
        u = [2, 0]
        T = Translation(u)
        x = [3, 1]
        @test all(T(x) .== [5, 1])
    end

    @testset "Rotation" begin
        rot = Rotation([0, 0, 1], π / 4)
        x = [3.0, 0, 0]
        x_rot_ref = 3√(2) / 2 .* [1, 1, 0]
        @test all(rot(x) .≈ x_rot_ref)
    end
end