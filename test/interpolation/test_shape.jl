@testset "Shape" begin
    @testset "Line" begin
        line = Line()
        @test nvertices(line) == 2
        @test nedges(line) == 2
        @test nfaces(line) == 2
        @test face_area(line) == SA[1.0, 1.0]
        @test faces2nodes(line) == (SA[1], SA[2])
        @test coords(line) == (SA[-1.0], SA[1.0])
        @test normals(line) == (SA[-1.0], SA[1.0])
    end

    @testset "Triangle" begin
        tri = Triangle()
        @test nvertices(tri) == 3
        @test nedges(tri) == 3
        @test nfaces(tri) == 3
        @test face_area(tri) == SA[1.0, sqrt(2.0), 1.0]
        @test faces2nodes(tri) == (SA[1, 2], SA[2, 3], SA[3, 1])
        @test face_shapes(tri) == (Line(), Line(), Line())
        @test coords(tri) == (SA[0.0, 0.0], SA[1.0, 0.0], SA[0.0, 1.0])
        @test normals(tri) == (SA[0.0, -1.0], SA[1.0, 1.0] ./ √(2), SA[-1.0, 0.0])
    end

    @testset "Square" begin
        square = Square()
        @test nvertices(square) == 4
        @test nedges(square) == 4
        @test nfaces(square) == 4
        @test face_area(square) == SA[2.0, 2.0, 2.0, 2.0]
        @test faces2nodes(square) == (SA[1, 2], SA[2, 3], SA[3, 4], SA[4, 1])
        @test face_shapes(square) == (Line(), Line(), Line(), Line())
        @test coords(square) == (SA[-1.0, -1.0], SA[1.0, -1.0], SA[1.0, 1.0], SA[-1.0, 1.0])
        @test normals(square) == (SA[0.0, -1.0], SA[1.0, 0.0], SA[0.0, 1.0], SA[-1.0, 0.0])
    end
end
