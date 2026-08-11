@testset "Shape" begin
    @testset "Line" begin
        line = Line()
        @test nvertices(line) == 2
        @test nedges(line) == 2
        @test nfaces(line) == 2
        @test face_area(line) == SA[1.0, 1.0]
        @test faces2nodes(line) == (SA[1], SA[2])
        @test get_coords(line) == (SA[-1.0], SA[1.0])
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
        @test get_coords(tri) == (SA[0.0, 0.0], SA[1.0, 0.0], SA[0.0, 1.0])
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
        @test get_coords(square) ==
              (SA[-1.0, -1.0], SA[1.0, -1.0], SA[1.0, 1.0], SA[-1.0, 1.0])
        @test normals(square) == (SA[0.0, -1.0], SA[1.0, 0.0], SA[0.0, 1.0], SA[-1.0, 0.0])
    end

    @testset "Point" begin
        pt = Point()
        @test topodim(pt) == 0
        @test nvertices(pt) == 1
        @test get_coords(pt) == (SA[0.0],)
        @test entity(Point(), Val(0)) == Node_t()
        @test shape(Node_t()) == Point()
    end

    @testset "Tetra" begin
        tetra = Tetra()
        @test topodim(tetra) == 3
        @test nvertices(tetra) == 4
        @test nedges(tetra) == 6
        @test nfaces(tetra) == 4
        @test get_coords(tetra) ==
              (SA[0.0, 0.0, 0.0], SA[1.0, 0.0, 0.0], SA[0.0, 1.0, 0.0], SA[0.0, 0.0, 1.0])
        @test get_coords(tetra, 1) == SA[0.0, 0.0, 0.0]
        @test get_coords(tetra, (1, 2)) == (SA[0.0, 0.0, 0.0], SA[1.0, 0.0, 0.0])
        @test get_coords(tetra, [1, 2]) isa AbstractVector
        @test get_coords(tetra, [1, 2])[1] == SA[0.0, 0.0, 0.0]
        @test get_coords(tetra, [1, 2])[2] == SA[1.0, 0.0, 0.0]
        @test face_area(tetra) == SA[0.5, 0.5, 0.5 * √(3.0), 0.5]
        @test faces2nodes(tetra) == (SA[1, 3, 2], SA[1, 2, 4], SA[2, 3, 4], SA[3, 1, 4])
        @test faces2nodes(tetra, 1) == SA[1, 3, 2]
        @test faces2nodes(tetra, -1) == reverse(faces2nodes(tetra, 1))
        @test normals(tetra) == (
            SA[0.0, 0.0, -1.0],
            SA[0.0, -1.0, 0.0],
            SA[1.0, 1.0, 1.0] / √3,
            SA[-1.0, 0.0, 0.0],
        )
        @test normal(tetra, 1) == SA[0.0, 0.0, -1.0]
        @test center(tetra) == SA[0.25, 0.25, 0.25]
        @test face_shapes(tetra) == (Triangle(), Triangle(), Triangle(), Triangle())
        @test face_shapes(tetra, 1) == Triangle()
        @test measure(tetra) == 1.0 / 6
        @test shape(Tetra4_t()) == Tetra()
        @test shape(Tetra10_t()) == Tetra()
        @test is_point_in_shape(tetra, SA[0.1, 0.1, 0.1]) == true
        @test is_point_in_shape(tetra, SA[-1.0, 0.0, 0.0]) == false
    end

    @testset "Cube" begin
        cube = Cube()
        @test topodim(cube) == 3
        @test nvertices(cube) == 8
        @test nedges(cube) == 12
        @test nfaces(cube) == 6
        @test face_area(cube) == SA[4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        @test faces2nodes(cube, 1) == SA[1, 4, 3, 2]
        @test faces2nodes(cube, -1) == reverse(faces2nodes(cube, 1))
        @test normals(cube)[1] == SA[0.0, 0.0, -1.0]
        @test normal(cube, 1) == SA[0.0, 0.0, -1.0]
        @test center(cube) == SA[0.0, 0.0, 0.0]
        @test face_shapes(cube) == ntuple(i -> Square(), 6)
        @test face_shapes(cube, 1) == Square()
        @test measure(cube) == 8.0
        @test shape(Hexa8_t()) == Cube()
        @test is_point_in_shape(cube, SA[0.0, 0.0, 0.0]) == true
        @test is_point_in_shape(cube, SA[2.0, 0.0, 0.0]) == false
    end

    @testset "Prism" begin
        prism = Prism()
        @test topodim(prism) == 3
        @test nvertices(prism) == 6
        @test nedges(prism) == 9
        @test nfaces(prism) == 5
        @test face_area(prism) == SA[2.0, 2 * √(2.0), 2.0, 0.5, 0.5]
        @test faces2nodes(prism, 1) == SA[1, 2, 5, 4]
        @test faces2nodes(prism, -1) == reverse(faces2nodes(prism, 1))
        @test normals(prism)[1] == SA[0.0, -1.0, 0.0]
        @test normal(prism, 1) == SA[0.0, -1.0, 0.0]
        @test center(prism) == SA[1.0 / 3.0, 1.0 / 3.0, 0.0]
        @test face_shapes(prism) == (Square(), Square(), Square(), Triangle(), Triangle())
        @test face_shapes(prism, 1) == Square()
        @test measure(prism) == 1.0
        @test shape(Penta6_t()) == Prism()
        @test is_point_in_shape(prism, SA[0.1, 0.1, 0.0]) == true
    end

    @testset "Pyramid" begin
        pyr = Pyramid()
        @test topodim(pyr) == 3
        @test nvertices(pyr) == 5
        @test nedges(pyr) == 8
        @test nfaces(pyr) == 5
        @test face_area(pyr) == SA[4.0, √(2.0), √(2.0), √(2.0), √(2.0)]
        @test faces2nodes(pyr, 1) == SA[1, 4, 3, 2]
        @test faces2nodes(pyr, -1) == reverse(faces2nodes(pyr, 1))
        @test normals(pyr)[1] == SA[0.0, 0.0, -1.0]
        @test normal(pyr, 1) == SA[0.0, 0.0, -1.0]
        @test center(pyr) == SA[0.0, 0.0, 1.0 / 5.0]
        @test face_shapes(pyr) == (Square(), Triangle(), Triangle(), Triangle(), Triangle())
        @test face_shapes(pyr, 1) == Square()
        @test measure(pyr) == 4.0 / 3.0
        @test shape(Pyra5_t()) == Pyramid()
        @test is_point_in_shape(pyr, SA[0.0, 0.0, 0.1]) == true
    end
end
