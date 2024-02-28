@testset "Shape functions" begin
    # Mesh
    nspa = 2
    mesh = one_cell_mesh(:quad)
    icell = 1
    s = shape(cells(mesh)[icell])
    cInfo = CellInfo(mesh, icell)
    cPoint = CellPoint(SA[0.0, 0.0], cInfo, ReferenceDomain())

    # Function and fe spaces
    fs = FunctionSpace(:Lagrange, 1)
    V_vec = TestFESpace(fs, mesh; size = 3)
    V_sca = TestFESpace(fs, mesh; size = 1)
    ndofs_sca = Bcube.get_ndofs(V_sca, s)

    # Shape functions
    λ_vec = Bcube.get_shape_functions(V_vec, s)
    λ_sca = Bcube.get_shape_functions(V_sca, s)

    # LazyMapOver them !
    λ_sca = Bcube.LazyMapOver(λ_sca)
    λ_vec = Bcube.LazyMapOver(λ_vec)

    # Scalar tests
    f = Bcube.materialize(λ_sca, cInfo)
    @test Bcube.unwrap(f(cPoint)) == (0.25, 0.25, 0.25, 0.25)
    f = Bcube.materialize(∇(λ_sca), cInfo)
    @test Bcube.unwrap(f(cPoint)) ==
          ([-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25])

    # Vector tests
    f = Bcube.materialize(λ_vec, cInfo)
    @test Bcube.unwrap(f(cPoint)) == (
        [0.25, 0.0, 0.0], # λ1, 0, 0
        [0.25, 0.0, 0.0], # λ2, 0, 0
        [0.25, 0.0, 0.0], # λ3, 0, 0
        [0.25, 0.0, 0.0], # λ4, 0, 0
        [0.0, 0.25, 0.0], # 0, λ1, 0
        [0.0, 0.25, 0.0], # 0, λ2, 0
        [0.0, 0.25, 0.0], # 0, λ3, 0
        [0.0, 0.25, 0.0], # 0, λ4, 0
        [0.0, 0.0, 0.25], # 0, 0, λ1
        [0.0, 0.0, 0.25], # 0, 0, λ2
        [0.0, 0.0, 0.25], # 0, 0, λ3
        [0.0, 0.0, 0.25], # 0, 0, λ4
    )
    f_sca = Bcube.materialize(∇(λ_sca), cInfo)
    _f_sca = Bcube.unwrap(f_sca(cPoint))
    f_vec = Bcube.materialize(∇(λ_vec), cInfo)
    _f_vec = Bcube.unwrap(f_vec(cPoint))
    for i in 1:ndofs_sca
        for j in 1:Bcube.get_size(V_vec)
            I = i + (j - 1) * ndofs_sca
            for k in 1:nspa
                @test _f_vec[I][j, k] == _f_sca[i][k]
            end
        end
    end
end
