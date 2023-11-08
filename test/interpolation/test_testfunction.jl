@testset "TestFunction" begin
    # Mesh
    mesh = one_cell_mesh(:quad)
    x = [0, 0.0]
    cell = CellInfo(mesh, 1)

    # Function space
    fs = FunctionSpace(:Lagrange, 1)

    # Finite element spaces
    fes_vec = FESpace(fs, :continuous; size = 3)
    fes_sca = FESpace(fs, :continuous; size = 1)

    # Test functions
    λ_vec = TestFunction(mesh, fes_vec)
    λ_sca = TestFunction(mesh, fes_sca)

    # Scalar tests
    f = λ_sca
    @test f[cell](x) == SA[0.25, 0.25, 0.25, 0.25]
    f = ∇(λ_sca)
    ∇λ_sca_ref = SA[-0.25 -0.25; 0.25 -0.25; -0.25 0.25; 0.25 0.25]
    @test f[cell](x) == ∇λ_sca_ref
    ndofs_sca, nspa = size(∇λ_sca_ref)

    # Vector tests
    f = λ_vec
    @test f[cell](x) == SA[
        0.25 0.0 0.0
        0.25 0.0 0.0
        0.25 0.0 0.0
        0.25 0.0 0.0
        0.0 0.25 0.0
        0.0 0.25 0.0
        0.0 0.25 0.0
        0.0 0.25 0.0
        0.0 0.0 0.25
        0.0 0.0 0.25
        0.0 0.0 0.25
        0.0 0.0 0.25
    ]
    f = ∇(λ_vec)
    n = size(fes_vec)
    for i in 1:ndofs_sca
        for j in 1:n
            I = i + (j - 1) * ndofs_sca
            for k in 1:nspa
                if (1 + (j - 1) * ndofs_sca <= I <= j * ndofs_sca)
                    @test f[cell](x)[I, j, k] == ∇λ_sca_ref[i, k]
                else
                    @test f[cell](x)[I, j, k] == 0.0
                end
            end
        end
    end
end
