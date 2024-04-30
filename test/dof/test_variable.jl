"""
This function compares the values of the input function `f` on the cell nodes with the values
obtained by a `CellVariable` intialized by `f` on the corresponding cell nodes.
"""
function test_setvalues!_with_function(
    mesh,
    meshOrder,
    fesType,
    fesDegree,
    fesContinuity,
    f,
)
    fs = FunctionSpace(fesType, fesDegree)
    isa(f, Tuple) ? size = length(f) : size = 1
    fes = FESpace(fs, fesContinuity; size = size)
    u = CellVariable(:u, mesh, fes)
    set_values!(u, f)
    uc = var_on_centers(u)

    cellTypes = cells(mesh)
    c2n = connectivities_indices(mesh, :c2n)

    for i in 1:ncells(mesh)
        ctype = cellTypes[i]
        cnodes = get_nodes(mesh, c2n[i])
        uᵢ = u[i, Val(:unstable)]
        a = [f(Bcube.mapping(ctype, cnodes, x)) for x in get_coords(shape(ctype))]
        b = [uᵢ(ξ) for ξ in get_coords(shape(ctype))]

        @test all(isapprox.(a, b, rtol = 1000eps()))
    end
end

function test_getindex_with_collocated_quad(
    mesh,
    meshOrder,
    fesType,
    fesDegree,
    fesContinuity,
    f,
)
    fs = FunctionSpace(fesType, fesDegree)
    isa(f, Tuple) ? size = length(f) : size = 2
    fes = FESpace(fs, fesContinuity; size = size)
    u = CellVariable(:u, mesh, fes)
    set_values!(u, f)

    cellTypes = cells(mesh)
    c2n = connectivities_indices(mesh, :c2n)

    for i in 1:ncells(mesh)
        ct = cellTypes[i]
        cn = get_nodes(mesh, c2n[i])
        uᵢ = u[i, Val(:unstable)]
        quadrule = QuadratureRule(shape(ct), get_quadrature(fs))
        a = uᵢ(quadrule)
        b = uᵢ.(get_nodes(quadrule))
        @test all(a .≈ b)
    end
end

@testset "CellVariable" begin
    @testset "Mean and cell center values" begin
        @testset "Discontinuous" begin
            @testset "Taylor - linear quad" begin
                for Ly in (1.0, 2.0, 3.0), Lx in (1.0, 2.0, 3.0) # different cell size
                    mesh = one_cell_mesh(
                        :quad;
                        xmin = 0.0,
                        xmax = Lx,
                        ymin = 0.0,
                        ymax = Ly,
                        order = 1,
                    )
                    degree = 1
                    fs1 = FunctionSpace(:Taylor, degree)
                    fes1 = FESpace(fs1, :discontinuous; size = 1) # DG, scalar
                    u = CellVariable(:u, mesh, fes1)

                    for f in (x -> 10, x -> 12x[1] + 2x[2] + 4) # test for constant and linear functions
                        set_values!(u, f)
                        u̅ = mean_values(u, Val(degree))
                        uc = var_on_centers(u)
                        @test length(u̅) === length(uc) === 1
                        @test u̅[1] ≈ uc[1] ≈ f([Lx / 2, Ly / 2])
                    end

                    fes2 = FESpace(fs1, :discontinuous; size = 2)
                    v = CellVariable(:v, mesh, fes2) # DG, 2-vector variable
                    f = (x -> 3x[1] + 4x[2] + 7 * x[1] * x[2], x -> 12x[1] + 2x[2] + 4) # here, one function per variable component
                    gradf = x -> [
                        3+7 * x[2] 4+7 * x[1]
                        12 2
                    ]
                    set_values!(v, f)
                    v̅ = mean_values(v, Val(degree))
                    vc = var_on_centers(v)
                    @test length(v̅) === 1
                    @test size(vc) === (1, 2)
                    @test v̅[1] ≈
                          vc[1, :] ≈
                          [f[1]([Lx / 2, Ly / 2]), f[2]([Lx / 2, Ly / 2])]
                end
            end

            @testset "Lagrange - linear and quadratic quad" begin
                for Ly in (1.0, 2.0, 3.0), Lx in (1.0, 2.0, 3.0) # different cell size
                    for continuity in (:discontinuous, :continuous)
                        for meshOrder in (1, 2)
                            mesh = one_cell_mesh(
                                :quad;
                                xmin = 0.0,
                                xmax = Lx,
                                ymin = 0.0,
                                ymax = Ly,
                                order = meshOrder,
                            )
                            degree = meshOrder
                            fs1 = FunctionSpace(:Lagrange, degree)
                            fes1 = FESpace(fs1, continuity; size = 1) # DG, scalar
                            u = CellVariable(:u, mesh, fes1)

                            for (f, ∇f) in zip(
                                (
                                    x -> 10x[1] * x[2] + 4x[1] + 7x[2],
                                    x -> 12x[1] + 2x[2] + 4,
                                ),
                                (x -> [10 * x[2] + 4 10 * x[1] + 7], x -> [12 2]),
                            ) # test for constant and linear functions
                                set_values!(u, f)
                                u̅ = mean_values(u, Val(degree))
                                uc = var_on_centers(u)
                                @test length(u̅) === length(uc) === 1
                                @test u̅[1] ≈ uc[1] ≈ f([Lx / 2, Ly / 2])
                                @test u[1, Val(:unstable)]([0.0, 0.0]) ≈ f([Lx / 2, Ly / 2])
                                @test all(
                                    isapprox.(
                                        ∇(u)[1, Val(:unstable)]([0.0, 0.0]),
                                        ∇f([Lx / 2, Ly / 2]),
                                    ),
                                )
                            end

                            fes2 = FESpace(fs1, continuity; size = 2)
                            v = CellVariable(:v, mesh, fes2) # DG, 2-vector variable
                            f = (
                                x -> 3x[1] + 4x[2] + 7 * x[1] * x[2],
                                x -> 12x[1] + 2x[2] + 4,
                            ) # here, one function per variable component
                            ∇f = x -> [
                                3+7 * x[2] 4+7 * x[1]
                                12 2
                            ]
                            set_values!(v, f)
                            v̅ = mean_values(v, Val(degree))
                            vc = var_on_centers(v)
                            @test length(v̅) === 1
                            @test size(vc) === (1, 2)
                            @test v̅[1] ≈
                                  vc[1, :] ≈
                                  [f[1]([Lx / 2, Ly / 2]), f[2]([Lx / 2, Ly / 2])]
                            @test all(
                                isapprox.(
                                    ∇(v)[1, Val(:unstable)]([0.0, 0.0]),
                                    ∇f([Lx / 2, Ly / 2]),
                                ),
                            )
                            @test all(
                                isapprox.(
                                    ∇(v)[1, Val(:unstable)]([1.0, 0.0]),
                                    ∇f([Lx, Ly / 2]),
                                ),
                            )
                        end
                    end
                end
            end
        end
    end

    dict_f = Dict(
        0 => x -> 6.0,  #deg 0
        1 => x -> 2x[1] + 4x[2] + 5,  #deg 1
        2 => x -> 3x[1] * x[2] + x[1]^2 + 2.5x[2] - 1.5, #deg 2
        3 => x -> 3x[1]^2 * x[2] + 10x[2] + 4,
    ) #deg 3

    dict_fesDegree = Dict(
        (:quad, :Lagrange) => 0:3,
        (:tri, :Lagrange) => 0:2,
        (:quad, :Taylor) => 0:1,
        (:tri, :Taylor) => 0:0,
    )
    Lx = 3.0
    Ly = 2.0
    nx = 3
    ny = 4
    @testset "setvalues! with function" begin
        @testset "mesh type=$meshElement and order=$meshOrder" for meshElement in
                                                                   (:quad, :tri),
            meshOrder in 1:2

            if meshElement == :tri # tri mesh generator  with several cells is not available
                mesh = one_cell_mesh(
                    meshElement;
                    xmin = 0.0,
                    xmax = Lx,
                    ymin = 0.0,
                    ymax = Ly,
                    order = meshOrder,
                )
            else
                mesh = rectangle_mesh(
                    nx,
                    ny;
                    type = meshElement,
                    xmin = 0.0,
                    xmax = Lx,
                    ymin = 0.0,
                    ymax = Ly,
                    order = meshOrder,
                )
            end
            @testset "fespace: type=$fesType degree=$fesDegree $fesContinuity" for fesType in
                                                                                   (
                    :Lagrange,
                    :Taylor,
                ),
                fesDegree in dict_fesDegree[(meshElement, fesType)],
                fesContinuity in (:discontinuous, :continuous)

                (fesType == :Taylor && fesContinuity == :continuous) && continue
                @testset "function deg=$d" for (d, f) in dict_f
                    d <= fesDegree && test_setvalues!_with_function(
                        mesh,
                        meshOrder,
                        fesType,
                        fesDegree,
                        fesContinuity,
                        f,
                    )
                end
            end
        end
    end

    @testset "Boundary dofs" begin
        # Note bmxam : to build this check, I sketched the rectangle
        # and built the dof numbering by hand.

        path = joinpath(tempdir, "mesh.msh")
        gen_rectangle_mesh(tmp_path, :quad; nx = 2, ny = 3)
        mesh = read_msh(path, 2)

        fs = FunctionSpace(:Lagrange, 1)
        fes = FESpace(fs, :continuous; size = 1)
        u = CellVariable(:u, mesh, fes)

        bnd_dofs = boundary_dofs(u, "East")
        @test bnd_dofs == [2, 4, 6]

        fes = FESpace(fs, :discontinuous; size = 1)
        u = CellVariable(:u, mesh, fes)
        bnd_dofs = boundary_dofs(u, "East")
        @test bnd_dofs == [2, 4, 6, 8]

        fes = FESpace(fs, :continuous; size = 2)
        u = CellVariable(:u, mesh, fes)
        bnd_dofs = boundary_dofs(u, "East")
        @test bnd_dofs == [2, 6, 4, 8, 10, 12]
    end

    @testset "Dof numbering checker" begin
        mesh = rectangle_mesh(
            4,
            4;
            type = :quad,
            xmin = 0.0,
            xmax = 6.0,
            ymin = 0.0,
            ymax = 6.0,
            order = 1,
        )

        degree = 1
        fs = FunctionSpace(:Lagrange, degree)
        fes_sca = FESpace(fs, :discontinuous; size = 1) # DG, scalar
        fes_vec = FESpace(fs, :discontinuous; size = 2) # DG, vectoriel

        u = CellVariable(:u, mesh, fes_sca)
        v = CellVariable(:v, mesh, fes_vec)

        @test Bcube.check_numbering(u; verbose = false, exit_on_error = false) == 0
        @test Bcube.check_numbering(v; verbose = false, exit_on_error = false) == 0

        # insert random errors
        u.dhl.iglob[32] = u.dhl.iglob[29]
        @test Bcube.check_numbering(u; verbose = false, exit_on_error = false) == 1

        u.dhl.iglob[32] = u.dhl.iglob[1]
        @test Bcube.check_numbering(u; verbose = false, exit_on_error = false) == 1

        u.dhl.iglob[14] = u.dhl.iglob[27]
        @test Bcube.check_numbering(u; verbose = false, exit_on_error = false) == 1

        # CONTINUOUS
        fes_sca = FESpace(fs, :continuous; size = 1) # DG, scalar
        fes_vec = FESpace(fs, :continuous; size = 2) # DG, vectoriel

        u = CellVariable(:u, mesh, fes_sca)
        v = CellVariable(:v, mesh, fes_vec)

        @test Bcube.check_numbering(u) == 0
        @test Bcube.check_numbering(v) == 0

        # insert random errors
        u.dhl.iglob[24] = u.dhl.iglob[31]
        @test Bcube.check_numbering(u; verbose = false, exit_on_error = false) == 4

        u.dhl.iglob[2] = u.dhl.iglob[22]
        @test Bcube.check_numbering(u; verbose = false, exit_on_error = false) == 6

        u.dhl.iglob[20] = u.dhl.iglob[5]
        @test Bcube.check_numbering(u; verbose = false, exit_on_error = false) == 14
    end
end
