@testset "face normals on curved faces" begin
    # Quad9 mesh with curved faces: on a conforming interior face, the normals of the two sides,
    # evaluated at the same face point, are opposite (the faces are run in opposite directions
    # by the two cells, so a normal evaluated at the mirrored point would not be)
    warp(x) = SA[x[1] + 0.3 * x[2] ^ 2, x[2] + 0.25 * x[1] ^ 3]
    for order in (1, 2)
        mesh = transform(rectangle_mesh(4, 4; order), warp)
        Γ = InteriorFaceDomain(mesh)
        nΓ = get_face_normals(Γ)
        err = 0.0
        Bcube.foreach_element(Γ) do fInfo, _, _
            for ζ in (-0.7, 0.2, 1.0)
                fp = FacePoint(SA[ζ], fInfo, ReferenceDomain())
                n⁻ = materialize(materialize(side_n(nΓ), fInfo), fp)
                n⁺ = materialize(materialize(side_p(nΓ), fInfo), fp)
                err = max(err, norm(n⁻ + n⁺))
            end
        end
        @test err < 1e-12

        # divergence theorem for a constant field, written with the normals of both sides
        c = SA[0.3, -1.1]
        V = TestFESpace(TrialFESpace(FunctionSpace(:Lagrange, 0), mesh, :discontinuous))
        dΓ = Measure(Γ, 4)
        dΓb = Measure(BoundaryFaceDomain(mesh), 4)
        nΓ, nΓb = get_face_normals(dΓ), get_face_normals(dΓb)
        l(v) =
            ∫((c ⋅ side⁻(nΓ)) * side⁻(v) + (c ⋅ side⁺(nΓ)) * side⁺(v))dΓ +
            ∫((c ⋅ side⁻(nΓb)) * side⁻(v))dΓb
        @test maximum(abs, assemble_linear(l, V)) < 1e-12
    end

    # 3D: warped trilinear hexahedra (curved quadrangular faces)
    function warp3(x)
        SA[
            x[1] + 0.15 * x[2] ^ 2 + 0.1 * x[3] ^ 2,
            x[2] + 0.12 * x[1] ^ 2 - 0.1 * x[3] ^ 2,
            x[3] + 0.1 * x[1] * x[2],
        ]
    end
    mesh = transform(hexa_mesh(3, 3, 3), warp3)
    Γ = InteriorFaceDomain(mesh)
    nΓ = get_face_normals(Γ)
    err = 0.0
    Bcube.foreach_element(Γ) do fInfo, _, _
        for ζ in (SA[-0.7, 0.4], SA[0.3, -0.2])
            fp = FacePoint(ζ, fInfo, ReferenceDomain())
            n⁻ = materialize(materialize(side_n(nΓ), fInfo), fp)
            n⁺ = materialize(materialize(side_p(nΓ), fInfo), fp)
            err = max(err, norm(n⁻ + n⁺))
        end
    end
    @test err < 1e-12
end

@testset "same face point from both sides, any numbering of the cells" begin
    # two tetrahedra sharing a face, the nodes of the second one in any order (orientations of
    # the shared face reversed, or rotated across an inverted cell)
    perms(v) = if length(v) ≤ 1
        [v]
    else
        [[v[i]; p] for i in eachindex(v) for p in perms(v[1:end .!= i])]
    end
    xs = [
        SA[0.0, 0.0, 0.0],
        SA[1.0, 0.0, 0.0],
        SA[0.0, 1.0, 0.0],
        SA[0.0, 0.0, 1.0],
        SA[0.3, 0.3, -1.0],
    ]
    phys(cInfo, ξ) = mapping(Bcube.celltype(cInfo), Bcube.nodes(cInfo), ξ)
    err, nperms = 0.0, 0
    for c2 in perms([1, 2, 3, 5])
        mesh = Bcube.Mesh(
            Node.(xs),
            [Tetra4_t(), Tetra4_t()],
            Connectivity([4, 4], [1, 2, 3, 4, c2...]),
        )
        Bcube.foreach_element(InteriorFaceDomain(mesh)) do fInfo, _, _
            cn, cp = Bcube.get_cellinfo_n(fInfo), Bcube.get_cellinfo_p(fInfo)
            for ζ in (SA[0.2, 0.3], SA[0.6, 0.1])
                fp = FacePoint(ζ, fInfo, ReferenceDomain())
                xn = phys(cn, get_coords(side_n(fp)))
                xp = phys(cp, get_coords(side_p(fp)))
                err = max(err, norm(xn - xp))
            end
        end
        nperms += 1
    end
    @test nperms == 24
    @test err < 1e-14
end

@testset "coplanar rotation on curved hypersurfaces" begin
    # Quad9 surface in 3D: on each side, the rotation maps the normal of the cell of this side on
    # the normal of the opposite cell, both evaluated at the same face point (`side⁺` evaluates
    # the rotation on `opposite_side` of the face point, which must be the same point)
    m = rectangle_mesh(4, 4; order = 2)
    surf(x) =
        SA[x[1], x[2], 0.3 * sin(5 * x[1]) * cos(4 * x[2]) + 0.2 * sin(6 * x[2]) + x[1] ^ 4]
    mesh = Bcube.Mesh(
        [Bcube.Node(surf(get_coords(n))) for n in get_nodes(m)],
        collect(Bcube.cells(m)),
        Bcube.connectivities_indices(m, :c2n),
    )
    Γ = InteriorFaceDomain(mesh)
    R, ν = Bcube.CoplanarRotation(), Bcube.CellNormal()
    err_point, err_n, err_p = 0.0, 0.0, 0.0
    Bcube.foreach_element(Γ) do fInfo, _, _
        for ζ in (-0.7, 0.2, 1.0)
            fp = FacePoint(SA[ζ], fInfo, ReferenceDomain())
            x⁺ = get_coords(side_p(fp))
            err_point =
                max(err_point, norm(get_coords(side_n(Bcube.opposite_side(fp))) - x⁺))
            ν⁻ = materialize(materialize(side_n(ν), fInfo), fp)
            ν⁺ = materialize(materialize(side_p(ν), fInfo), fp)
            R⁻ = materialize(materialize(side_n(R), fInfo), fp)
            R⁺ = materialize(materialize(side_p(R), fInfo), fp)
            err_n = max(err_n, norm(R⁻ - Bcube._coplanar_rotation(ν⁻, ν⁺)))
            err_p = max(err_p, norm(R⁺ - Bcube._coplanar_rotation(ν⁺, ν⁻)))
        end
    end
    @test err_point < 1e-12
    @test err_n < 1e-12
    @test err_p < 1e-12
end
