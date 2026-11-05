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
