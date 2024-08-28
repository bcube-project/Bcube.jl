@testset "issue #130" begin
    function run()
        # Settings
        Zin = 4
        Zout = 8
        nr = 3
        nz = 3
        degree = 1
        qdeg = 2 * degree + 1 # quad degree

        # Mesh and measures
        mesh = rectangle_mesh(
            nr,
            nz;
            ymin = Zin,
            ymax = Zout,
            bnd_names = ("AXIS", "WALL", "INLET", "OUTLET"),
        )
        Ω = CellDomain(mesh)
        dΩ = Measure(Ω, qdeg)

        # FESpace
        fs = FunctionSpace(:Lagrange, degree)
        U_up = TrialFESpace(fs, mesh)
        U_ur = TrialFESpace(fs, mesh)
        U_uz = TrialFESpace(fs, mesh)

        V_up = TestFESpace(U_up)
        V_ur = TestFESpace(U_ur)
        V_uz = TestFESpace(U_uz)

        U = MultiFESpace(U_up, U_ur, U_uz)
        V = MultiFESpace(V_up, V_ur, V_uz)

        # Bilinear forms
        function test(u, ∇u, v, ∇v)
            up, ur, uz = u
            vp, vr, vz = v
            ∇up, ∇ur, ∇uz = ∇u

            ∂u∂r = ∇ur ⋅ SA[1, 0]
            return ∂u∂r * vp
        end

        a((up, ur, uz), (vp, vr, vz)) = ∫(∇(ur) ⋅ SA[1, 0] * vp)dΩ
        A = assemble_bilinear(a, U, V)

        b(u, v) = ∫(test ∘ (u, map(∇, u), v, map(∇, v)))dΩ
        B = assemble_bilinear(b, U, V)

        @test A == B
    end

    run()
end