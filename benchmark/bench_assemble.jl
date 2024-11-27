module BenchAssemble

using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Bcube

suite = BenchmarkGroup()

function basic_bilinear()
    suite = BenchmarkGroup()
    degree = 1
    degquad = 2 * degree + 1
    mesh = rectangle_mesh(251, 11; xmax = 1.0, ymax = 0.1)
    Uspace = TrialFESpace(FunctionSpace(:Lagrange, degree), mesh; size = 2)
    Vspace = TestFESpace(Uspace)
    dΩ = Measure(CellDomain(mesh), degquad)

    a(u, v) = ∫(u ⋅ v)dΩ
    m(u, v) = ∫(∇(u) ⋅ ∇(v))dΩ

    #warmup
    assemble_bilinear(a, Uspace, Vspace)
    assemble_bilinear(m, Uspace, Vspace)

    suite["mass matrix"] = @benchmarkable assemble_bilinear($a, $Uspace, $Vspace)
    suite["stiffness matrix"] = @benchmarkable assemble_bilinear($m, $Uspace, $Vspace)
    return suite
end

avg(u) = 0.5 * (side⁺(u) + side⁻(u))

function poisson_dg()
    suite = BenchmarkGroup()

    degree = 3
    degree_quad = 2 * degree + 1
    γ = degree * (degree + 1)
    n = 4
    Lx = 1.0
    h = Lx / n

    uₐ = PhysicalFunction(x -> 3 * x[1] + x[2]^2 + 2 * x[1]^3)
    f = PhysicalFunction(x -> -2 - 12 * x[1])
    g = uₐ

    # Build mesh
    mesh = rectangle_mesh(
        n + 1,
        n + 1;
        xmin = -Lx / 2,
        xmax = Lx / 2,
        ymin = -Lx / 2,
        ymax = Lx / 2,
    )

    # Choose degree and define function space, trial space and test space
    fs = FunctionSpace(:Lagrange, degree)
    U = TrialFESpace(fs, mesh, :discontinuous)
    V = TestFESpace(U)

    # Define volume and boundary measures
    dΩ = Measure(CellDomain(mesh), degree_quad)
    dΓ = Measure(InteriorFaceDomain(mesh), degree_quad)
    dΓb = Measure(BoundaryFaceDomain(mesh), degree_quad)
    nΓ = get_face_normals(dΓ)
    nΓb = get_face_normals(dΓb)

    a_Ω(u, v) = ∫(∇(v) ⋅ ∇(u))dΩ
    l_Ω(v) = ∫(v * f)dΩ

    function a_Γ(u, v)
        ∫(
            -jump(v, nΓ) ⋅ avg(∇(u)) - avg(∇(v)) ⋅ jump(u, nΓ) +
            γ / h * jump(v, nΓ) ⋅ jump(u, nΓ),
        )dΓ
    end

    fa_Γb(u, ∇u, v, ∇v, n) = -v * (∇u ⋅ n) - (∇v ⋅ n) * u + (γ / h) * v * u
    a_Γb(u, v) = ∫(fa_Γb ∘ map(side⁻, (u, ∇(u), v, ∇(v), nΓb)))dΓb

    fl_Γb(v, ∇v, n, g) = -(∇v ⋅ n) * g + (γ / h) * v * g
    l_Γb(v) = ∫(fl_Γb ∘ map(side⁻, (v, ∇(v), nΓb, g)))dΓb

    a(u, v) = a_Ω(u, v) + a_Γ(u, v) + a_Γb(u, v)
    l(v) = l_Ω(v) + l_Γb(v)

    suite["a_Ω(u, v)"] = @benchmarkable assemble_bilinear($a_Ω, $U, $V)
    suite["a_Γ(u, v)"] = @benchmarkable assemble_bilinear($a_Γ, $U, $V)
    suite["a_Γb(u, v)"] = @benchmarkable assemble_bilinear($a_Γb, $U, $V)
    suite["l_Ω(v)"] = @benchmarkable assemble_linear($l_Ω, $V)
    suite["l_Γb(v)"] = @benchmarkable assemble_linear($l_Γb, $V)
    suite["AffineFESystem"] = @benchmarkable Bcube.AffineFESystem($a, $l, $U, $V)

    return suite
end

suite = BenchmarkGroup()
suite["basic bilinear"] = basic_bilinear()
suite["poisson DG"] = poisson_dg()

end  # module

BenchAssemble.suite
