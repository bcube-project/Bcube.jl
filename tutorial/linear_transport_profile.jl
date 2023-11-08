module LinearTransport #hide

using Bcube
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using Profile
#using Cthulhu

const degree = 1 # Function-space degree (Taylor(0) = first degree Finite Volume)
const nite = 100 # Number of time iteration(s)
# const c = 2.0 # Convection velocity (must be a vector)
const c = SA[-2.5, 4.0] # Convection velocity (must be a vector)
const CFL = 1 # 0.1 for degree 1
const nx = 128 # Number of nodes in the x-direction
const ny = 128 # Number of nodes in the y-direction
const lx = 2.0 # Domain width
const ly = 2.0 # Domain height
const Δt = CFL * (lx / nx) / norm(c) # Time step

function run_linear_transport()

    # tmp_path = "tmp.msh"
    # gen_rectangle_mesh(tmp_path, :quad; nx = nx, ny = ny, lx = lx, ly = ly, xc = 0., yc = 0.)
    # mesh = read_msh(tmp_path)
    # rm(tmp_path)
    # mesh = line_mesh(nx; xmin=-lx / 2, xmax=lx / 2)
    mesh =
        rectangle_mesh(nx, ny; xmin = -lx / 2, xmax = lx / 2, ymin = -ly / 2, ymax = ly / 2)
    # mesh = rectangle_mesh(3, 3)

    fs = FunctionSpace(:Lagrange, degree)
    U = TrialFESpace(fs, mesh; isContinuous = false)
    V = TestFESpace(U)

    # Define measures for cell and interior face integrations
    Γ = InteriorFaceDomain(mesh)
    dΩ = Measure(CellDomain(mesh), 2 * degree + 1)
    dΓ = Measure(Γ, 2 * degree + 1)
    nΓ = get_face_normals(Γ)

    m(u, v) = ∫(u ⋅ v)dΩ
    M = assemble_bilinear(m, U, V)
    #invM = inv(Matrix(M))

    u = FEFunction(V, [1.0 * i for i in 1:get_ndofs(V)])

    function upwind(ui, uj, nij)
        cij = c ⋅ nij
        if cij > zero(cij)
            flux = cij * ui
        else
            flux = cij * uj
        end
        flux
    end

    l1(v) = ∫((c * u) ⋅ ∇(v))dΩ #
    l2(v) = ∫(side_n(v))dΓ
    l3(v) = ∫((c ⋅ Bcube.side_p(nΓ)) ⋅ Bcube.side_p(v))dΓ

    flux4 = upwind ∘ (side_n(u), Bcube.side_p(u), side_n(nΓ))
    l4(v) = ∫(flux4 * side_n(v))dΓ

    # multi-linear variant of case 4
    function l5((v1, v2, v3))
        ∫(flux4 * (side_n(v1) + 2 * side_n(v2) + 3 * side_n(v3)))dΓ
    end

    # Allocate
    # b1 = assemble_linear(l1, V)
    # b01 = assemble_linear(l01, V)
    # @show all(isapprox.(b1, b01))
    b2 = assemble_linear(l2, V)
    b3 = assemble_linear(l3, V)
    b4 = assemble_linear(l4, V)

    V3 = MultiFESpace(V, V, V; arrayOfStruct = false)
    b5 = assemble_linear(l5, V3)
    #@descend assemble_linear(l5, V3)

    f6(x) = 3x - 5x + 10x
    l6(v) = f6(l5(v))
    #Bcube.show_lazy_operator(l6(V3))
    b6 = assemble_linear(l6, V3)

    TEST_2 = true
    BENCH_2 = true
    PROFILING_2 = false
    TEST_3 = true
    BENCH_3 = true
    PROFILING_3 = false
    TEST_4 = true
    BENCH_4 = true
    PROFILING_4 = false
    TEST_5 = true
    BENCH_5 = true
    PROFILING_5 = true
    TEST_6 = true
    BENCH_6 = true
    PROFILING_6 = false

    # TESTS with Legacy
    if true
        fes = FESpace(fs, :discontinuous; size = 1)
        ϕ = CellVariable(:ϕ, mesh, fes)
        λ = TestFunction(mesh, fes)
        nΓ_legacy = FaceNormals(dΓ)
        set_values!(ϕ, get_dof_values(u))

        for icell in 1:ncells(mesh)
            @assert all(get_dof_values(u, icell) .≈ get_values(ϕ, icell))
        end

        b2_legacy(λ) = zeros(ϕ) + ∫(side⁻(λ))dΓ
        if TEST_2
            _b2_legacy = b2_legacy(λ)
            if b2 ≉ get_values(_b2_legacy)
                @show b2
                @show get_values(_b2_legacy)
                error("b2 ≉ get_values(_b2_legacy) ")
            end
            println("TEST_2: OK")
        end
        ## bench
        if BENCH_2
            println("=== TEST 2")
            print("Legacy  :  ")
            @btime $b2_legacy($λ)                   # Legacy  :    22.129 ms (383041 allocations: 31.25 MiB)
            print("New API :  ")
            @btime assemble_linear($l2, $V)   # New API :    9.544 ms (47 allocations: 5.32 MiB)
        end
        ## profiling
        if PROFILING_2
            Profile.init(; n = 5 * 10^7)
            Profile.clear()
            Profile.clear_malloc_data()
            @profile begin
                for i in 1:100
                    assemble_linear(l2, V)
                end
            end
        end

        b3_legacy(λ) = zeros(ϕ) + ∫((c ⋅ side⁺(nΓ_legacy)) * side⁺(λ))dΓ
        if TEST_3
            _b3_legacy = b3_legacy(λ)
            if b3 ≉ get_values(_b3_legacy)
                @show b3
                @show get_values(_b3_legacy)
                @show length(b3), length(get_values(_b3_legacy))
                error("b3 ≉ get_values(_b3_legacy) ")
            end
            println("TEST_3: OK")
        end
        ## bench
        if BENCH_3
            println("=== TEST 3")
            print("Legacy  :  ")
            @btime $b3_legacy($λ)                   # Legacy  :    28.970 ms (383041 allocations: 61.53 MiB)
            print("New API :  ")
            @btime assemble_linear($l3, $V)   # New API :    14.680 ms (47 allocations: 5.32 MiB)
        end
        ## profiling
        if PROFILING_3
            Profile.init(; n = 5 * 10^7)
            Profile.clear()
            Profile.clear_malloc_data()
            @profile begin
                for i in 1:100
                    #b3_legacy(λ)
                    assemble_linear(l3, V)
                end
            end
        end

        function upwind_legacy((ui, uj, nij, λi))
            flux = λi * upwind(ui, uj, nij)
        end
        flux4_legacy(λ) = upwind_legacy ∘ (side⁻(ϕ), side⁺(ϕ), side⁻(nΓ_legacy), side⁻(λ))
        b4_legacy(λ) = zeros(ϕ) + ∫(flux4_legacy(λ))dΓ
        if TEST_4
            _b4_legacy = b4_legacy(λ)
            if b4 ≉ get_values(_b4_legacy)
                display(b4[1:10])
                display(get_values(_b4_legacy)[1:10])
                display(findall(x -> abs(x) > 1.0e-10, b4 .- get_values(_b4_legacy))[1:40])
                @show length(b4), length(get_values(_b4_legacy))
                error("b4 ≉ get_values(_b4_legacy) ")
            end
            println("TEST_4: OK")
        end
        ## bench
        if BENCH_4
            println("=== TEST 4")
            print("Legacy  :  ")
            @btime $b4_legacy($λ)                   # Legacy  :    33.259 ms (383041 allocations: 69.35 MiB)
            print("New API :  ")
            @btime assemble_linear($l4, $V)   # New API :    18.733 ms (47 allocations: 5.32 MiB)
        end
        ## profiling
        if PROFILING_4
            Profile.init(; n = 5 * 10^7)
            Profile.clear()
            Profile.clear_malloc_data()
            @profile begin
                for i in 1:100
                    assemble_linear(l4, V)
                end
            end
        end

        function upwind_legacy_multi(w)
            ui, uj, nij, λi = w
            λi1, λi2, λi3 = λi
            flux = upwind(ui, uj, nij)
            (λi1 * flux, λi2 * flux, λi3 * flux)
        end
        function flux5_legacy(λ)
            upwind_legacy_multi ∘ (side⁻(ϕ), side⁺(ϕ), side⁻(nΓ_legacy), side⁻(λ))
        end
        b5_legacy(λ) = zeros.((ϕ, ϕ, ϕ)) + ∫(flux5_legacy(λ))dΓ
        if TEST_5
            _b5_legacy = vcat(map(get_values, b5_legacy((λ, 2 * λ, 3 * λ)))...)
            if b5 ≉ _b5_legacy
                display(b5[1:10])
                display(_b5_legacy[1:10])
                display(findall(x -> abs(x) > 1.0e-10, b5 .- _b5_legacy)[1:40])
                @show length(b5), length(_b5_legacy)
                @show b5 ≈ vcat(b4, b4, b4)
                display(
                    findall(x -> abs(x) > 1.0e-10, b5 .- vcat(b4, 2 * b4, 3 .* b4))[1:40],
                )
                error("b5 ≉ get_values(_b5_legacy) ")
            end
            println("TEST_5: OK")
        end
        ## bench
        if BENCH_5
            println("=== TEST 5")
            print("Legacy  :  ")
            @btime $b5_legacy(($λ, 2 * $λ, 3 * $λ))        # Legacy  :    24.119 ms (127034 allocations: 33.24 MiB)
            print("New API :  ")
            @btime assemble_linear($l5, $V3)   # New API :    27.059 ms (131 allocations: 15.96 MiB)
        end
        ## profiling
        if PROFILING_5
            Profile.init(; n = 5 * 10^7)
            Profile.clear()
            Profile.clear_malloc_data()
            @profile begin
                for i in 1:100
                    assemble_linear(l5, V3)
                end
            end
        end

        if TEST_6
            _b5_legacy = vcat(map(get_values, b5_legacy((λ, 2 * λ, 3 * λ)))...)
            _b6_legacy = f6(_b5_legacy)
            if b6 ≉ _b6_legacy
                display(b6[1:10])
                display(_b6_legacy[1:10])
                display(findall(x -> abs(x) > 1.0e-10, b6 .- _b6_legacy)[1:40])
                @show length(b5), length(_b6_legacy)
                error("b6 ≉ get_values(_b6_legacy) ")
            end
            _b6 = assemble_linear(l5, V3)
            @assert f6(_b6) ≈ _b6_legacy
            println("TEST_6: OK")
        end
        ## bench
        if BENCH_6
            println("=== TEST 6")
            print("New API (combine b)          :  ")
            @btime $f6(assemble_linear($l5, $V3))          # New API (combine b)          :    28.634 ms (143 allocations: 23.35 MiB)
            print("New API (MuliIntegration{3}) :  ")
            @btime assemble_linear($l6, $V3)               # New API (MuliIntegration{3}) :    73.027 ms (148 allocations: 15.97 MiB)
        end
        ## profiling
        if PROFILING_6
            Profile.init(; n = 5 * 10^7)
            Profile.clear()
            Profile.clear_malloc_data()
            @profile begin
                for i in 1:100
                    assemble_linear(l6, V3)
                end
            end
        end

        @show length(get_values(_b2_legacy))
        @show get_ndofs(V)
        println("ALL TESTS OK")
    end
    #
    # du = zeros.((ϕ,))
    # du += du_Γ
    # b3 = get_values(du[1])
    # @show b3

    # for i = 1:nite
    #     assemble_linear!(b1, l1, V)
    #     assemble_linear!(b2, l2, V)

    #     rhs = Δt .* invM * (b1 - b2)
    #     u.dofValues .+= rhs
    # end

end

run_linear_transport()

end #hide
