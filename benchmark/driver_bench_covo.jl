function run_covo()
    suite = BenchmarkGroup()

    # alias
    _u, U, V, params, cache = Covo.run_covo()
    u = Covo.Bcube.get_fe_functions(_u)
    dΓ = params.dΓ
    dΩ = params.dΩ
    nΓ = Covo.Bcube.get_face_normals(dΓ)
    Δt = Covo.Δt

    l_vol(v) = Covo.Bcube.∫(Covo.flux_Ω(u, v))dΩ
    l_Γ(v) = Covo.Bcube.∫(Covo.flux_Γ(u, v, nΓ))dΓ

    b_vol = zeros(Covo.Bcube.get_ndofs(U))
    b_fac = zeros(Covo.Bcube.get_ndofs(U))

    # warmup (force precompilation of @generated functions if necessary)
    Covo.Bcube.assemble_linear!(b_vol, l_vol, V)
    Covo.Bcube.assemble_linear!(b_fac, l_Γ, V)
    _rhs(u, t) = Covo.compute_residual(u, V, params, cache)
    Covo.forward_euler(_u, _rhs, 0.0, Δt)

    suite["integral_volume"]  = @benchmarkable Covo.Bcube.assemble_linear!($b_vol, $l_vol, $V)
    suite["integral_surface"] = @benchmarkable Covo.Bcube.assemble_linear!($b_fac, $l_Γ, $V)
    suite["explicit_step"]    = @benchmarkable Covo.forward_euler($_u, $_rhs, 0.0, $Δt)

    return suite
end
