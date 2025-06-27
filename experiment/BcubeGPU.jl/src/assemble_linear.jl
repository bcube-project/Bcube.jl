function kernabs_assemble_linear!(backend, y, f, V, measure, rdhl)
    quadrature = get_quadrature(measure) # not sure if it's needed here
    domain = get_domain(measure)

    # idof = 2
    # @code_warntype assemble_linear_elemental!(idof, y, f, domain, V, quadrature, rdhl)
    # @btime assemble_linear_elemental!($idof, $y, $f, $domain, $V, $quadrature, $rdhl)
    # error("dbg")
    #
    #@device_code_warntype interactive = true
    assemble_linear_kernel!(backend, WORKGROUP_SIZE)(
        y,
        f,
        domain,
        V,
        quadrature,
        rdhl;
        ndrange = size(y),
    )
end

@kernel function assemble_linear_kernel!(
    b,
    @Const(f),
    @Const(domain),
    @Const(V),
    @Const(quadrature),
    @Const(rdhl)
)
    # Here  `I` is a global index of a dof
    I = @index(Global)

    assemble_linear_elemental!(I, b, f, domain, V, quadrature, rdhl)
end

"""
Assemble the idof-th element of a linear form
"""
function assemble_linear_elemental!(
    idof::I,
    b::B,
    f::F,
    domain::D,
    V::TV,
    quadrature,
    rdhl::R,
) where {I, B, F, D, TV, R}
    for i in 1:get_n_elts(rdhl, idof)
        ielt = get_ielt(rdhl, idof, i)
        iloc = get_iloc(rdhl, idof, i)
        eltInfo = _get_index(domain, ielt)

        φ = MyShapeFunction(V, iloc)
        _f = f(φ)
        x = _compute_value(_f, eltInfo, quadrature)
        # @assert length(x) == 1 "Result is not scalar" # KO on CUDA
        # b[idof] += first(x) # KO on CUDA
        b[idof] += x[1] # quite bad because first element of `x` could be something else than "1"
    end
end

function _compute_value(f::F, eltInfo::E, quadrature::Q) where {F, E, Q}
    fᵥ = Bcube.materialize(f, eltInfo)
    value = integrate_on_ref_element(fᵥ, eltInfo, quadrature)
    return value
end