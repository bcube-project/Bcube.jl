"""
    ∂λξ_∂x(::AbstractFunctionSpace, ::Val{N}, ctype::AbstractEntityType, cnodes, ξ) where N

Gradient, with respect to the physical coordinate system, of the shape functions associated to the `FunctionSpace`.

Depending on the value of `N`, the shape functions are interpreted as associated to a scalar FESpace (N = 1) or
a vector FESpace. For a vector FESpace, the result gradient is an array of size `(n*Nc, n, d)` where `Nc` is the
number of dofs of one component (i.e scalar case), `n` is the size of the FESpace, and `d` the number of spatial
dimensions.

Default version : the gradient shape functions are "replicated".

Specialize with a given FESpace for a custom behaviour.

# Implementation
We cannot use the `topology_style` to dispatch because this style is too specific to integration methods. For instance for
the integration it is important the consider any line as `isCurvilinear`. However for the gradient computation we must distinguish
a line in 1D, a line in 2D and a line in 3D...
# """
function ∂λξ_∂x end

function ∂λξ_∂x(fs::AbstractFunctionSpace, n::Val{1}, ctype::AbstractEntityType, cnodes, ξ)
    # Gradient of reference shape functions
    ∇λ = ∂λξ_∂ξ(fs, n, shape(ctype), ξ)
    return ∇λ * mapping_jacobian_inv(ctype, cnodes, ξ)
end

function ∂λξ_∂x(
    fs::AbstractFunctionSpace,
    n::Val{N},
    ctype::AbstractEntityType,
    cnodes,
    ξ,
) where {N}
    ∇λ_sca = ∂λξ_∂x(fs, Val(1), ctype, cnodes, ξ) # Matrix of size (ndofs_sca, nspa)
    ndofs_sca, nspa = size(∇λ_sca)
    a = SArray{Tuple{ndofs_sca, 1, nspa}, eltype(∇λ_sca)}(
        ∇λ_sca[i, k] for i in 1:ndofs_sca, j in 1:1, k in 1:nspa
    )
    ∇λ_vec = _block_diag_cat(a, n)
    return ∇λ_vec
end

@generated function _block_diag_cat(
    a::SArray{Tuple{N1, N2, N3}},
    ::Val{N},
) where {N1, N2, N3, N}
    T = eltype(a)
    z = zero(T)
    exprs = [:(a[$i, $j, $k]) for i in 1:N1, j in 1:N2, k in 1:N3]
    exprs0 = [:($z) for i in 1:N1, j in 1:N2, k in 1:N3]
    row_exprs = []
    mat_exprs = []
    for j in 1:N
        for i in 1:N
            i == j ? _exprs = exprs : _exprs = exprs0
            i == 1 ? row_exprs = _exprs : row_exprs = [row_exprs; _exprs]
        end
        j == 1 ? mat_exprs = row_exprs : mat_exprs = [mat_exprs row_exprs]
    end
    N1_glo = N1 * N
    N2_glo = N2 * N
    T = eltype(a)
    return quote
        SArray{Tuple{$N1_glo, $N2_glo, N3}, $T}(tuple($(mat_exprs...)))
    end
end

@generated function _build_grad_impl(a::SMatrix{N1, N2}, ::Val{N}) where {N1, N2, N}
    T = eltype(a)
    z = zero(T)
    exprs = [:(a[$i, $j]) for i in 1:N1, j in 1:N2]
    exprs0 = Any[:($z) for i in 1:N, j in 1:N2]

    mat_exprs = []
    for n in 1:N
        for k in 1:N1
            _exprs = deepcopy(exprs0)
            _exprs[n, :] = exprs[k, :]
            push!(mat_exprs, _exprs)
        end
    end
    mat_exprs2 = Any[]
    for x in mat_exprs
        if N == 1
            push!(mat_exprs2, :(SVector{$N2}(tuple($(x...)))))
        else
            push!(mat_exprs2, :(SMatrix{$N, $N2}(tuple($(x...)))))
        end
    end
    return quote
        tuple($(mat_exprs2...))
    end
end

function ∂λξ_∂x_hypersurface(
    fs::AbstractFunctionSpace,
    ::Val{1},
    ctype::AbstractEntityType{2},
    cnodes::AbstractArray{<:Node{3}},
    ξ,
)
    return _∂λξ_∂x_hypersurface(fs, ctype, cnodes, ξ)
end

function ∂λξ_∂x_hypersurface(
    fs::AbstractFunctionSpace,
    ::Val{1},
    ctype::AbstractEntityType{1},
    cnodes::AbstractArray{<:Node{2}},
    ξ,
)
    return _∂λξ_∂x_hypersurface(fs, ctype, cnodes, ξ)
end

function _∂λξ_∂x_hypersurface(fs, ctype, cnodes, ξ)
    # First, we compute the "augmented" jacobian.
    J = mapping_jacobian_hypersurface(ctype, cnodes, ξ)

    # Compute shape functions gradient : we "add a dimension" to the ref gradient,
    # and then right-multiply by the inverse of the jacobian
    s = shape(ctype)
    z = @SVector zeros(ndofs(fs, s))
    ∇λ = hcat(∂λξ_∂ξ(fs, s, ξ), z) * inv(J)

    return ∇λ
end

function ∂λξ_∂x_hypersurface(
    ::AbstractFunctionSpace,
    ::AbstractEntityType{1},
    ::AbstractArray{Node{3}},
    ξ,
)
    error("Line gradient in 3D not implemented yet")
end

"""
    ∂fξ_∂x(f, n::Val{N}, ctype::AbstractEntityType, cnodes, ξ) where N

Compute the gradient, with respect to the physical coordinates, of a function `f` on a point
in the reference domain. `N` is the size of the codomain of `f`.
"""
function ∂fξ_∂x(f::F, ::Val{1}, ctype::AbstractEntityType, cnodes, ξ) where {F}
    return transpose(mapping_jacobian_inv(ctype, cnodes, ξ)) * ForwardDiff.gradient(f, ξ)
end
function ∂fξ_∂x(f::F, ::Val{N}, ctype::AbstractEntityType, cnodes, ξ) where {F, N}
    return ForwardDiff.jacobian(f, ξ) * mapping_jacobian_inv(ctype, cnodes, ξ)
end

function ∂fξ_∂x_hypersurface(f, ::Val{1}, ctype::AbstractEntityType, cnodes, ξ)
    # Gradient in the reference domain. Add missing dimensions. Warning : we always
    # consider a hypersurface (topodim = spacedim - 1) and not a line in R^3 for instance.
    # Hence we always add only one 0.
    ∇f = ForwardDiff.gradient(f, ξ)
    ∇f = vcat(∇f, 0)

    # Then, we compute the "augmented" jacobian.
    J = mapping_jacobian_hypersurface(ctype, cnodes, ξ)

    return transpose(inv(J)) * ∇f
end

function ∂fξ_∂x_hypersurface(f, ::Val{N}, ctype::AbstractEntityType, cnodes, ξ) where {N}
    # Gradient in the reference domain. Add missing dimensions. Warning : we always
    # consider a hypersurface (topodim = spacedim - 1) and not a line in R^3 for instance.
    # Hence we always add only one 0.
    ∇f = ForwardDiff.jacobian(f, ξ)
    z = @SVector zeros(N)
    ∇f = hcat(∇f, z)

    # Then, we compute the "augmented" jacobian.
    J = mapping_jacobian_hypersurface(ctype, cnodes, ξ)

    return ∇f * inv(J)
end

"""
    interpolate(λ, q)
    interpolate(λ, q, ncomps)

Create the interpolation function from a set of value on dofs and the shape functions, given by:
```math
    f(x) = \\sum_{i=1}^N q_i \\lambda_i(x)
```

So `q` is a vector whose size equals the number of dofs in the cell.

If `ncomps` is present, create the interpolation function for a vector field given by a set of
value on dofs and the shape functions.

The interpolation formulae is the same than `interpolate(λ, q)` but the result is a vector function. Here
`q` is a vector whose size equals the total number of dofs in the cell (all components mixed).

Note that the result function is expressed in the same coordinate system as the input shape functions
(i.e reference or local).
"""
function interpolate(λ, q)
    #@assert length(λ) === length(q) "Error : length of `q` must equal length of `lambda`"
    return ξ -> interpolate(λ, q, ξ)
end

interpolate(λ, q, ξ) = sum(λᵢ * qᵢ for (λᵢ, qᵢ) in zip(λ(ξ), q))

function interpolate(λ, q::SVector{N}, ::Val{ncomps}) where {N, ncomps}
    return x -> transpose(reshape(q, Size(Int(N / ncomps), ncomps))) * λ(x)
end

function interpolate(λ, q, ::Val{ncomps}) where {ncomps}
    return x -> transpose(reshape(q, :, ncomps)) * λ(x)
end

function interpolate(λ, q, ncomps::Integer)
    return ξ -> interpolate(λ, q, ncomps, ξ)
end

function interpolate(λ, q, ncomps::Integer, ξ)
    return transpose(reshape(q, :, ncomps)) * λ(ξ)
end

function interpolate(λ, q::SVector)
    return x -> transpose(q) * λ(x)
end

# """
# WARNING : INTERPOLATION IN LOCAL ELEMENT FOR THE MOMENT
# """
# @inline function interpolate(q, dhl::DeprecatedDofHandler, icell, ctype::AbstractEntityType, cnodes, varname::Val{N}) where {N}
#     fs = function_space(dhl,varname)
#     λ = x -> shape_functions(fs, shape(ctype), mapping_inv(cnodes, ctype, x))
#     ncomp = ncomps(dhl,varname)
#     return interpolate(λ, q[dof(dhl, icell, shape(ctype), varname)], Val(ncomp))
# end

# """
# WARNING : INTERPOLATION IN LOCAL ELEMENT FOR THE MOMENT
# """
# function interpolate(q, dhl::DeprecatedDofHandler, icell, ctype::AbstractEntityType, cnodes)
#     _varnames = name.(getvars(dhl))
#     _nvars = nvars(dhl)
#     _ncomps = ncomps.(getvars(dhl))

#     # Gather all Variable interpolation functions in a Tuple
#     w = ntuple(i->interpolate(q, dhl, icell, ctype, cnodes, Val(_varnames[i])), _nvars)

#     # Build a 'vector-valued' function, one row per variable and component
#     return x -> vcat(map(f->f(x), w)...)
# end
