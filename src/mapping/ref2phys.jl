
"""
    normal(ctype::AbstractEntityType, cnodes, iside, ξ)
    normal(::TopologyStyle, ctype::AbstractEntityType, cnodes, iside, ξ)

Normal vector of the `iside`th face of a cell, evaluated at position `ξ` in the face reference element.
So for the normal vector of the face of triangle living in a 3D space, `ξ` will be 1D (because the face
is a line, which 1D).

Beware this function needs the nodes `cnodes` and the type `ctype` of the cell (and not of the face).

TODO: If `iside` is positive, then the outward normal (with respect to the cell) is returned, otherwise
the inward normal is returned.

# `::isCurvilinear`
Note that the "face" normal vector of a curve is the "direction" vector at the given extremity.

# `::isVolumic`
``n^{loc} = J^{-\\intercal} n^{ref}``

"""
function normal(ctype::AbstractEntityType, cnodes, iside, ξ)
    normal(topology_style(ctype, cnodes), ctype, cnodes, iside, ξ)
end

function normal(::isCurvilinear, ctype::AbstractEntityType, cnodes, iside, ξ)
    # mapping face-reference-element (here, a node) to cell-reference-element (here, a Line)
    # Since a Line has always only two nodes, the node is necessary the `iside`-th
    ξ_cell = coords(Line())[iside]

    return normalize(mapping_jacobian(ctype, cnodes, ξ_cell) .* normal(shape(ctype), iside))
end

function normal(::isSurfacic, ctype::AbstractEntityType, cnodes, iside, ξ)
    # Get cell shape and face type and shape
    cshape = shape(ctype)
    ftype = facetypes(ctype)[iside]

    # Get face nodes
    fnodes = [cnodes[i] for i in faces2nodes(ctype)[iside]] # @ghislainb : need better solution to index

    # Get face direction vector (face Jacobian)
    u = mapping_jacobian(ftype, fnodes, ξ)

    # Get face parametrization function
    fp = mapping_face(cshape, iside) # mapping face-ref -> cell-ref

    # Compute surface jacobian
    J = mapping_jacobian(ctype, cnodes, fp(ξ))

    # Compute vector that will help orient outward normal
    orient = mapping(ctype, cnodes, fp(ξ)) - center(ctype, cnodes)

    # Normal direction
    n = J[:, 1] × J[:, 2] × u

    # Orient normal outward and normalize
    return normalize(orient ⋅ n .* n)
end

function normal(::isVolumic, ctype::AbstractEntityType, cnodes, iside, ξ)
    # Cell shape
    cshape = shape(ctype)

    # Face parametrization to send ξ from ref-face-element to the ref-cell-element
    fp = mapping_face(cshape, iside) # mapping face-ref -> cell-ref

    # Inverse of the Jacobian matrix (but here `y` is in the cell-reference element)
    # Warning : do not use `mapping_inv_jacobian` which requires the knowledge of `mapping_inv` (useless here)
    Jinv(y) = mapping_jacobian_inv(ctype, cnodes, y)

    return normalize(transpose(Jinv(fp(ξ))) * normal(cshape, iside))
end

"""
    cell_normal(ctype::AbstractEntityType, cnodes, ξ) where {T, N}

Compute the cell normal vector of an entity of topology dimension equals to (d-1) in a n-D space,
for instance a curve in a 2D space. This vector is expressed in the cell-reference coordinate system.

Do not confuse the cell normal vector with the cell-side (i.e face) normal vector.

# Topology dimension 1
the curve direction vector, u, is J/||J||. Then n = [-u.y, u.x].

"""
function cell_normal(
    ctype::AbstractEntityType{1},
    cnodes::AbstractArray{Node{2, T}, N},
    ξ,
) where {T, N}
    Jref = mapping_jacobian(ctype, cnodes, ξ)
    return normalize(SA[-Jref[2], Jref[1]])
end

function cell_normal(
    ctype::AbstractEntityType{2},
    cnodes::AbstractArray{Node{3, T}, N},
    ξ,
) where {T, N}
    J = mapping_jacobian(ctype, cnodes, ξ)
    return normalize(J[:, 1] × J[:, 2])
end

"""
    center(ctype::AbstractEntityType, cnodes)

Return the center of the `AbstractEntityType` by mapping the center of the corresponding `Shape`.

# Warning
Do not use this function on a face of a cell : since the face is of dimension "n-1", the mapping
won't be appropriate.
"""
center(ctype::AbstractEntityType, cnodes) = mapping(ctype, cnodes, center(shape(ctype)))

"""
    grad_shape_functions(::AbstractFunctionSpace, ::Val{N}, ctype::AbstractEntityType, cnodes, ξ) where N

Gradient, with respect to the local coordinate system, of the shape functions associated to the `FunctionSpace`.

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
function grad_shape_functions(
    fs::AbstractFunctionSpace,
    n::Val{1},
    ctype::AbstractEntityType,
    cnodes,
    ξ,
)
    # Gradient of reference shape functions
    ∇λ = grad_shape_functions(fs, n, shape(ctype), ξ)
    return ∇λ * mapping_jacobian_inv(ctype, cnodes, ξ)
end

function grad_shape_functions(
    fs::AbstractFunctionSpace,
    n::Val{N},
    ctype::AbstractEntityType,
    cnodes,
    ξ,
) where {N}
    ∇λ_sca = grad_shape_functions(fs, Val(1), ctype, cnodes, ξ) # Matrix of size (ndofs_sca, nspa)
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

function grad_shape_functions(
    fs::AbstractFunctionSpace,
    ::Val{1},
    ctype::AbstractEntityType{2},
    cnodes::AbstractArray{<:Node{3}},
    ξ,
)
    return _grad_shape_functions_hypersurface(fs, ctype, cnodes, ξ)
end

function grad_shape_functions(
    fs::AbstractFunctionSpace,
    ::Val{1},
    ctype::AbstractEntityType{1},
    cnodes::AbstractArray{<:Node{2}},
    ξ,
)
    return _grad_shape_functions_hypersurface(fs, ctype, cnodes, ξ)
end

function _grad_shape_functions_hypersurface(fs, ctype, cnodes, ξ)
    # First, we compute the "augmented" jacobian.
    #
    # Rq: we could do this elsewhere, for instance in mapping.jl.
    # However, since this augmented jacobian already calls `mapping_jacobian`
    # it would not be as easy as specializing `mapping_jacobian` for the hypersurface
    # case. As long as this portion of code is not duplicated elsewhere, I prefer to
    # keep it here.
    s = shape(ctype)
    Jref = mapping_jacobian(ctype, cnodes, ξ)
    ν = cell_normal(ctype, cnodes, ξ)
    J = hcat(Jref, ν)

    # Compute shape functions gradient : we "add a dimension" to the ref gradient,
    # and then right-multiply by the inverse of the jacobian
    z = @SVector zeros(ndofs(fs, s))
    ∇λ = hcat(grad_shape_functions(fs, s, ξ), z) * inv(J)

    return ∇λ
end

function grad_shape_functions(
    ::AbstractFunctionSpace,
    ::AbstractEntityType{1},
    ::AbstractArray{Node{3}},
    ξ,
)
    error("Line gradient in 3D not implemented yet")
end

"""
    grad_function(f, n::Val{N}, ctype::AbstractEntityType, cnodes, ξ) where N

Compute the gradient of a function `f` on a point in the reference domain. `N` is the
size of the codomain of `f`.
"""
function grad_function(f, ::Val{1}, ctype::AbstractEntityType, cnodes, ξ)
    return transpose(mapping_jacobian_inv(ctype, cnodes, ξ)) * ForwardDiff.gradient(f, ξ)
end
function grad_function(f, ::Val{N}, ctype::AbstractEntityType, cnodes, ξ) where {N}
    return ForwardDiff.jacobian(f, ξ) * mapping_jacobian_inv(ctype, cnodes, ξ)
end

""" get mesh cell centers coordinates (assuming perfectly flat cells)"""
function get_cell_centers(mesh::Mesh)
    c2n = connectivities_indices(mesh, :c2n)
    celltypes = cells(mesh)
    centers = map(1:ncells(mesh)) do icell
        ctype = celltypes[icell]
        cnodes = get_nodes(mesh, c2n[icell])
        center(ctype, cnodes)
    end
    return centers
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
