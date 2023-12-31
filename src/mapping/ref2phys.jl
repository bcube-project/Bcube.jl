
"""
normal(cnodes, ctype::AbstractEntityType, iside, ξ)

Normal vector of the `iside`th face of a cell, evaluated at position `ξ` in the face reference element.
So for the normal vector of the face of triangle living in a 3D space, `ξ` will be 1D (because the face
is a line, which 1D).

Beware this function needs the nodes `cnodes` and the type `ctype` of the cell (and not of the face).

TODO: If `iside` is positive, then the outward normal (with respect to the cell) is returned, otherwise
the inward normal is returned.
"""
function normal(cnodes, ctype::AbstractEntityType, iside, ξ)
    normal(topology_style(cnodes, ctype), cnodes, ctype, iside, ξ)
end

"""
    normal(::isCurvilinear, cnodes, ctype::AbstractEntityType, iside, ξ)

Compute the "face" normal vector of a curve at its `iside`-th side.

Note that the "face" normal vector of a curve is the "direction" vector at the given extremity.
"""
function normal(::isCurvilinear, cnodes, ctype::AbstractEntityType, iside, ξ)
    # mapping face-reference-element (here, a node) to cell-reference-element (here, a Line)
    # Since a Line has always only two nodes, the node is necessary the `iside`-th
    ξ_cell = coords(Line())[iside]

    return normalize(mapping_jacobian(cnodes, ctype, ξ_cell) .* normal(shape(ctype), iside))
end

"""
    normal(::isVolumic, cnodes, ctype::AbstractEntityType, iside, ξ)

Compute the normal of the `iside`-th side of a cell (same topology dim. as the space).

``n^{loc} = J^{-\\intercal} n^{ref}``
"""
function normal(::isVolumic, cnodes, ctype::AbstractEntityType, iside, ξ)
    # Cell shape
    cshape = shape(ctype)

    # Face parametrization to send ξ from ref-face-element to the ref-cell-element
    fp = mapping_face(cshape, iside) # mapping face-ref -> cell-ref

    # Inverse of the Jacobian matrix (but here `y` is in the cell-reference element)
    # Warning : do not use `mapping_inv_jacobian` which requires the knowledge of `mapping_inv` (useless here)
    Jinv(y) = mapping_jacobian_inv(cnodes, ctype, y)

    return normalize(transpose(Jinv(fp(ξ))) * normal(cshape, iside))
end

function normal(::isSurfacic, cnodes, ctype::AbstractEntityType, iside, ξ)
    # Get cell shape and face type and shape
    cshape = shape(ctype)
    ftype = facetypes(ctype)[iside]

    # Get face nodes
    fnodes = [cnodes[i] for i in faces2nodes(ctype)[iside]] # @ghislainb : need better solution to index

    # Get face direction vector (face Jacobian)
    u = mapping_jacobian(fnodes, ftype, ξ)

    # Get face parametrization function
    fp = mapping_face(cshape, iside) # mapping face-ref -> cell-ref

    # Compute surface jacobian
    J = mapping_jacobian(cnodes, ctype, fp(ξ))

    # Compute vector that will help orient outward normal
    orient = mapping(cnodes, ctype, fp(ξ)) - center(cnodes, ctype)

    # Normal direction
    n = J[:, 1] × J[:, 2] × u

    # Orient normal outward and normalize
    return normalize(orient ⋅ n .* n)
end

"""
    cell_normal(cnodes::AbstractArray{Node{2,T},N}, ctype::AbstractEntityType{1}, ξ) where {T, N}

Compute the cell normal vector of an entity of topology dimension equals to 1 in a 2D space,
i.e a curve in a 2D space. This vector is expressed in the cell-reference coordinate system.

Do not confuse the cell normal vector with the cell-side (i.e face) normal vector.

Method : the curve direction vector, u, is J/||J||. Then n = [-u.y, u.x].
"""
function cell_normal(
    cnodes::AbstractArray{Node{2, T}, N},
    ctype::AbstractEntityType{1},
    ξ,
) where {T, N}
    Jref = mapping_jacobian(cnodes, ctype, ξ)
    return normalize(SA[-Jref[2], Jref[1]])
end

"""
    cell_normal(cnodes::AbstractArray{Node{3,T},N}, ctype::AbstractEntityType{2}, ξ) where {T, N}

Compute the cell normal vector of an entity of topology dimension equals to 2 (a surface) in a 3D space.
This vector is expressed in the cell-reference coordinate system.

Do not confuse the cell normal vector with the cell-side (i.e face) normal vector.

"""
function cell_normal(
    cnodes::AbstractArray{Node{3, T}, N},
    ctype::AbstractEntityType{2},
    ξ,
) where {T, N}
    J = mapping_jacobian(cnodes, ctype, ξ)
    return normalize(J[:, 1] × J[:, 2])
end

"""
    center(cnodes, ctype::AbstractEntityType)

Return the center of the `AbstractEntityType` by mapping the center of the corresponding `Shape`.

# Warning
Do not use this function on a face of a cell : since the face is of dimension "n-1", the mapping
won't be appropriate.
"""
center(cnodes, ctype::AbstractEntityType) = mapping(cnodes, ctype, center(shape(ctype)))

"""
    shape_functions(fs::AbstractFunctionSpace, etype::AbstractEntityType, nodes, ξ)

Shape functions in the local coordinate system.

This function needs `mapping_inv` to be known for the given `AbstractEntityType`. This is
not always the case. Please not that it's always possible to solve a FE/DG problem without
using this function (working only in the ref element).

# Warning
Contrary to `grad_shape_functions`, which is evaluated in the cell-ref-element; `shape_functions`
is evaluated in the local element.

@ghislainb : I find this very dangerous...
"""
#function shape_functions(fs::AbstractFunctionSpace, etype::AbstractEntityType, nodes, ξ)
#    return x -> shape_functions(fs, shape(etype), mapping_inv(nodes, etype, x))
#end

"""
    grad_shape_functions(fs::AbstractFunctionSpace, etype::AbstractEntityType, nodes, ξ)

Gradient, with respect to the local coordinate system, of the shape functions. The result function is expressed in the reference
coordinate system :
``\\nabla \\lambda = \\frac{\\partial \\lambda_i}{\\partial x_j}(\\xi) = \\left( \\nabla (\\hat{\\lambda}) J^{-1}``
where ``J^{-1}`` is the inverse of the ref->loc mapping jacobian matrix.

# Implementation
We cannot use the `topology_style` to dispatch because this style is too specific to integration methods. For instance for
the integration it is important the consider any line as `isCurvilinear`. However for the gradient computation we must distinguish
a line in 1D, a line in 2D and a line in 3D...
# """
function grad_shape_functions(
    fs::AbstractFunctionSpace,
    n::Val{1},
    etype::AbstractEntityType,
    nodes,
    ξ,
)
    # Gradient of reference shape functions
    ∇λ = grad_shape_functions(fs, n, shape(etype), ξ)
    return ∇λ * mapping_jacobian_inv(nodes, etype, ξ)
end
function grad_shape_functions(
    fs::AbstractFunctionSpace,
    n::Val{1},
    etype::AbstractEntityType,
    nodes,
)
    ξ -> grad_shape_functions(fs, n, etype, nodes, ξ)
end

# alias for scalar
function grad_shape_functions(
    fs::AbstractFunctionSpace,
    etype::AbstractEntityType,
    nodes,
    ξ,
)
    grad_shape_functions(fs, Val(1), etype, nodes, ξ)
end
function grad_shape_functions(fs::AbstractFunctionSpace, etype::AbstractEntityType, nodes)
    grad_shape_functions(fs, Val(1), etype, nodes)
end

function grad_shape_functions(
    fs::AbstractFunctionSpace,
    n::Val{1},
    etype::AbstractEntityType,
    nodes,
    ξ::AbstractQuadratureNode,
)
    # Gradient of reference shape functions
    ∇λ = grad_shape_functions(fs, n, ξ)
    return ∇λ * mapping_jacobian_inv(nodes, etype, get_coord(ξ))
end

"""
    grad_shape_functions(::AbstractFunctionSpace, ::Val{N}, ::AbstractEntityType, nodes, ξ) where N

Return the gradient, in the local coordinate system, of shape functions (evaluated at reference position ξ)
for a vector finite element space. The result is an array of size `(n*Nc, n, d)` where `Nc` is the number of dofs
of one component (i.e scalar case), `n` is the size of the Finite Element Space, and `d` the number of spatial
dimensions.

Default version : the gradient shape functions are "replicated".

Specialize with a given FESpace for a custom behaviour.
"""
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

function grad_shape_functions(
    fs::AbstractFunctionSpace,
    n::Val{N},
    etype::AbstractEntityType,
    nodes,
    ξ,
) where {N}
    ∇λ_sca = grad_shape_functions(fs, Val(1), etype, nodes, ξ) # Matrix of size (ndofs_sca, nspa)
    ndofs_sca, nspa = size(∇λ_sca)
    a = SArray{Tuple{ndofs_sca, 1, nspa}, eltype(∇λ_sca)}(
        ∇λ_sca[i, k] for i in 1:ndofs_sca, j in 1:1, k in 1:nspa
    )
    ∇λ_vec = _block_diag_cat(a, n)
    return ∇λ_vec
end

function grad_shape_functions(
    fs::AbstractFunctionSpace,
    n::Val{N},
    etype::AbstractEntityType,
    nodes,
) where {N}
    ξ -> grad_shape_functions(fs, n, etype, nodes, ξ)
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

function grad_shape_functions_NA(
    fs::AbstractFunctionSpace,
    n::Val{N},
    etype::AbstractEntityType,
    nodes,
    ξ,
) where {N}
    ∇λ_sca = grad_shape_functions(fs, Val(1), etype, nodes, ξ) # Matrix of size (ndofs_sca, nspa)
    ∇λ_vec = _build_grad_impl(∇λ_sca, n)
    return ∇λ_vec
end

function grad_shape_functions_NA(
    fs::AbstractFunctionSpace,
    n::Val{N},
    etype::AbstractEntityType,
    nodes,
) where {N}
    ξ -> grad_shape_functions_NA(fs, n, etype, nodes, ξ)
end

"""
    grad_shape_functions(fs::AbstractFunctionSpace, ctype::AbstractEntityType{2}, cnodes::AbstractArray{Node{3,T},N}, ξ) where {T, N}

Gradient computation for a hypersurface : a surface (of topo dim 2) in a 3D space.
"""
function grad_shape_functions(
    fs::AbstractFunctionSpace,
    ctype::AbstractEntityType{2},
    cnodes::AbstractArray{Node{3, T}, N},
    ξ,
) where {T, N}
    # Corresponding shape
    s = shape(ctype)

    # Get reference shape functions
    λ = shape_functions(fs, s)

    # Mapping jacobian
    Jref = mapping_jacobian(cnodes, ctype, ξ)

    # Compute cell normal
    n⃗ = cell_normal(cnodes, ctype, ξ)

    # Form augmented jacobian (if Jref and n⃗ are StaticArrays, the below array is also a SA)
    J = [Jref n⃗]

    # Compute shape functions gradient : we "add a dimension" to the ref gradient
    # The original formulae is grad(lambda_i) = transpose(Jinv) * grad(lambda_ref_i)
    # But given our definition of grad(lambda) (matrix), the formulae becomes the below one
    ∇λ = hcat(grad_shape_functions(fs, s, ξ), zeros(ndofs(fs, s))) * inv(J)

    return ∇λ
end

"""
    grad_shape_functions(fs::AbstractFunctionSpace, ctype::AbstractEntityType{1}, cnodes::AbstractArray{Node{2,T},N}, ξ) where {T, N}

Gradient computation for a hypersurface : a line (of topo dim 1) in a 2D space.

Exact copy of `grad_shape_functions(fs::AbstractFunctionSpace, ctype::AbstractEntityType{2}, cnodes::AbstractArray{Node{3,T},N}, ξ) where {T, N}`,
but I don't feel fixing it right now cause we will soon improve the whole dispatch on grad_shape_functions`
"""
function grad_shape_functions(
    fs::AbstractFunctionSpace,
    ctype::AbstractEntityType{1},
    cnodes::AbstractArray{Node{2, T}, N},
    ξ,
) where {T, N}
    # Corresponding shape
    s = shape(ctype)

    # Get reference shape functions
    λ = shape_functions(fs, s)

    # Mapping jacobian
    Jref = mapping_jacobian(cnodes, ctype, ξ)

    # Compute cell normal
    n⃗ = cell_normal(cnodes, ctype, ξ)

    # Form augmented jacobian (if Jref and n⃗ are StaticArrays, the below array is also a SA)
    J = [Jref n⃗]

    # Compute shape functions gradient : we "add a dimension" to the ref gradient
    # The original formulae is grad(lambda_i) = transpose(Jinv) * grad(lambda_ref_i)
    # But given our definition of grad(lambda) (matrix), the formulae becomes the below one
    ∇λ = hcat(grad_shape_functions(fs, s, ξ), zeros(ndofs(fs, s))) * inv(J)

    return ∇λ
end

function grad_shape_functions(
    ::AbstractFunctionSpace,
    ::AbstractEntityType{1},
    ::AbstractArray{Node{3, T}, N},
    ξ,
) where {T, N}
    error("Line gradient in 3D not implemented yet")
end

""" get mesh cell centers coordinates (assuming perfectly flat cells)"""
function get_cell_centers(mesh::Mesh)
    c2n = connectivities_indices(mesh, :c2n)
    celltypes = cells(mesh)
    centers = map(1:ncells(mesh)) do icell
        ct = celltypes[icell]
        cn = get_nodes(mesh, c2n[icell])
        center(cn, ct)
    end
    return centers
end

"""
    interpolate(λ, q)

Create the interpolation function from a set of value on dofs and the shape functions, given by:
```math
    f(x) = \\sum_{i=1}^N q_i \\lambda_i(x)
```

So `q` is a vector whose size equals the number of dofs in the cell.
"""
function interpolate(λ, q)
    #@assert length(λ) === length(q) "Error : length of `q` must equal length of `lambda`"
    return ξ -> interpolate(λ, q, ξ)
end

interpolate(λ, q, ξ) = sum(λᵢ * qᵢ for (λᵢ, qᵢ) in zip(λ(ξ), q))

"""
    interpolate(λ, q, ncomps)

Create the interpolation function for a vector field given by a set of value on dofs and the shape functions.

The interpolation formulae is the same than `interpolate(λ, q)` but the result is a vector function. Here
`q` is a vector whose size equals the total number of dofs in the cell (all components mixed).

Note that the result function is expressed in the same coordinate system as the input shape functions (i.e reference
or local).
"""
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
