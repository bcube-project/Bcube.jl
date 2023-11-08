module ShallowWater #hide
const dir = string(@__DIR__, "/../../") # Bcube dir
include(dir * "src/Bcube.jl")
using .Bcube
using WriteVTK
using LinearAlgebra
using StaticArrays
using ForwardDiff

# # Shallow water equations
# Following "A conservative Saint-Venant type model to describe the dynamics of thien partially wetting films with
# regularized forces at the contact line".
# The gravity is noted ``g = g_n \vec{e_n} + \vec{g_t}`` (note that ``g_n`` is a scalar while ``g_t`` is a vector). The goal is to solve:
# ```math
# \begin{cases}
#   \partial_t \rho h + \nabla \cdot \rho h u = 0 \\
#   \partial_t \rho h u + \nabla \cdot \mathcal{F}_{\rho h u} = h \left( \rho g_t - \nabla P_{gaz} \right) - \tilde{\tau}_{wall} + \tilde{\tau}_{air}
# \end{cases}
# ```
#
# To simplify a little bit, we assume a constant density. The systems becomes:
# ```math
# \begin{cases}
#   \partial_t h + \nabla \cdot h u = 0 \\
#   \partial_t h u + \nabla \cdot \mathcal{F}_{hu} = h \left( g_t - \nabla P_{gaz} \right) - \tau_{wall} + \tau_{air}
# \end{cases}
# ```
# where
# ```math
# \tau_{wall} = \frac{3 \nu u}{h + b} - \frac{\tau_{air}h}{2(h+b)}
# ```
# ``b`` being a slip length and
# ```math
# \mathcal{F}_{h u} = h u \otimes u + g_n \frac{h^2}{2} \mathcal{I} + \frac{1}{\rho}\left[h \partial_h e_d(h) - e_d(h) \right]
# - \frac{\gamma_{lg}}{\rho \sqrt{1 + ||\nabla h||^2}} + \frac{1}{\rho} \gamma_{lg} h \kappa
# ```
#
# ## Explicit time integration
# ```math
# \begin{cases}
#   h^{n+1} = h^n - \Delta t \nabla \cdot h u^n \\
#   h u^{n+1} =  hu^n - \Delta t \left[
#     \nabla \cdot \mathcal{F}_{hu}(h^n,hu^n)
#     - h^n \left( g_t - \nabla P_{gaz} \right) + \tau_{wall}(h^n, hu^n) - \tau_{air}
#   \right]
# \end{cases}
# ```
#
# ## Implicit time integration
# ```math
# \begin{cases}
#   h^{n+1} = h^n - \Delta t \nabla \cdot h u^{n+1} \\
#   h u^{n+1} =  hu^n - \Delta t \left[
#     \nabla \cdot \mathcal{F}_{hu}(h^{n+1},hu^{n+1})
#     - h^{n+1} \left( g_t - \nabla P_{gaz} \right) + \tau_{wall}(h^{n+1}, hu^{n+1}) - \tau_{air}
#   \right]
# \end{cases}
# ```
#
# ## IMEX time integration (not finished / not relevant)
# The wall friction term, ``\tau_w`` is singular when ``h \rightarrow 0``. To overcome this difficulty, an implicit-explicit (IMEX)
# scheme is used : all terms are integrated explicitely except the wall friction. More precisely, the system is first written:
# ```math
# \begin{cases}
#   h^{n+1} = h^n - \Delta t \nabla \cdot h u^n \\
#   h u^{n+1} =  hu^n - \Delta t \left[
#     \nabla \cdot \mathcal{F}_{hu}(h^n,hu^n)
#     - h^n \left( g_t - \nabla P_{gaz} \right) + \tau_{wall}(h^{n+1}, hu^{n+1}) - \tau_{air}
#   \right]
# \end{cases}
# ```
# At each time step, the mass equation can be solved explicitely independantly from the momentum equation. Besides, the wall
# friction can be expressed as:
# ```math
# \tau_{wall} = \frac{3 \nu hu^{n+1}}{{h^{n+1}}^2} - \frac{\tau_{air}}{2}
# ```
# where the slipping length, ``b``, is neglected (which is fine when working with an implicit formulation). The momentum
# equation can then be rearranged to obtain:
# ```math
#   h u^{n+1}\left( 1 + \frac{3  \nu \Delta t }{{h^{n+1}}^2} \right) =  hu^n - \Delta t \left[
#     \nabla \cdot \mathcal{F}_{hu}(h^n,hu^n)
#     - h^n \left( g_t - \nabla P_{gaz} \right) - \frac{3}{2}\tau_{air}
#   \right]
# ```
# Moving the multiplying factor to the right-hand-side, we finally obtain:
# ```math
#   h u^{n+1} = \frac{{h^{n+1}}^2}{{h^{n+1}}^2 + 3 \nu \Delta t} \left[
#     hu^n - \Delta t \left[
#       \nabla \cdot \mathcal{F}_{hu}(h^n,hu^n)
#       - h^n \left( g_t - \nabla P_{gaz} \right) - \frac{3}{2}\tau_{air}
#     \right]
#  \right]
# ```
#
#
# ## Weak form
# First we write the different equation with a full explicit scheme to improve clarity.
#
# ### Mass conservation equation
# We multiply the equation by a test function ``\phi_{h}`` and integrate on a control volume ``\Omega``. After an integration by parts,
# we obtain (for an explicit integration time scheme):
# ```math
# \int_{\Omega} h^{n+1} \phi_{h} \mathrm{\,d}\Omega = \int_{\Omega} h^n \phi_{h} \mathrm{\,d}\Omega
# + \Delta t \left[ \int_{\Omega} h u^n \cdot \nabla \phi_{h} \mathrm{\,d}\Omega
# - \oint_{\Gamma} F_{h}^*(h^n, h u^n) \phi_{h} \mathrm{\,d} \Gamma \right]
# ```
# where ``F^*_{h}`` is the numerical flux corresponding to ``hu``.
#
# ### Momentum conservation equation
# We first consider the case without contact line forces nor curvature. Multiplying by a test function ``\phi_{h u}`` and integrating
# by parts leads to:
# ```math
#   \int_{\Omega} h u^{n+1} \phi_{h u} \mathrm{\,d}\Omega = \int_{\Omega} h u^n \phi_{h u} \mathrm{\,d}\Omega
#   + \Delta t \left[
#     \int_{\Omega} \left[
#         \mathcal{F}^n \cdot \nabla \phi_{h u}
#         + \left( h^n(g_t - \nabla P_g) - \tau_w + \tau_a \right) \phi_{h u}
#     \right] \mathrm{\,d}\Omega
#     - \oint_{\Gamma} F_{h u}^*(h^n, h u^n) \phi_{h u} \mathrm{\,d} \Gamma
#   \right]
# ```
# where ``F^*_{h u}`` is the numerical flux corresponding to ``h u \otimes u + g_n h^2 /2 \mathcal{I}``.
#

const ε = eps()

"""
    Inverse of the mass matrix in a given cell.
"""
function inv_mass_matrix(λ, cnodes, ctype, order)
    M = integrate_ref(ξ -> ⊗(λ(ξ)), cnodes, ctype, order)
    return inv(M)
end

"""
    Build the inverse of mass matrix for all mesh cells
    @TODO : use projection.L2_projector
"""
function build_mass_matrix_inv(mesh, h::CellVariable, hu::CellVariable, order_h, order_hu)
    ## Get cell -> node connectivity and cell types
    c2n = connectivities_indices(mesh, :c2n)
    cellTypes = cells(mesh)

    ## Function spaces
    fs_h  = function_space(h)
    fs_hu = function_space(hu)

    ## Integration order
    qorder_h  = Val(order_h^2)
    qorder_hu = Val(order_hu^2)

    ## Allocate
    iM_h  = [zeros(ntuple(i -> ndofs(fs_h, shape(cellTypes[icell])), 2)) for icell in 1:ncells(mesh)]
    iM_hu = [zeros(ntuple(i -> ndofs(fs_hu, shape(cellTypes[icell])), 2)) for icell in 1:ncells(mesh)]

    ## Loop over cells
    for icell in 1:ncells(mesh)

        ## Cell type and shape
        ctype = cellTypes[icell]
        cshape = shape(ctype)

        ## Alias for nodes
        cnodes = get_nodes(mesh, c2n[icell])

        ## Shape functions and gradients
        λ_h  = shape_functions(fs_h, cshape)
        λ_hu = shape_functions(fs_hu, cshape)

        ## Compute inverse of mass matrix
        iM_h[icell]  .= inv_mass_matrix(λ_h, cnodes, ctype, qorder_h)
        iM_hu[icell] .= inv_mass_matrix(λ_hu, cnodes, ctype, qorder_hu)
    end

    return iM_h, iM_hu
end

"""
    ξᵢ and ξⱼ are in the cell-ref-element
"""
function upwind(valᵢ, valⱼ, hᵢ, hⱼ, huᵢ, huⱼ, nᵢⱼ, ξᵢ, ξⱼ)
    upwind(valᵢ(ξᵢ), valⱼ(ξⱼ), hᵢ(ξᵢ), hⱼ(ξⱼ), huᵢ(ξᵢ), huⱼ(ξⱼ), nᵢⱼ)
end

function upwind(valᵢ, valⱼ, hᵢ, hⱼ, huᵢ, huⱼ, nᵢⱼ)
    ## Centered velocity
    #vij = 0.5 * (vᵢ + vⱼ) ⋅ nᵢⱼ

    ## Face velocity
    vij = (huᵢ + huⱼ) / (hᵢ + hⱼ + ε) ⋅ nᵢⱼ

    ## min/max
    vij⁺ = max(0.0, vij)
    vij⁻ = min(0.0, vij)

    return valᵢ * vij⁺ + valⱼ * vij⁻
end

centered(valᵢ, valⱼ, nᵢⱼ) = 0.5 * (valᵢ + valⱼ) * nᵢⱼ
centered(valᵢ, valⱼ, nᵢⱼ, ξᵢ, ξⱼ) = centered(valᵢ(ξᵢ), valⱼ(ξⱼ), nᵢⱼ)

function momentum_flux(hᵢ, hⱼ, huᵢ, huⱼ, gnᵢ, gnⱼ, nᵢⱼ)

    ## Centered height
    hij = 0.5 * (hᵢ + hⱼ)

    ## Centered gravity
    gnij = 0.5 * (gnᵢ + gnⱼ)

    ## Upwind for convection + center for gravity
    return upwind(huᵢ, huⱼ, hᵢ, hⱼ, huᵢ, huⱼ, nᵢⱼ) + gnij * hij^2 / 2 .* nᵢⱼ # erreur?
end

function momentum_flux(hᵢ, hⱼ, huᵢ, huⱼ, gnᵢ, gnⱼ, nᵢⱼ, ξᵢ, ξⱼ)
    return momentum_flux(hᵢ(ξᵢ), hⱼ(ξⱼ), huᵢ(ξᵢ), huⱼ(ξⱼ), gnᵢ(ξᵢ), gnⱼ(ξⱼ), nᵢⱼ)
end

"""
    Compute flux on each face of the mesh.

    Same shape functions assumed for all variables for now
"""
function compute_flux!(
    mesh::AbstractMesh,
    sys::System,
    h::CellVariable,
    hu::CellVariable,
    params,
    q::Vector,
    t::Real,
    f::Vector,
)
    ## Unpack params
    g = params.g

    ## Get connectivities
    c2n = connectivities_indices(mesh, :c2n)
    f2n = connectivities_indices(mesh, :f2n)
    f2c = connectivities_indices(mesh, :f2c)

    ## Cell and face types
    cellTypes = cells(mesh)
    faceTypes = faces(mesh)

    ## Alias : number of components of hu
    n_hu = ncomponents(hu)

    ## Integration order
    order_h  = Val(3 * params.order_h) # lucky guess (for the term '∇λ * h  * u')
    order_hu = Val(3 * params.order_hu) # lucky guess (for the term '∇λ * hu * u')

    ## Function spaces
    fs_h  = function_space(h)
    fs_hu = function_space(hu)

    ## Reset flux vector
    f .= 0.0

    # Loop on all the inner faces
    for kface in inner_faces(mesh)
        ## Face nodes, type and shape
        ftype = faceTypes[kface]
        fnodes = get_nodes(mesh, f2n[kface])

        ## Neighbor cell i
        i = f2c[kface][1]
        xᵢ = get_nodes(mesh, c2n[i])
        ctᵢ = cellTypes[i]
        shapeᵢ = shape(ctᵢ)
        sideᵢ = cell_side(ctᵢ, c2n[i], f2n[kface])
        fpᵢ = mapping_face(shapeᵢ, sideᵢ)
        λᵢ_h = shape_functions(fs_h, shapeᵢ)
        λᵢ_hu = shape_functions(fs_hu, shapeᵢ)
        hᵢ = interpolate(λᵢ_h, q[dof(h, i)])
        huᵢ = interpolate(λᵢ_hu, q[dof(hu, i)], n_hu)
        gnᵢ(ξ) = g ⋅ cell_normal(xᵢ, ctᵢ, ξ)

        ## Neighbor cell j
        j = f2c[kface][2]
        xⱼ = get_nodes(mesh, c2n[j])
        ctⱼ = cellTypes[j]
        shapeⱼ = shape(ctⱼ)
        sideⱼ = cell_side(ctⱼ, c2n[j], f2n[kface])
        nv = length(faces2nodes(shapeⱼ, sideⱼ)) # number of vertices of the face
        iglob_vertices_of_face_of_cell_j =
            [c2n[j][faces2nodes(ctⱼ, sideⱼ)[l]] for l in 1:nv]
        g2l = indexin(f2n[kface][1:nv], iglob_vertices_of_face_of_cell_j)
        fpⱼ = mapping_face(shapeⱼ, sideⱼ, g2l)
        λⱼ_h = shape_functions(fs_h, shape(ctⱼ))
        λⱼ_hu = shape_functions(fs_hu, shape(ctⱼ))
        hⱼ = interpolate(λⱼ_h, q[dof(h, j)])
        huⱼ = interpolate(λⱼ_hu, q[dof(hu, j)], n_hu)
        gnⱼ(ξ) = g ⋅ cell_normal(xⱼ, ctⱼ, ξ)

        ##------- Mass conservation
        ## Flux definition in face-ref-element
        fluxn = (nᵢⱼ, ξ) -> upwind(hᵢ, hⱼ, hᵢ, hⱼ, huᵢ, huⱼ, nᵢⱼ, fpᵢ(ξ), fpⱼ(ξ))

        ## Compute flux contribution of face `kface` to cell `i`, performing a surfacic integration
        g_ref = ξ -> λᵢ_h(fpᵢ(ξ)) .* fluxn(normal(xᵢ, ctᵢ, sideᵢ, ξ), ξ)
        f[dof(sys, h, i)] .+= integrate_ref(g_ref, fnodes, ftype, order_h)

        ## Compute flux contribution of face `kface` to cell `j`, performing a surfacic integration
        g_ref = ξ -> λⱼ_h(fpⱼ(ξ)) * fluxn(-normal(xⱼ, ctⱼ, sideⱼ, ξ), ξ)
        f[dof(sys, h, j)] .-= integrate_ref(g_ref, fnodes, ftype, order_h)

        ##------- Momentum conservation
        ## Flux definition in face-ref-element
        fluxn = (nᵢⱼ, ξ) -> momentum_flux(hᵢ, hⱼ, huᵢ, huⱼ, gnᵢ, gnⱼ, nᵢⱼ, fpᵢ(ξ), fpⱼ(ξ))

        ## Compute flux contribution of face `kface` to cell `i`, performing a surfacic integration
        g_ref = ξ -> λᵢ_hu(fpᵢ(ξ)) * transpose(fluxn(normal(xᵢ, ctᵢ, sideᵢ, ξ), ξ))
        flux⁺ = integrate_ref(g_ref, fnodes, ftype, order_hu)

        ## Compute flux contribution of face `kface` to cell `j`, performing a surfacic integration
        g_ref = ξ -> λⱼ_hu(fpⱼ(ξ)) * transpose(fluxn(-normal(xⱼ, ctⱼ, sideⱼ, ξ), ξ))
        flux⁻ = integrate_ref(g_ref, fnodes, ftype, order_hu)

        ## Add contribution to the residual of each adjacent cells
        i_dofs = dof(sys, hu, i)
        j_dofs = dof(sys, hu, j)
        f[i_dofs] .+= vec(flux⁺)
        f[j_dofs] .-= vec(flux⁻)
    end

    ## Loop on all the boundary of type 'faces'
    for tag in keys(mesh.bc_faces)

        ## Loop over this boundary faces
        for kface in boundary_faces(mesh, tag)

            ## Face nodes and type
            ftype = faceTypes[kface]
            fnodes = get_nodes(mesh, f2n[kface])

            ## Neighbor cell i
            i = f2c[kface][1]
            cnodes = get_nodes(mesh, c2n[i])
            ctype = cellTypes[i]
            s = shape(ctype)
            side = cell_side(ctype, c2n[i], f2n[kface])
            F = mapping(cnodes, ctype)
            fp = mapping_face(s, side)
            λ_h = shape_functions(fs_h, s)
            λ_hu = shape_functions(fs_hu, s)
            hᵢ = interpolate(λ_h, q[dof(sys, h, i)])
            huᵢ = interpolate(λ_hu, q[dof(sys, hu, i)], n_hu)
            gn(ξ) = g ⋅ cell_normal(cnodes, ctype, ξ)

            ##----- Mass conservation
            if haskey(params.cdts, (tag, :h))

                ## Get associated boundary condition
                cdt = params.cdts[(tag, :h)]

                ## Dofs
                i_dofs = dof(sys, h, i)

                ## Flux boundary condition
                if type(cdt) == :flux
                    ## Append flux contribution of face `kface` to cell `i`
                    g_ref =
                        ξ ->
                            normal(cnodes, ctype, side, ξ) ⋅ apply(cdt, F(fp(ξ)), t) *
                            λ_h(fp(ξ))
                    f[i_dofs] .+= integrate_ref(g_ref, fnodes, ftype, order_h)

                elseif type(cdt) == :outlet
                    fluxn = (nᵢⱼ, ξ) -> upwind(hᵢ, hᵢ, hᵢ, hᵢ, huᵢ, huᵢ, nᵢⱼ, fp(ξ), fp(ξ))
                    g_ref = ξ -> λ_h(fp(ξ)) .* fluxn(normal(cnodes, ctype, side, ξ), ξ)
                    f[i_dofs] .+= integrate_ref(g_ref, fnodes, ftype, order_h)
                end
            end

            ##----- Momentum conservation
            if haskey(params.cdts, (tag, :hu))

                ## Get associated boundary condition
                cdt = params.cdts[(tag, :hu)]

                ## Dofs
                i_dofs = dof(sys, hu, i)

                ## Flux boundary condition
                if (type(cdt) == :flux)
                    ## Append flux contribution of face `kface` to cell `i`
                    g_ref =
                        ξ ->
                            λ_hu(fp(ξ)) * transpose(
                                apply(cdt, F(fp(ξ)), t) .* normal(cnodes, ctype, side, ξ),
                            )
                    f[i_dofs] .+= vec(integrate_ref(g_ref, fnodes, ftype, order_hu))

                elseif (type(cdt) == :outlet)
                    fluxn =
                        (nᵢⱼ, ξ) ->
                            momentum_flux(hᵢ, hᵢ, huᵢ, huᵢ, gn, gn, nᵢⱼ, fp(ξ), fp(ξ))
                    g_ref =
                        ξ ->
                            λ_hu(fp(ξ)) *
                            transpose(fluxn(normal(cnodes, ctype, side, ξ), ξ))
                    f[i_dofs] .+= vec(integrate_ref(g_ref, fnodes, ftype, order_hu))
                end
            end
        end # end loop on boundary faces
    end # end loop on bnd tags
end

function rhs!(
    mesh::AbstractMesh,
    sys::System,
    h::CellVariable,
    hu::CellVariable,
    params,
    t::Real,
    rhs::Vector,
    q::Vector,
)
    ## Unpack params
    g = params.g
    ν = params.ν
    b = params.b

    ## Flux
    f = similar(q) # needed for ForwardDiff
    compute_flux!(mesh, sys, h, hu, params, q, t, f)

    ## Cell -> node connectivity and cell types
    c2n = connectivities_indices(mesh, :c2n)
    cellTypes = cells(mesh)

    ## Identity matrix
    Id = SMatrix{spacedim(mesh), spacedim(mesh)}(1.0I)

    ## Alias : number of components of hu
    n_hu = ncomponents(hu)

    ## Integration order
    order_h  = Val(3 * params.order_h) # lucky guess (for the term '∇λ * h  * u')
    order_hu = Val(3 * params.order_hu) # lucky guess (for the term '∇λ * hu * u')

    ## Function spaces
    fs_h  = function_space(h)
    fs_hu = function_space(hu)

    ## Loop over cells
    for icell in 1:ncells(mesh)
        ## Unpack cache
        iM_h  = params.iM_h[icell]
        iM_hu = params.iM_hu[icell]

        ## Cell type and shape
        ctype = cellTypes[icell]
        cshape = shape(ctype)

        ## Alias for nodes
        cnodes = get_nodes(mesh, c2n[icell])

        ## Cell mapping : ref -> loc
        F = mapping(cnodes, ctype)

        ## Cell normal
        n(ξ) = cell_normal(cnodes, ctype, ξ)

        ## Shape functions and gradients
        λ_h   = shape_functions(fs_h, cshape)
        λ_hu  = shape_functions(fs_hu, cshape)
        ∇λ_h  = grad_shape_functions(fs_h, ctype, cnodes)
        ∇λ_hu = grad_shape_functions(fs_hu, ctype, cnodes)

        ## Alias
        i_h  = dof(sys, h, icell)
        i_hu = dof(sys, hu, icell)

        ## Interpolations
        hᵢ = interpolate(λ_h, q[i_h])
        huᵢ = interpolate(λ_hu, q[i_hu], Val(n_hu))
        u(ξ) = huᵢ(ξ) / (hᵢ(ξ) + ε) # `+ ε` to avoid division by zero

        ## Gravity
        gn(ξ) = g ⋅ n(ξ)          # scalar
        gt(ξ) = g - gn(ξ) .* n(ξ) # vector

        ## External sources (ρ is already included)
        ∇Pg(ξ) = params.∇Pg(F(ξ))
        τₐ(ξ) = params.τₐ(F(ξ))

        ## Wall friction
        τw(ξ) = wall_friction(hᵢ(ξ), u(ξ), ν, τₐ(ξ), b)

        ## Mass conservation
        FV = integrate_ref(ξ -> ∇λ_h(ξ) ⋅ huᵢ(ξ), cnodes, ctype, order_h)
        @views rhs[i_h] .= iM_h * (FV .- f[i_h])

        # Momentum conservation
        FV = integrate_ref(
            ξ ->
                ∇λ_hu(ξ) * transpose(huᵢ(ξ) * transpose(u(ξ)) + gn(ξ) * hᵢ(ξ)^2 / 2 * Id) +
                λ_hu(ξ) .* transpose(hᵢ(ξ) * (gt(ξ) - ∇Pg(ξ)) - τw(ξ) + τₐ(ξ)),
            cnodes,
            ctype,
            order_hu,
        )
        @views rhs[i_hu] .= vec(iM_hu * (FV - reshape(f[i_hu], size(FV))))
    end
end

"""
    Explicit step in time
"""
function explicit_step!(
    mesh::AbstractMesh,
    sys,
    h,
    hu,
    params,
    rhs::Vector,
    dq::Vector,
    q::Vector,
    t::Real,
)
    rhs!(mesh, sys, h, hu, params, t, rhs, q)
    dq .= params.Δt .* rhs
end

"""
    Implicit step in time
"""
function implicit_step!(
    mesh::AbstractMesh,
    sys,
    h,
    hu,
    params,
    rhs::Vector,
    dq::Vector,
    q::Vector,
    t::Real,
)
    # Unpack
    Δt = params.Δt

    # Compute RHS
    rhs!(mesh, sys, h, hu, params, t, rhs, q)
    #@show rhs

    # RHS Jacobian
    J = ForwardDiff.jacobian((rhs, q) -> rhs!(mesh, sys, h, hu, params, t, rhs, q), rhs, q)
    #display(J)

    # Invert
    dq .= (I - Δt .* J) \ (Δt .* rhs)
end

"""

Return the wall friction effort divided by ρ

Note : input τₐ is already divided by ρ
"""
function wall_friction(h, u, ν, τₐ, b)
    return (6 * ν * u - τₐ * h) / (2 * (h + b))
end

# We are all set to solve a linear transport equation. However, we will add two more things to ease the solution VTK output : a
# structure to store the vtk filename and the number of iteration:
mutable struct VtkHandler
    basename::Any
    ite::Any
    VtkHandler(basename) = new(basename, 0)
end

# ... and a method to interpolate the discontinuous solution on cell centers and to append it to the VTK file:
function append_vtk(vtk, mesh, sys::System, h::CellVariable, hu::CellVariable, q, t)
    ## Values on center
    set_values!(h, q[dof(sys, h)])
    set_values!(hu, q[dof(sys, hu)])
    cv2val = var_on_centers((h, hu))

    ## Write
    dict_vars = Dict(
        "h" => (cv2val[h], VTKCellData()),
        "hu" => (transpose(cv2val[hu]), VTKCellData()),
    )
    write_vtk(vtk.basename, vtk.ite, t, mesh, dict_vars; append = vtk.ite > 0)

    ## Update counter
    vtk.ite += 1
end

# Physical settings
const g      = [0.0, 0.0] # gravity vector
const ρ      = 1000 # water density
const ν      = 1e-6 # water cinematic viscosity (1e-6 at 20C)
const b      = 1e-9 # slipping length
const τₐ(x)  = [0.0, 0.0] # Tangent gas friction (divided by rho)
const ∇Pg(x) = [0.0, 0.0] # Tangent gas pressure gradient (divided by rho)
const h_in   = 1e-3 # (m)
const u_in   = [1e-2, 0.0] # (m/s) - column vector

# Numerical settings
const lx = 1.0 # Domain length
const nx = 51 # Number of mesh points in `x` direction
const ny = 0 # Number of mesh points in `y` direction
const nite = 2000 # Number of time iterations
const Δt = lx / (norm(u_in) * (nx - 1)) # time step
const order_h = 0  # discretization order for `h`
const order_hu = 0 # discretization order for `hu`
const nout = 100 # Maximum number of VTK outputs

# Build mesh
mesh = line_mesh(nx; names = ("West", "East")) # 1D mesh in a 1D space
mesh = transform(mesh, x -> [x[1], 0.0]) # 1D mesh in a 2D space

# Create variables
fes_h = FESpace(FunctionSpace(:Taylor, order_h), :discontinuous)
fes_hu = FESpace(FunctionSpace(:Taylor, order_hu), :discontinuous; size = spacedim(mesh))
h = CellVariable(:h, mesh, fes_h)
hu = CellVariable(:hu, mesh, fes_hu)
sys = System((h, hu))

# Boundary condition
cdt_h_in  = BoundaryCondition(:flux, (x, t) -> h_in .* u_in)
cdt_hu_in = BoundaryCondition(:flux, (x, t) -> h_in .* u_in .^ 2)
cdts      = Dict((boundary_tag(mesh, "West"), :h) => cdt_h_in, (boundary_tag(mesh, "West"), :hu) => cdt_hu_in, (boundary_tag(mesh, "East"), :h) => BoundaryCondition(:outlet, nothing), (boundary_tag(mesh, "East"), :hu) => BoundaryCondition(:outlet, nothing))

# Inverse mass matrix
iM_h, iM_hu = build_mass_matrix_inv(mesh, h, hu, order_h, order_hu)

# Then we create a `NamedTuple` to hold the simulation parameters.
params = (
    order_h = order_h,
    order_hu = order_hu,
    iM_h = iM_h,
    iM_hu = iM_hu,
    cdts = cdts,
    g = g,
    ρ = ρ,
    ν = ν,
    b = b,
    τₐ = τₐ,
    ∇Pg = ∇Pg,
    Δt = Δt,
)

# Let's allocate the unknown vector and set it to zero. Along with this vector, we also allocate the "increment" vector.
nd = get_ndofs(sys)
q = zeros(nd)
q[1:3:end] .= h_in
q[2:3:end] .= h_in .* u_in[1]
dq = zeros(size(q))
rhs = zeros(size(q))

# Init vtk handler
vtk = VtkHandler(dir * "myout/shallow_water")

# Init time
t = 0.0

# Save initial solution
append_vtk(vtk, mesh, sys, h, hu, q, t)

# Let's loop to solve the equation.
for i in 1:nite
    ## Infos
    println("\nIteration ", i)

    ## Step forward in time
    #explicit_step!(mesh, dhl, params, rhs, dq, q, t)
    implicit_step!(mesh, sys, h, hu, params, rhs, dq, q, t)
    q .+= dq
    global t += Δt
    #@show q

    ## Write solution to file (respecting max. number of output)
    if (i % Int(max(floor(nite / nout), 1)) == 0)
        append_vtk(vtk, mesh, sys, h, hu, q, t)
    end
end

@show Δt

end #hide
