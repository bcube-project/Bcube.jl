# Dev

This section is intended for `Bcube` developers.

## Interpolating a vector field on a curved surfacic element

Here we consider an $$n$$-dimensionnal surface in a $$\mathbb{R}^{n+1}$$ space, i.e an hypersurface (for instance a sphere in $$\mathbb{R}^3$$). To ease the notations, we consider 2-dimensionnal surface in $$\mathbb{R}^3$$. The physical coordinate system $$(x,y,z)$$ is associated to a vector basis $$\mathcal{B}_p = (e_x,e_y,e_z)$$. In the reference element, the coordinates are noted $$(\xi, \eta, \zeta)$$ and are associated to a local basis $$\mathcal{B}_r(\xi,\eta,\zeta) = (u,v,w)$$. The $$u$$, $$v$$ and $$w$$ are nothing else than the normalized $$\partial_\xi F$$, $$\partial_\eta F$$ and $$\nu$$, where $$F$$ is the reference to physical mapping, and $$\nu$$ the cell normal.Note that $$\mathcal{B}_p$$ is "constant" while $$\mathcal{B}_r$$ depends on the position in the reference element in the case of a curved element. In the rest of this section, we will use $$x$$ for $$(x,y,z)$$ and $$\xi$$ for $$(\xi, \eta, \zeta)$$.

Now, let's consider a discrete vector field $q$ over this cell. We restrict ourselves to nodal FE basis. For a "planar" (ie not curved) element the discrete representation of $q$ on a point $\xi$ in the reference element reads
```math
\begin{equation}
    \label{eqn-std-interpolation}
    q(\xi) = \sum_{i=1}^N q_i \hat{\lambda}_i(\xi)
\end{equation}
```
where $$N$$ is the number of dofs and $q_i$ are the dof values of the vector $q$ in the physical basis $$\mathcal{B}_p$$. Since we resctrict to Lagrange element, $$q_i = q(\xi_i)$$.

It gets harder when dealing with a curved element. Imagine that all $q_i$ have a given angle with respect to the surface. We would like the interpolation of $q$ to have this angle with the surface on any point $\xi$. A standard interpolation like ``\eqref{eqn-std-interpolation}`` doesn't ensure this property. Imagine that instead of knowning the $q_i$ (in the physical element), we know $\tilde{q}_i$, the dof values of $q$ in the reference coordinate system, ie in the basis $$\mathcal{B}_r$$. We would then write
```math
\begin{equation}
    \label{eqn-interpolation-in-ref}
    q(\xi) = P_\xi^x(\xi) \sum_{i=1}^N \tilde{q}_i \hat{\lambda}_i(\xi)
\end{equation}
```
where $$P_\xi^x(\xi)$$ is the change-of-basis matrix from $$\mathcal{B}_r(\xi)$$ to $$\mathcal{B}_p$$. In other words, the interpolation is performed in the reference element basis, and the result is mapped to the physical basis. Note that $$P_\xi^x(\xi)$$ is actually the jacobian matrix of the mapping $$F$$: $$P_\xi^x(\xi)=J(\xi)$$. Now if we don't know the $$\tilde{q}_i$$ but only the $$q_i$$ (ie the dofs in the physical basis), we look for the rotation / linear map $$R_\xi^x$$ to apply on each dof such that
```math
\begin{equation}
  \label{eqn-interpolation-in-phys-with-rot}
    q(\xi) = \sum_{i=1}^N R_\xi^x(\xi, \xi_i) q_i \hat{\lambda}_i(\xi).
\end{equation}
```
We used two arguments ($$\xi$$ and $$\xi_i$$) for $$R_\xi^x$$ to emphasize that for each value of $\xi$, there is one different $$R_\xi^x$$ to apply on each dof node $$\xi_i$$. Combining ``\eqref{eqn-interpolation-in-ref}`` with ``\eqref{eqn-interpolation-in-phys-with-rot}`` leads to
```math
    \forall i,~P_\xi^x(\xi)\tilde{q}_i = R_\xi^x(\xi, \xi_i) q_i.
```
Then, by definition the $\tilde{q}_i$ are the representation of $q_i$ in the reference basis, that is to say $$\tilde{q}_i = P_x^\xi(x_i) q_i$$, where $$P_x^\xi(x_i)$$ is the change-of-basis matrix from $$\mathcal{B}_p$$ to $$\mathcal{B}_r(F^{-1}(x_i))$$. And note that
```math
    P_x^\xi(x_i) = (P_\xi^x(F^{-1}(x_i)))^{-1} = (J(\xi_i))^{-1}.
```
Hence $$R_\xi^x(\xi, \xi_i)$$ is identified to $$J(\xi)J^{-1}(F(\xi_i))$$. Finally, the correct interpolation to apply is
```math
\begin{equation}
    \label{eqn-corrected-interpolation}
    q(\xi) = J(\xi) \sum_{i=1}^N (J(\xi_i))^{-1} q_i \hat{\lambda_i}(\xi_i).
\end{equation}
```

!!! note "Implementation details"
    It might be tempting to implement ``\eqref{eqn-interpolation-in-ref}`` instead of ``\eqref{eqn-corrected-interpolation}``. Although possible, note that equations are solved in the physical space in `Bcube`, so the "result" (or "rhs") of the equation will be in the physical space. If the dof values are (stored) in the reference space, any update of an `FEFunction` with the *rhs* would require a custom `set_dof_values!` function that apply the $$P_x^\xi$$ operation on each dof. Hence, the mesh would be required to pass to `set_dof_values!` etc, even for planar cases. For now, it's seem more reasonable to keep the dof in the physical space, and apply and a specific treatment for surfacic curved elements.