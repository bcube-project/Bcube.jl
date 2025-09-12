# Dev

This section is intended for `Bcube` developers.

## Interpolating a vector field on a curved surfacic element

Here we consider an $$n$$-dimensionnal surface in a $$\mathbb{R}^{n+1}$$ space, i.e an hypersurface (for instance a sphere in $$\mathbb{R}^3$$). To ease the notations, we consider a 2-dimensionnal surface in $$\mathbb{R}^3$$. The physical coordinate system $$(x,y,z)$$ is associated to the "universe" vector basis $$\mathcal{B}_p = (\vec{e}_x,\vec{e}_y,\vec{e}_z)$$. On each point of the surface, there also exists a so-called "local" vector basis $$\mathcal{B}_l(x,y,z)$$ formed by the surface tangent plane and the normal. Recall that this vector basis depends on the position on the surface. Finally, the vector basis associated with the reference element is noted $$\mathcal{B}_r$$, and the coordinates are noted $$(\xi,\eta)$$. Since our reference elements are "flat" (a square, a cube, etc), the basis $$\mathcal{B}_r$$ does not depend on the position in the reference element. Note that in practice $$\mathcal{B}_r$$ is actually identified to $$\mathcal{B}_p$$ by considering an additionnal direction along the normal of the reference element.

To ease the notations, we introduce the vectors $$\vec{x}=(x,y,z)$$ and $$\vec{\xi}=(\xi,\eta,\zeta)$$. The mapping between the reference element and the physical element is noted $$\vec{x} = \vec{F}(\vec{\xi})$$. For an hypersurface, since the mapping is $$\vec{F}(\vec{\xi}) = \vec{F}_\Gamma(\xi,\eta) + \vec{\nu}(\xi,\eta) \zeta$$ where $$\vec{F}_\Gamma$$ is the "classic" [reference to physical mapping](#geometry-and-mesh) and $$\vec{\nu}$$ is the surface normal vector (in $$\mathcal{B}_p$$). This allows to identify the $$\mathcal{B}_l(x,y,z)$$ basis vectors as $$(\partial_\xi \vec{F}_\Gamma, \partial_\eta \vec{F}_\Gamma, \vec{\nu})$$. This triplet of vectors form the mapping jacobian $$J(\xi)$$. This jacobian is nothing else than the change-of-basis operator/matrix between $$\mathcal{B}_l$$ and $$\mathcal{B}_p$$.

Now, let's a cell in the mesh of the surface, and consider a discrete vector field $\vec{q}$ over this cell. We restrict ourselves to nodal FE basis. For a "planar" (ie not curved) element the discrete representation of $\vec{q}$ on a point $\vec{\xi}$ in the reference element reads
```math
\begin{equation}
    \label{eqn-std-interpolation}
    \vec{q}(\vec{\xi}) = \sum_{i=1}^N \vec{q}_i \hat{\lambda}_i(\vec{\xi})
\end{equation}
```
where $$N$$ is the number of dofs and $\vec{q}_i$ are the dof values of the vector $\vec{q}$ in the physical basis $$\mathcal{B}_p$$. Since we resctrict to Lagrange element, $$\vec{q}_i = \vec{q}(\vec{\xi_i})$$.

It gets harder when dealing with a curved element. Imagine that all $\vec{q}_i$ have a given angle with respect to the surface. We would like the interpolation of $\vec{q}$ to have this angle with the surface on any point $$\vec{\xi}$$. A standard interpolation like ``\eqref{eqn-std-interpolation}`` doesn't ensure this property. Imagine that instead of knowning the $\vec{q}_i$ (in the physical element), we know $\vec{\tilde{q}}_i$, the dof values of $\vec{q}$ in the local basis $$\mathcal{B}_l$$. We would then write
```math
\begin{equation}
    \label{eqn-interpolation-in-ref}
    \vec{q}(\xi) = J(\xi) \sum_{i=1}^N \vec{\tilde{q}}_i \hat{\lambda}_i(\vec{\xi})
\end{equation}
```
where we recall that $$J$$ is the change-of-basis matrix from $$\mathcal{B}_l(\xi)$$ to $$\mathcal{B}_p$$. In other words, the interpolation is performed in the local basis, and the result is mapped to the physical basis. Now, if we don't know the $$\vec{\tilde{q}}_i$$ but only the $$\vec{q}_i$$ (ie the dofs in the physical basis), we look for the rotation / linear map $$R_\xi^x$$ to apply on each dof such that
```math
\begin{equation}
  \label{eqn-interpolation-in-phys-with-rot}
    \vec{q}(\vec{\xi}) = \sum_{i=1}^N R_\xi^x(\vec{\xi}, \vec{\xi}_i) \vec{q}_i \hat{\lambda}_i(\vec{\xi}).
\end{equation}
```
We used two arguments ($$\vec{\xi}$$ and $$\vec{\xi}_i$$) for $$R_\xi^x$$ to emphasize that for each point $\vec{\xi}$, there is one different $$R_\xi^x$$ to apply on each dof node $$\vec{\xi}_i$$. Combining ``\eqref{eqn-interpolation-in-ref}`` with ``\eqref{eqn-interpolation-in-phys-with-rot}`` leads to
```math
    \forall i,~J(\vec{\xi})\vec{\tilde{q}}_i = R_\xi^x(\vec{\xi}, \vec{\xi}_i) \vec{q}_i.
```
Then, by definition the $$\vec{\tilde{q}}_i$ are the representation of $\vec{q}_i$ in the local basis, that is to say $$\vec{\tilde{q}}_i = P_x^\xi(\vec{x_i}) \vec{q}_i$$, where $$P_x^\xi(\vec{x_i})$$ is the change-of-basis matrix from $$\mathcal{B}_p$$ to $$\mathcal{B}_l(F^{-1}(\vec{x_i}))$$. And note that
```math
    P_x^\xi(\vec{x_i}) = (P_\xi^x(\vec{F}^{-1}(\vec{x}_i)))^{-1} = (J(\vec{\xi}_i))^{-1}.
```
Hence, $$R_\xi^x(\vec{\xi}, \vec{\xi}_i)$$ is identified to $$J(\vec{\xi})J^{-1}(\vec{F}(\vec{\xi}_i))$$. Finally, the correct interpolation to apply is
```math
\begin{equation}
    \label{eqn-corrected-interpolation}
    \vec{q}(\vec{\xi}) = J(\vec{\xi}) \sum_{i=1}^N (J(\vec{\xi}_i))^{-1} \vec{q}_i \hat{\lambda_i}(\vec{\xi}_i).
\end{equation}
```

!!! note "Implementation details"
    It might be tempting to implement ``\eqref{eqn-interpolation-in-ref}`` instead of ``\eqref{eqn-corrected-interpolation}``. Although possible, note that equations are solved in the physical space in `Bcube`, so the "result" (or "rhs") of the equation will be in the physical space. If the dof values are (stored) in the reference space, any update of an `FEFunction` with the *rhs* would require a custom `set_dof_values!` function that apply the $$P_x^\xi$$ operation on each dof. Hence, the mesh would be required to pass to `set_dof_values!` etc, even for planar cases. For now, it's seem more reasonable to keep the dof in the physical space, and apply and a specific treatment for surfacic curved elements.