"""
Represent the affine system associated to a(u,v) = l(v), where `a` is bilinear, with eventual
Dirichlet conditions on `u`

# Dev notes
* we could avoid having `mesh` as an attribute by storing one (or the two) form : `a` and/or `l`.
Then the mesh can be retrieved from those two forms.
"""
struct AffineFESystem{
    Mat <: AbstractMatrix{<:Number},
    Vec <: AbstractVector{<:Number},
    TriFE,
    TesFE,
    M <: AbstractMesh,
    L,
}
    A::Mat
    b::Vec
    U::TriFE
    V::TesFE
    mesh::M
    linsolve!::L
end

"""
    AffineFESystem(
        A::AbstractMatrix{<:Number},
        b::AbstractVector{<:Number},
        U,
        V,
        mesh::AbstractMesh,
        linsolve! = default_linsolve!,
    )

Build an AffineFESystem from the bilinear form a and the linear form v.
`U` and `V` are the test FESpace and the trial FESpace respectively.

`linsolve!` should be a `Function`` of the form `f(y, A, x)` where `A` is the bilinear form
matrix, `x` the right hand side, and `y` a preallocated output vector in which the result
should be stored. The default solver is `(y, A, x) -> y .= A \\ x`.

# Warning
For now, `U` and `V` must be defined with the same FESpace(s) ("Petrov-Galerkin" not allowed)
"""
function AffineFESystem(a, l, U, V, linsolve! = default_linsolve!)
    # Preliminary check to ensure same FESpace
    if U isa MultiFESpace
        @assert all((_U, _V) -> parent(_U) isa typeof(parent(_V)), zip(U, V)) "U and V must be defined on same FEspace"
    else
        @assert parent(U) isa typeof(parent(V))
    end

    # Assemble the system
    A = assemble_bilinear(a, U, V)
    b = assemble_linear(l, V)

    # Retrieve Mesh from `a`
    op = a(NullOperator(), NullOperator())
    if op isa MultiIntegration
        measures = map(get_measure, (op...,))
        domains = map(get_domain, measures)
        meshes = map(get_mesh, domains)
        @assert all(x -> x == meshes[1], meshes) "All integrations must be defined on the same mesh"
        mesh = meshes[1]
    else
        measure = get_measure(op)
        domain = get_domain(measure)
        mesh = get_mesh(domain)
    end

    return AffineFESystem(A, b, U, V, mesh, linsolve!)
end

default_linsolve!(y, A, x) = y .= A \ x

_get_arrays(system::AffineFESystem) = (system.A, system.b)
_get_fe_spaces(system::AffineFESystem) = (system.U, system.V)

"""
    solve(system::AffineFESystem, t::Number = 0.0)
    solve!(u::SingleFieldFEFunction, system::AffineFESystem, t::Number = 0.0)

Solve the AffineFESystem, i.e invert the Ax=b system taking into account
the dirichlet conditions.

The out-of-place returns a `FEFunction` with the solution.
"""
function solve(system::AffineFESystem, t::Number = 0.0)
    U, _ = _get_fe_spaces(system)

    # Create FEFunction to hold the result
    u = FEFunction(U)

    # Solve
    solve!(u, system, t)

    return u
end

function solve!(u::SingleFieldFEFunction, system::AffineFESystem, t::Number = 0.0)
    A, b = _get_arrays(system)
    U, V = _get_fe_spaces(system)

    # Create the "Dirichlet" vector
    d = assemble_dirichlet_vector(U, V, system.mesh, t)

    # "step" the solution with dirichlet values
    b0 = b - A * d

    # Apply homogeneous dirichlet on A and b
    apply_homogeneous_dirichlet!(A, b0, U, V, system.mesh)

    # Inverse linear system
    x = Bcube.allocate_dofs(U)
    system.linsolve!(x, A, b0)

    # Update FEFunction
    set_dof_values!(u, x .+ d)
end
