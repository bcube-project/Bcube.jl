"""
Represent the affine system associated to a(u,v) = l(v), where `a` is bilinear, with eventual
Dirichlet conditions on `u`

# Dev notes
* we could avoid having `mesh` as an attribute by storing one (or the two) form : `a` and/or `l`.
Then the mesh can be retrieved from those two forms.
"""
struct AffineFESystem{T <: Number, Mat <: AbstractMatrix{<:Number}, TriFE, TesFE}
    A::Mat
    b::Vector{T}
    U::TriFE
    V::TesFE
    mesh::Mesh

    function AffineFESystem(
        A::AbstractMatrix{<:Number},
        b::Vector{T},
        U,
        V,
        mesh::Mesh,
    ) where {T}
        new{T, typeof(A), typeof(U), typeof(V)}(A, b, U, V, mesh)
    end
end

"""
Build an AffineFESystem from the bilinear form a and the linear form v.
`U` and `V` are the test FESpace and the trial FESpace respectively.

# Warning
For now, `U` and `V` must be defined with the same FESpace(s) ("Petrov-Galerkin" not authorised)
"""
function AffineFESystem(a, l, U, V)
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
    measure = get_measure(op)
    domain = get_domain(measure)
    mesh = get_mesh(domain)

    return AffineFESystem(A, b, U, V, mesh)
end

_get_arrays(system::AffineFESystem) = (system.A, system.b)
_get_fe_spaces(system::AffineFESystem) = (system.U, system.V)

"""
Solve the AffineFESystem, i.e invert the Ax=b system taking into account
the dirichlet conditions.

# Dev notes
* should we return an FEFunction instead of a Vector?
* we need to enable other solvers
"""
function solve(system::AffineFESystem, t::Number = 0.0; alg = nothing)
    U, _ = _get_fe_spaces(system)

    # Create FEFunction to hold the result
    u = FEFunction(U)

    # Solve
    solve!(u, system, t; alg)

    return u
end

function solve!(
    u::SingleFieldFEFunction,
    system::AffineFESystem,
    t::Number = 0.0;
    alg = nothing,
)
    A, b = _get_arrays(system)
    U, V = _get_fe_spaces(system)

    # Create the "Dirichlet" vector
    d = assemble_dirichlet_vector(U, V, system.mesh, t)

    # "step" the solution with dirichlet values
    b0 = b - A * d

    # Apply homogeneous dirichlet on A and b
    apply_homogeneous_dirichlet!(A, b0, U, V, system.mesh)

    # Inverse linear system
    prob = LinearSolve.LinearProblem(A, b0)
    sol = LinearSolve.solve(prob, alg)

    # Update FEFunction
    set_dof_values!(u, sol.u .+ d)
end
