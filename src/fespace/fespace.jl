
const AOS_DEFAULT = true # default value for Array Of Struct / Struct of Array

"""
Abstract type to represent an finite-element space of size `S`. See `SingleFESpace`
for more details about what looks like a finite-element space.

# Devs notes
All subtypes should implement the following functions:
* `get_function_space(feSpace::AbstractFESpace)`
* `get_shape_functions(feSpace::AbstractFESpace, shape::AbstractShape)`
* `get_cell_shape_functions(feSpace::AbstractFESpace, shape::AbstractShape)`
* `get_ndofs(feSpace::AbstractFESpace)`
* `is_continuous(feSpace::AbstractFESpace)`

Alternatively, you may define a "parent" to your structure by implementing
the `Base.parent` function. Then, all the above functions will be redirected
to the "parent" FESpace.
"""
abstract type AbstractFESpace{S} end

"""
Return the size `S` associated to `AbstractFESpace{S}`.
"""
get_size(::AbstractFESpace{S}) where {S} = S

"""
Return the size `S`(= number of components) associated to `AbstractFESpace{S}`.
"""
get_ncomponents(feSpace::AbstractFESpace) = get_size(feSpace)

"""
Return the `FunctionSpace` (eventually multiple spaces) associated to
the `AbstractFESpace`.
"""
function get_function_space(feSpace::AbstractFESpace)
    get_function_space(parent(feSpace))
end

"""
Return the shape functions associated to the `AbstractFESpace`.
"""
function get_shape_functions(feSpace::AbstractFESpace, shape::AbstractShape)
    get_shape_functions(parent(feSpace), shape)
end

"""
Return the shape functions associated to the `AbstractFESpace` in "packed" form:
 λ(x) = (λ₁(x),...,λᵢ(x),...λₙ(x)) for the `n` dofs.
"""
function get_cell_shape_functions(feSpace::AbstractFESpace, shape::AbstractShape)
    get_cell_shape_functions(parent(feSpace), shape)
end

"""
Return the total number of dofs of the FESpace, taking into account the
continuous/discontinuous type of the space. If the FESpace contains itself
several FESpace (see MultiFESpace), the sum of all dofs is returned.
"""
get_ndofs(feSpace::AbstractFESpace) = get_ndofs(parent(feSpace))

function get_ndofs(feSpace::AbstractFESpace, shape::AbstractShape)
    get_ndofs(get_function_space(feSpace), shape) * get_ncomponents(feSpace)
end

function get_dofs(feSpace::AbstractFESpace, icell::Int, n::Val{N}) where {N}
    get_dofs(parent(feSpace), icell, n)
end

"""
Return the dofs indices for the cell `icell`

Result is an array of integers.
"""
get_dofs(feSpace::AbstractFESpace, icell::Int) = get_dofs(parent(feSpace), icell)

is_continuous(feSpace::AbstractFESpace) = is_continuous(parent(feSpace))
is_discontinuous(feSpace::AbstractFESpace) = !is_continuous(feSpace)

_get_dof_handler(feSpace::AbstractFESpace) = _get_dof_handler(parent(feSpace))
_get_dhl(feSpace::AbstractFESpace) = _get_dof_handler(feSpace)

"""
Return the boundary tags where a Dirichlet condition applies
"""
function get_dirichlet_boundary_tags(feSpace::AbstractFESpace)
    get_dirichlet_boundary_tags(parent(feSpace))
end

"""
    allocate_dofs(feSpace::AbstractFESpace, T = Float64)

Allocate a vector with a size equal to the number of dof of the FESpace, with the type `T`.
For a MultiFESpace, a vector of the total size of the space is returned (and not a Tuple of vectors)
"""
function allocate_dofs(feSpace::AbstractFESpace, T = Float64)
    allocate_dofs(parent(feSpace), T)
end

abstract type AbstractSingleFESpace{S, FS} <: AbstractFESpace{S} end

"""
An finite-element space (FESpace) is basically a function space,
associated to degrees of freedom (on a mesh).

A FESpace can be either scalar (to represent a Temperature for instance)
or vector (to represent a Velocity). In case of a "vector" `SingleFESpace`,
all the components necessarily share the same `FunctionSpace`.
"""
struct SingleFESpace{S, FS <: AbstractFunctionSpace} <: AbstractSingleFESpace{S, FS}
    fSpace::FS # function space
    dhl::DofHandler # degrees of freedom of this FESpace
    isContinuous::Bool # finite-element or discontinuous-galerkin
    dirichletBndTags::Vector{Int} # mesh boundary tags where Dirichlet condition applies
end

Base.parent(feSpace::SingleFESpace) = feSpace
get_function_space(feSpace::SingleFESpace) = feSpace.fSpace

function get_shape_functions(feSpace::SingleFESpace, shape::AbstractShape)
    fSpace = get_function_space(feSpace)
    domainstyle = DomainStyle(fSpace)

    _λs = shape_functions_vec(fSpace, Val(get_size(feSpace)), shape)
    λs = map(
        f -> ShapeFunction(f, domainstyle, Val(get_size(feSpace)), fSpace),
        _svector_to_tuple(_λs),
    )
    return λs
end
_svector_to_tuple(a::SVector{N}) where {N} = ntuple(i -> a[i], Val(N))

function get_cell_shape_functions(feSpace::SingleFESpace, shape::AbstractShape)
    fSpace = get_function_space(feSpace)
    domainstyle = DomainStyle(fSpace)
    return CellShapeFunctions(domainstyle, Val(get_size(feSpace)), fSpace, shape)
end

function get_multi_shape_function(feSpace::SingleFESpace, shape::AbstractShape)
    return MultiShapeFunction(get_shape_functions(feSpace, shape))
end

get_ncomponents(feSpace::SingleFESpace) = get_size(feSpace)
is_continuous(feSpace::SingleFESpace) = feSpace.isContinuous

_get_dof_handler(feSpace::SingleFESpace) = feSpace.dhl

get_dofs(feSpace::SingleFESpace, icell::Int) = get_dof(feSpace.dhl, icell)
function get_dofs(feSpace::SingleFESpace, icell::Int, n::Val{N}) where {N}
    get_dof(feSpace.dhl, icell, n)
end
get_ndofs(feSpace::SingleFESpace) = get_ndofs(_get_dhl(feSpace))

get_dirichlet_boundary_tags(feSpace::SingleFESpace) = feSpace.dirichletBndTags

"""
    SingleFESpace(
        fSpace::AbstractFunctionSpace,
        mesh::AbstractMesh,
        dirichletBndNames = String[];
        size::Int = 1,
        isContinuous::Bool = true,
        kwargs...
    )

Build a finite element space (scalar or vector) from a `FunctionSpace` and a `Mesh`.

# Arguments
- `fSpace::AbstractFunctionSpace` : the function space associated to the `FESpace`
- `mesh::AbstractMesh` : the mesh on which the `FESpace` is discretized
- `dirichletBndNames = String[]` : list of mesh boundary labels where a Dirichlet condition applies

# Keywords
- `size::Int = 1` : the number of components of the `FESpace`
- `isContinuous::Bool = true` : if `true`, a continuous dof numbering is created. Otherwise, dof lying
on cell nodes or cell faces are duplicated, not shared (discontinuous dof numbering)
- `kwargs` : for things such as parallel cache (internal/dev usage only)
"""
function SingleFESpace(
    fSpace::AbstractFunctionSpace,
    mesh::AbstractMesh,
    dirichletBndNames = String[];
    size::Int = 1,
    isContinuous::Bool = true,
    kwargs...,
)
    dhl = DofHandler(mesh, fSpace, size, isContinuous)

    # Rq : cannot use a "map" here because `dirichletBndNames` can be a Set
    dirichletBndTags = Int[]
    bndNames = values(boundary_names(mesh))
    for name in dirichletBndNames
        @assert name ∈ bndNames "Error with the Dirichlet condition on '$name' : this is not a boundary name. Boundary names are : $bndNames"
        push!(dirichletBndTags, boundary_tag(mesh, name))
    end

    return SingleFESpace{size, typeof(fSpace)}(fSpace, dhl, isContinuous, dirichletBndTags)
end

@inline allocate_dofs(feSpace::SingleFESpace, T = Float64) = zeros(T, get_ndofs(feSpace))

"""
A TrialFESpace is basically a SingleFESpace plus other attributes (related to boundary conditions)

# Dev notes
* we cannot directly store Dirichlet values on dofs because the Dirichlet values needs "time" to apply
"""
struct TrialFESpace{S, FE <: AbstractSingleFESpace} <: AbstractFESpace{S}
    feSpace::FE
    dirichletValues::Dict{Int, Function} # <boundary tag> => <dirichlet value (function of x, t)>
end

"""
    TrialFESpace(feSpace, dirichletValues)
    TrialFESpace(
        fSpace::AbstractFunctionSpace,
        mesh::AbstractMesh,
        dirichlet::Dict{String} = Dict{String, Any}();
        size::Int = 1,
        isContinuous::Bool = true,
        kwargs...
    )
    TrialFESpace(
        fSpace::AbstractFunctionSpace,
        mesh::AbstractMesh,
        type::Symbol,
        dirichlet::Dict{String} = Dict{String, Any}();
        size::Int = 1,
        kwargs...
    )

Build a trial finite element space.

See [`SingleFESpace`](@ref) for hints about the function arguments. Only arguments specific to
`TrialFESpace` are detailed below.

# Arguments
- `dirichlet::Dict{String} = Dict{String, Any}()` : dictionnary specifying the Dirichlet
    valued-function (or function) associated to each mesh boundary label. The function `f(x,t)`
    to apply is expressed in the physical coordinate system. Alternatively, a constant value
    can be provided instead of a function.
- `type::Symbol` : `:continuous` or `:discontinuous`

# Warning
For now the Dirichlet condition can only be applied to nodal bases.

# Examples
```julia-repl
julia> mesh = one_cell_mesh(:line)
julia> fSpace = FunctionSpace(:Lagrange, 2)
julia> U = TrialFESpace(fSpace, mesh)
julia> V = TrialFESpace(fSpace, mesh, :discontinuous; size = 3)
julia> W = TrialFESpace(fSpace, mesh, Dict("North" => 3., "South" => (x,t) -> t .* x))
```

"""
function TrialFESpace(
    fSpace::AbstractFunctionSpace,
    mesh::AbstractMesh,
    dirichlet::Dict{String} = Dict{String, Any}();
    size::Int = 1,
    isContinuous::Bool = true,
    kwargs...,
)
    # Build FESpace
    feSpace = SingleFESpace(fSpace, mesh, keys(dirichlet); size, isContinuous, kwargs...)

    # Transform any constant value into a function of (x,t)
    dirichletValues = Dict(
        boundary_tag(mesh, k) => (v isa Function ? v : (x, t) -> v) for (k, v) in dirichlet
    )

    return TrialFESpace(feSpace, dirichletValues)
end

function TrialFESpace(feSpace, dirichletValues)
    TrialFESpace{get_size(feSpace), typeof(feSpace)}(feSpace, dirichletValues)
end

function TrialFESpace(
    fSpace::AbstractFunctionSpace,
    mesh::AbstractMesh,
    type::Symbol,
    dirichlet::Dict{String} = Dict{String, Any}();
    size::Int = 1,
    kwargs...,
)
    @assert type ∈ (:continuous, :discontinuous) "Invalid variable type. Must be ':continuous' or ':discontinuous'"
    TrialFESpace(
        fSpace,
        mesh,
        dirichlet;
        size,
        isContinuous = type == :continuous,
        kwargs...,
    )
end

"""
A MultiplierFESpace can be viewed as a set of independant P0 elements.
It is used to define Lagrange multipliers and assemble the associated augmented system (the system that adds the multipliers as unknowns).
"""
function MultiplierFESpace(mesh::AbstractMesh, size::Int = 1, kwargs...)
    fSpace = FunctionSpace(:Lagrange, 0)

    iglob = collect(1:size)
    offset = zeros(ncells(mesh), size)
    for i in 1:size
        offset[:, i] .= i - 1
    end
    ndofs = ones(ncells(mesh), size)
    ndofs_tot = length(unique(iglob))
    dhl = DofHandler(iglob, offset, ndofs, ndofs_tot)

    feSpace = SingleFESpace{size, typeof(fSpace)}(fSpace, dhl, true, Int[])

    return TrialFESpace{size, typeof(feSpace)}(feSpace, Dict{Int, Function}())
end

"""
Return the values associated to a Dirichlet condition
"""
get_dirichlet_values(feSpace::TrialFESpace) = feSpace.dirichletValues
get_dirichlet_values(feSpace::TrialFESpace, ibnd::Int) = feSpace.dirichletValues[ibnd]

"""
A TestFESpace is basically a SingleFESpace plus other attributes (related to boundary conditions)
"""
struct TestFESpace{S, FE <: AbstractSingleFESpace} <: AbstractFESpace{S}
    feSpace::FE
end

"""
    TestFESpace(trialFESpace::TrialFESpace)
    TestFESpace(
        fSpace::AbstractFunctionSpace,
        mesh::AbstractMesh,
        dirichletBndNames = String[];
        size::Int = 1,
        isContinuous::Bool = true,
        kwargs...,
    )

Build a test finite element space.

A `TestFESpace` can be built from a `TrialFESpace`. See [`SingleFESpace`](@ref)
for hints about the function arguments. Only arguments specific to
`TrialFESpace` are detailed below.

# Examples
```julia-repl
julia> mesh = one_cell_mesh(:line)
julia> fSpace = FunctionSpace(:Lagrange, 2)
julia> U = TrialFESpace(fSpace, mesh)
julia> V = TestFESpace(U)
```
"""
function TestFESpace(
    fSpace::AbstractFunctionSpace,
    mesh::AbstractMesh,
    dirichletBndNames = String[];
    size::Int = 1,
    isContinuous::Bool = true,
    kwargs...,
)
    # Build FESpace
    feSpace = SingleFESpace(fSpace, mesh, dirichletBndNames; size, isContinuous, kwargs...)

    return TestFESpace(feSpace)
end

TestFESpace(feSpace) = TestFESpace{get_size(feSpace), typeof(feSpace)}(feSpace)
TestFESpace(trialFESpace::TrialFESpace) = TestFESpace(parent(trialFESpace))

const TrialOrTest{S, FE} = Union{TrialFESpace{S, FE}, TestFESpace{S, FE}}

Base.parent(tfeSpace::TrialOrTest) = tfeSpace.feSpace

"""
# Devs notes
All subtypes should implement the following functions:
* `get_fespace(mfeSpace::AbstractMultiFESpace)`
* `get_mapping(mfeSpace::AbstractMultiFESpace)`
* `get_dofs(mfeSpace::AbstractMultiFESpace, icell::Int)`
* `get_shape_functions(mfeSpace::AbstractMultiFESpace, shape::AbstractShape)`
* `get_cell_shape_functions(mfeSpace::AbstractMultiFESpace, shape::AbstractShape)`
"""
abstract type AbstractMultiFESpace{N, FE} end

"""
    get_fespace(mfeSpace::AbstractMultiFESpace, iSpace)
    get_fespace(mfeSpace::AbstractMultiFESpace)

Return the i-th FESpace composing this `AbstractMultiFESpace`. If no index is provided,
the tuple of FESpace composing this `AbstractMultiFESpace`` is returnted.
"""
get_fespace(mfeSpace::AbstractMultiFESpace) = get_fespace(parent(mfeSpace))
get_fespace(mfeSpace::AbstractMultiFESpace, iSpace) = get_fespace(mfeSpace)[iSpace]

"""
    get_mapping(mfeSpace::AbstractMultiFESpace, iSpace)
    get_mapping(mfeSpace::AbstractMultiFESpace)

Return the mapping for the ith `FESpace` composing the `MultiFESpace`.
If no index is provided, the tuple of mapping for each `FESpace`` is returnted.
"""
get_mapping(mfeSpace::AbstractMultiFESpace) = get_mapping(parent(mfeSpace))
get_mapping(mfeSpace::AbstractMultiFESpace, iSpace) = get_mapping(mfeSpace)[iSpace]

""" Number of `FESpace` composing the `MultiFESpace` """
get_n_fespace(::AbstractMultiFESpace{N}) where {N} = N

_is_AoS(mfeSpace::AbstractMultiFESpace) = _is_AoS(parent(mfeSpace))

Base.iterate(mfeSpace::AbstractMultiFESpace) = iterate(get_fespace(mfeSpace))
Base.iterate(mfeSpace::AbstractMultiFESpace, state) = iterate(get_fespace(mfeSpace), state)
Base.length(mfeSpace::AbstractMultiFESpace) = get_n_fespace(mfeSpace)

get_dofs(mfeSpace::AbstractMultiFESpace, icell::Int) = get_dofs(parent(mfeSpace), icell)
function get_shape_functions(mfeSpace::AbstractMultiFESpace, shape::AbstractShape)
    get_shape_functions(parent(mfeSpace), shape)
end
function get_cell_shape_functions(mfeSpace::AbstractMultiFESpace, shape::AbstractShape)
    get_cell_shape_functions(parent(mfeSpace), shape)
end

"""
A `MultiFESpace` represents a "set" of TrialFESpace or TestFESpace.
This structure provides a global dof numbering for each FESpace.

`N` is the number of FESpace contained in this `MultiFESpace`.

Note that the FESpace can be different from each other (one continous,
one discontinuous; one scalar, one vector...)
"""
struct MultiFESpace{N, FE <: Tuple{Vararg{AbstractFESpace, N}}} <:
       AbstractMultiFESpace{N, FE}
    feSpaces::FE
    mapping::NTuple{N, Vector{Int}}
    arrayOfStruct::Bool
end

const AbstractMultiTestFESpace{N} = AbstractMultiFESpace{N, <:Tuple{Vararg{TestFESpace, N}}}

const AbstractMultiTrialFESpace{N} =
    AbstractMultiFESpace{N, <:Tuple{Vararg{TrialFESpace, N}}}

Base.parent(mfeSpace::MultiFESpace) = mfeSpace

"""
Return the tuple of FESpace composing this MultiFESpace
"""
get_fespace(mfeSpace::MultiFESpace) = mfeSpace.feSpaces
get_mapping(mfeSpace::MultiFESpace) = mfeSpace.mapping
_is_AoS(mfeSpace::MultiFESpace) = mfeSpace.arrayOfStruct

""" Total number of dofs contained in this MultiFESpace """
function get_ndofs(mfeSpace::AbstractMultiFESpace)
    sum(feSpace -> get_ndofs(feSpace), get_fespace(mfeSpace))
end

"""
    get_dofs(feSpace::MultiFESpace, icell::Int)

Return the dofs indices for the cell `icell` for each single-feSpace.
Result is a tuple of array of integers, where each array of integers
are the indices relative to the numbering of each singleFESpace.

# Warning:
Combine `get_dofs` with `get_mapping` if global dofs indices are needed.
"""
function get_dofs(feSpace::MultiFESpace, icell::Int)
    map(Base.Fix2(get_dofs, icell), get_fespace(feSpace))
end

function get_shape_functions(feSpace::MultiFESpace, shape::AbstractShape)
    map(Base.Fix2(get_shape_functions, shape), get_fespace(feSpace))
end
function get_cell_shape_functions(feSpace::MultiFESpace, shape::AbstractShape)
    map(Base.Fix2(get_cell_shape_functions, shape), get_fespace(feSpace))
end

""" Low-level constructor """
function _MultiFESpace(
    feSpaces::Tuple{Vararg{TrialOrTest, N}};
    arrayOfStruct::Bool = AOS_DEFAULT,
) where {N}
    # Trick to avoid providing "mesh" as an argument: we read the number
    # of cells in an array of the DofHandler whose size is this number
    _get_ncells_from_fespace = feSpace::TrialOrTest -> size(_get_dhl(feSpace).offset, 1) # TODO : use getters
    ncells = _get_ncells_from_fespace(feSpaces[1])

    # Ensure all SingleFESpace are define on the "same mesh" (checking
    # only the number of cells though...)
    all(feSpace -> _get_ncells_from_fespace(feSpace) == ncells, feSpaces)

    # Build global numbering
    _, mapping = if arrayOfStruct
        _build_mapping_AoS(feSpaces, ncells)
    else
        _build_mapping_SoA(feSpaces, ncells)
    end

    return MultiFESpace{N, typeof(feSpaces)}(feSpaces, mapping, arrayOfStruct)
end

"""
    MultiFESpace(
        feSpaces::Tuple{Vararg{TrialOrTest, N}};
        arrayOfStruct::Bool = AOS_DEFAULT,
    ) where {N}
    MultiFESpace(
        feSpaces::AbstractArray{FE};
        arrayOfStruct::Bool = AOS_DEFAULT,
    ) where {FE <: TrialOrTest}
    MultiFESpace(feSpaces::Vararg{TrialOrTest}; arrayOfStruct::Bool = AOS_DEFAULT)

Build a finite element space representing several sub- finite element spaces.

This is particulary handy when several variables are in play since it provides a global
dof numbering (for the whole system). The finite element spaces composing the
`MultiFESpace` can be different from each other (some continuous, some discontinuous,
some scalar, some vectors...).

# Arguments
- `feSpaces` : the finite element spaces composing the `MultiFESpace`.
    Note that they must be of type `TrialFESpace` or `TestFESpace`.

# Keywords
- `arrayOfStruct::Bool = AOS_DEFAULT` : indicates if the dof numbering should be of type "Array of Structs" (AoS)
    or "Struct of Arrays" (SoA).
"""
function MultiFESpace(
    feSpaces::Tuple{Vararg{TrialOrTest, N}};
    arrayOfStruct::Bool = AOS_DEFAULT,
) where {N}
    _MultiFESpace(feSpaces; arrayOfStruct)
end

function MultiFESpace(
    feSpaces::AbstractArray{FE};
    arrayOfStruct::Bool = AOS_DEFAULT,
) where {FE <: TrialOrTest}
    MultiFESpace((feSpaces...,); arrayOfStruct)
end

function MultiFESpace(feSpaces::Vararg{TrialOrTest}; arrayOfStruct::Bool = AOS_DEFAULT)
    MultiFESpace(feSpaces; arrayOfStruct = arrayOfStruct)
end

"""
Build a global numbering using an Array-Of-Struct strategy
"""
function _build_mapping_AoS(feSpaces::Tuple{Vararg{TrialOrTest}}, ncells::Int)
    # mapping = ntuple(i -> zeros(Int, get_ndofs(_get_dhl(feSpaces[i]))), length(feSpaces))
    # mapping = ntuple(i -> zeros(Int, get_ndofs(feSpaces[i])), N)
    mapping = ntuple(i -> zeros(Int, get_ndofs(feSpaces[i])), length(feSpaces))
    ndofs = 0
    for icell in 1:ncells
        for (ivar, feSpace) in enumerate(feSpaces)
            for d in get_dofs(feSpace, icell)
                if mapping[ivar][d] == 0
                    ndofs += 1
                    mapping[ivar][d] = ndofs
                end
            end
        end
    end
    @assert all(map(x -> all(x .≠ 0), mapping)) "invalid mapping"
    return ndofs, mapping
end

""" Build a global numbering using an Struct-Of-Array strategy """
function _build_mapping_SoA(feSpaces::Tuple{Vararg{TrialOrTest}}, ncells::Int)
    # mapping = ntuple(i -> zeros(Int, get_ndofs(_get_dhl(feSpaces[i]))), length(feSpaces))
    # mapping = ntuple(i -> zeros(Int, get_ndofs(feSpaces[i])), N)
    mapping = ntuple(i -> zeros(Int, get_ndofs(feSpaces[i])), length(feSpaces))
    ndofs = 0
    for (ivar, feSpace) in enumerate(feSpaces)
        for icell in 1:ncells
            for d in get_dofs(feSpace, icell)
                if mapping[ivar][d] == 0
                    ndofs += 1
                    mapping[ivar][d] = ndofs
                end
            end
        end
    end
    @assert all(map(x -> all(x .≠ 0), mapping)) "invalid mapping"
    return ndofs, mapping
end

function build_jacobian_sparsity_pattern(u::TrialFESpace, mesh::AbstractMesh)
    build_jacobian_sparsity_pattern(MultiFESpace(u), mesh)
end
function build_jacobian_sparsity_pattern(u::AbstractMultiFESpace, mesh::AbstractMesh)
    build_jacobian_sparsity_pattern(parent(u), parent(mesh))
end
function build_jacobian_sparsity_pattern(u::MultiFESpace, mesh::Mesh)
    if _is_AoS(u)
        return _build_jacobian_sparsity_pattern_AoS(u, mesh)
    else
        return _build_jacobian_sparsity_pattern_SoA(u, mesh)
    end
end

function _build_jacobian_sparsity_pattern_AoS(u::AbstractMultiFESpace, mesh)
    I = Int[]
    J = Int[]
    f2c = connectivities_indices(mesh, :f2c)
    c2f = connectivities_indices(mesh, :c2f)

    for ic in 1:ncells(mesh)
        for (j, uj) in enumerate(u)
            for (i, ui) in enumerate(u)
                if true #varsDependency[i, j]
                    for jdof in get_mapping(u, j)[get_dofs(uj, ic)]
                        for idof in get_mapping(u, i)[get_dofs(ui, ic)]
                            push!(I, idof)
                            push!(J, jdof)
                        end
                    end
                end
            end
        end

        for ifa in c2f[ic]
            for ic2 in f2c[ifa]
                ic2 == ic && continue
                for (j, uj) in enumerate(u)
                    for (i, ui) in enumerate(u)
                        if true #varsDependency[i, j]
                            for jdof in get_mapping(u, j)[get_dofs(uj, ic2)]
                                for idof in get_mapping(u, i)[get_dofs(ui, ic)]
                                    push!(I, idof)
                                    push!(J, jdof)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    m = n = get_ndofs(u)
    return sparse(I, J, 1.0, m, n, max)
end

function _build_jacobian_sparsity_pattern_SoA(u::MultiFESpace, mesh)
    error("Function `_build_jacobian_sparsity_pattern_SoA` is not implemented yet")
end

allocate_dofs(mfeSpace::MultiFESpace, T = Float64) = zeros(T, get_ndofs(mfeSpace))

# WIP
# """
# Check the `FESpace` `DofHandler` numbering by looking at shared dofs using geometrical criteria.

# Only compatible with Lagrange and Taylor elements for now (no Hermite for instance). For a discontinuous
# variable, simply checks that the dofs are all unique.

# # Example
# ```julia
# mesh = rectangle_mesh(4, 4)
# fes = SingleFESpace(FunctionSpace(:Lagrange, 1), mesh, :continuous)
# @show Bcube.check_numbering(fes, mesh)
# ```
# """
# function check_numbering(space::SingleFESpace, mesh::Mesh; rtol=1e-3, verbose=true, exit_on_error=true)
#     # Track number of errors
#     nerrors = 0

#     # Cell variable infos
#     dhl = _get_dhl(space)
#     fs = get_function_space(space)

#     # For discontinuous, each dof must be unique
#     if is_discontinuous(space)
#         if length(unique(dhl.iglob)) != length(dhl.iglob)
#             nerrors += 1
#             verbose && println("ERROR : two dofs share the same identifier whereas it is a discontinuous variable")
#             exit_on_error && error("DofHandler.check_numbering exited prematurely")
#         end

#         # Exit prematurely
#         return nerrors
#     end

#     # Mesh infos
#     celltypes = cells(mesh)
#     c2n = connectivities_indices(mesh, :c2n)
#     c2c = connectivity_cell2cell_by_nodes(mesh)

#     # Loop over cell
#     for icell in 1:ncells(mesh)
#         # Cell infos
#         ct_i = celltypes[icell]
#         cnodes_i = get_nodes(mesh, c2n[icell])
#         shape_i = shape(ct_i)

#         # Check that all the dofs in this cell are unique
#         iglobs = get_dof(dhl, icell)
#         if length(unique(iglobs)) != length(iglobs)
#             nerrors += 1
#             verbose && println("ERROR : two dofs in the same cell share the same identifier")
#             exit_on_error && error("DofHandler.check_numbering exited prematurely")
#         end

#         # Compute tolerance : cell diagonal divided by 100
#         min_xyz = get_coords(cnodes_i[1])
#         max_xyz = min_xyz
#         for node in cnodes_i
#             max_xyz = max.(max_xyz, get_coords(node))
#             min_xyz = min.(min_xyz, get_coords(node))
#         end
#         atol = norm(max_xyz - min_xyz) * rtol

#         # Coordinates of dofs in cell i for this FunctionSpace
#         coords_i = [mapping(cnodes_i, ct_i, ξ) for ξ in get_coords(fs, shape_i)]

#         # Loop over neighbor cells
#         for jcell in c2c[icell]
#             # Cell infos
#             ct_j = celltypes[jcell]
#             cnodes_j = get_nodes(mesh, c2n[jcell])
#             shape_j = shape(ct_j)

#             # Coordinates of dofs in cell j for this FunctionSpace
#             coords_j = [mapping(cnodes_j, ct_j, ξ) for ξ in get_coords(fs, shape_j)]

#             # n-to-n comparison
#             for (idof_loc, xi) in enumerate(coords_i), (jdof_loc, xj) in enumerate(coords_j)
#                 coincident = norm(xi - xj) < atol

#                 for kcomp in 1:ncomponents(cv)
#                     iglob = get_dof(dhl, icell, kcomp, idof_loc)
#                     jglob = get_dof(dhl, jcell, kcomp, jdof_loc)

#                     msg = ""

#                     # Coordinates are identical but dof numbers are different
#                     if coincident && (iglob != jglob)
#                         msg = "ERROR : two dofs share the same location but have different identifiers"

#                         # Coordinates are different but dof numbers are the same
#                     elseif !coincident && (iglob == jglob)
#                         msg = "ERROR : two dofs share the same number but have different location"
#                     end

#                     # Error encountered?
#                     if length(msg) > 0
#                         nerrors += 1
#                         verbose && println(msg)
#                         verbose && println("icell=$icell, jcell=$jcell, xi=$xi, xj=$xj, iglob=$iglob, jglob=$jglob")
#                         exit_on_error && error("DofHandler.check_numbering exited prematurely")
#                     end

#                 end
#             end
#         end # loop on jcells
#     end # loop on icells

#     return nerrors
# end
