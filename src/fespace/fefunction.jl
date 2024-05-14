"""
    AbstractFEFunction{S}

`S` is the size of the associated `FESpace`

# Interface
Subtypes should implement:
- `get_fespace(f::AbstractFEFunction)`
"""
abstract type AbstractFEFunction{S} <: AbstractLazy end

function show_type_tree(f::AbstractFEFunction; level = 1, indent = 4, prefix = "")
    println(prefix * string(get_name(f)))
end
@inline get_name(::AbstractFEFunction) = "FEFunction"

function get_fespace(f::AbstractFEFunction)
    error("`get_fespace` is not defined for $(typeof(f))")
end
function get_dof_type(f::AbstractFEFunction)
    error("`get_dof_type` is not defined for type $(typeof(f))")
end
function get_dof_values(f::AbstractFEFunction)
    error("`get_dof_values` is not defined for type $(typeof(f))")
end
function get_dof_values(f::AbstractFEFunction, icell)
    error("`get_dof_values` is not defined for type $(typeof(f))")
end
function get_dof_values(f::AbstractFEFunction, icell, n::Val{N}) where {N}
    error("`get_dof_values` is not defined for type $(typeof(f))")
end

function Base.getindex(f::AbstractFEFunction{S}, i::CellInfo) where {S}
    feSpace = get_fespace(f)
    fSpace = get_function_space(feSpace)
    domainStyle = DomainStyle(fSpace)
    cshape = shape(celltype(i))
    λ = shape_functions(fSpace, cshape) # shape functions for one scalar component
    ndofs = get_ndofs(feSpace, cshape) # total number of dofs for this shape (all components included)
    ncomps = get_ncomponents(feSpace)
    q₀ = get_dof_values(f, cellindex(i), Val(ndofs))
    fcell = _interpolate(Val(ncomps), q₀, λ)
    CellFunction(fcell, domainStyle, Val(S))
end

# scalar case:
_reshape_dofs_for_interpolate(::Val{1}, q) = transpose(q)
_reshape_dofs_for_interpolate(::Val{1}, q::SVector) = transpose(q) # remove ambiguity
# vector case:
function _reshape_dofs_for_interpolate(::Val{N}, q::AbstractVector) where {N}
    transpose(reshape(q, :, N))
end
function _reshape_dofs_for_interpolate(::Val{N}, q::SVector{N2}) where {N, N2}
    transpose(reshape(q, Size(Int(N2 / N), N)))
end

function _interpolate(n::Val{N}, q::AbstractVector, λ) where {N}
    x -> _reshape_dofs_for_interpolate(n, q) * λ(x)
end

"""
    materialize(f::AbstractFEFunction, x)

Implement function `materialize` of the `AbstractLazy` interface.
"""
LazyOperators.materialize(f::AbstractFEFunction, i::CellInfo) = f[i]

function LazyOperators.materialize(f::AbstractFEFunction, side::AbstractSide)
    op_side = get_operator(side)
    return materialize(f, op_side(get_args(side)...))
end

""" dev notes : introduced for BcubeParallel """
abstract type AbstractSingleFieldFEFunction{S} <: AbstractFEFunction{S} end
struct SingleFieldFEFunction{S, FE <: AbstractFESpace, V} <:
       AbstractSingleFieldFEFunction{S}
    feSpace::FE
    dofValues::V
end

function SingleFieldFEFunction(feSpace::AbstractFESpace, dofValues)
    size = get_size(feSpace)
    FE = typeof(feSpace)
    V = typeof(dofValues)
    return SingleFieldFEFunction{size, FE, V}(feSpace, dofValues)
end

function FEFunction(feSpace::AbstractFESpace, dofValues)
    SingleFieldFEFunction(feSpace, dofValues)
end

function FEFunction(feSpace::AbstractFESpace, T::Type{<:Number} = Float64)
    dofValues = allocate_dofs(feSpace, T)
    FEFunction(feSpace, dofValues)
end

function FEFunction(feSpace::AbstractFESpace, constant::Number)
    feFunction = FEFunction(feSpace, typeof(constant))
    feFunction.dofValues .= constant
    return feFunction
end

get_fespace(f::SingleFieldFEFunction) = f.feSpace
get_ncomponents(f::SingleFieldFEFunction) = get_ncomponents(get_fespace(f))
get_dof_type(f::SingleFieldFEFunction) = eltype(get_dof_values(f))
get_dof_values(f::SingleFieldFEFunction) = f.dofValues
function get_dof_values(f::SingleFieldFEFunction, icell)
    feSpace = get_fespace(f)
    idofs = get_dofs(feSpace, icell)
    return view(f.dofValues, idofs)
end
function get_dof_values(f::SingleFieldFEFunction, icell, n::Val{N}) where {N}
    feSpace = get_fespace(f)
    idofs = get_dofs(feSpace, icell, n)
    return f.dofValues[idofs]
end

@inline function set_dof_values!(f::SingleFieldFEFunction, values::AbstractArray)
    f.dofValues .= values
end

"""
Represent a Tuple of FEFunction associated to a MultiFESpace
"""
struct MultiFieldFEFunction{
    S,
    FEF <: Tuple{Vararg{AbstractSingleFieldFEFunction}},
    MFE <: AbstractMultiFESpace,
} <: AbstractFEFunction{S}
    feFunctions::FEF
    mfeSpace::MFE

    function MultiFieldFEFunction(f, space)
        new{
            ntuple(i -> get_size(get_fespace(space, i)), get_n_fespace(space)),
            typeof(f),
            typeof(space),
        }(
            f,
            space,
        )
    end
end

@inline get_fe_functions(mfeFunc::MultiFieldFEFunction) = mfeFunc.feFunctions
@inline function get_fe_function(mfeFunc::MultiFieldFEFunction, iSpace::Int)
    mfeFunc.feFunctions[iSpace]
end
@inline _get_mfe_space(mfeFunc::MultiFieldFEFunction) = mfeFunc.mfeSpace

function get_dof_type(mfeFunc::MultiFieldFEFunction)
    mapreduce(get_dof_type, promote, (mfeFunc...,))
end

"""
Update the vector `u` with the values of each `FEFunction` composing this MultiFieldFEFunction.
The mapping of the associated MultiFESpace is respected.
"""
function get_dof_values!(u::AbstractVector{<:Number}, mfeFunc::MultiFieldFEFunction)
    for (iSpace, feFunction) in enumerate(get_fe_functions(mfeFunc))
        u[get_mapping(_get_mfe_space(mfeFunc), iSpace)] .= get_dof_values(feFunction)
    end
end

function get_dof_values(mfeFunc::MultiFieldFEFunction)
    u = mapreduce(get_dof_values, vcat, (mfeFunc...,))
    get_dof_values!(u, mfeFunc)
    return u
end

"""
Constructor for a FEFunction built on a MultiFESpace. All dofs are initialized to zero by default,
but an array `dofValues` can be passed.
"""
function FEFunction(
    mfeSpace::AbstractMultiFESpace,
    dofValues::AbstractVector = allocate_dofs(mfeSpace),
)
    feFunctions = ntuple(
        iSpace -> FEFunction(
            get_fespace(mfeSpace, iSpace),
            view(dofValues, get_mapping(mfeSpace, iSpace)),
        ),
        get_n_fespace(mfeSpace),
    )
    return MultiFieldFEFunction(feFunctions, mfeSpace)
end

"""
Update the dofs of each FEFunction composing the MultiFieldFEFunction
"""
function set_dof_values!(mfeFunc::MultiFieldFEFunction, u::AbstractVector)
    for (iSpace, feFunction) in enumerate(get_fe_functions(mfeFunc))
        set_dof_values!(feFunction, view(u, get_mapping(_get_mfe_space(mfeFunc), iSpace)))
    end
end

Base.iterate(mfeFunc::MultiFieldFEFunction) = iterate(get_fe_functions(mfeFunc))
function Base.iterate(mfeFunc::MultiFieldFEFunction, state)
    iterate(get_fe_functions(mfeFunc), state)
end
