"""
    AbstractConnectivity{T}

Supertype for connectivity with indices of type `T`.
"""
abstract type AbstractConnectivity{T} end

Base.length(::AbstractConnectivity) = error("Undefined")
Base.size(::AbstractConnectivity) = error("Undefined")

"""
    Connectivity{T}

Type for connectivity with elements of type `T`.
"""
struct Connectivity{T <: Integer, O, I} <: AbstractConnectivity{T}
    minsize::T
    maxsize::T
    offsets::O
    indices::I
end

function Connectivity(
    numIndices::AbstractVector{T},
    indices::AbstractVector{T},
) where {T <: Integer}
    ne = length(numIndices)
    minsize = minimum(numIndices)
    maxsize = maximum(numIndices)
    offsets = zeros(T, ne + 1)
    offsets[1] = 1
    for (i, size) in enumerate(numIndices)
        offsets[i + 1] = offsets[i] + size
    end
    _check_connectivity(minsize, maxsize, offsets, indices)
    return Connectivity{T, typeof(offsets), typeof(indices)}(
        minsize,
        maxsize,
        offsets,
        indices,
    )
end

function _check_connectivity(minsize, maxsize, offsets, indices)
    if offsets[end] - 1 ≠ length(indices)
        @show offsets[end] - 1, length(indices)
        error("Invalid offset range")
    end
    for i in firstindex(offsets):(lastindex(offsets) - 1)
        offsets[i + 1] < offsets[i] ? error("Invalid offset ", i) : nothing
    end
end

Base.length(c::Connectivity) = length(c.offsets) - 1
Base.size(c::Connectivity) = (length(c),)
Base.axes(c::Connectivity) = 1:length(c)

@propagate_inbounds Base.firstindex(c::Connectivity, i) = c.offsets[i]
@propagate_inbounds Base.lastindex(c::Connectivity, i) = c.offsets[i + 1] - 1
@propagate_inbounds Base.length(c::Connectivity, i) = lastindex(c, i) - firstindex(c, i) + 1
Base.size(c::Connectivity, i) = (length(c, i),)
@propagate_inbounds function Base.axes(c::Connectivity, i)
    if i > 0
        return firstindex(c, i):1:lastindex(c, i)
    else
        return lastindex(c, -i):-1:firstindex(c, -i)
    end
end

@inline Base.getindex(c::Connectivity) = c.indices
@propagate_inbounds Base.getindex(c::Connectivity, i) = view(c.indices, axes(c, i))
function Base.getindex(c::Connectivity, i, ::Val{N}) where {N}
    @assert length(c, i) == N "invalid length (length(c,i)!=N)"
    SVector{N}(c[i])
end

@inline minsize(c::Connectivity) = c.minsize
@inline maxsize(c::Connectivity) = c.maxsize

#@inline minconnectivity(c::Connectivity) = c.minConnectivity
#@inline maxconnectivity(c::Connectivity) = c.maxConnectivity
#@inline extrema(c::Connectivity) = (c.minElementInSet,c.maxElementInSet)

Base.eltype(::Connectivity{T}) where {T <: Integer} = T
function Base.iterate(c::Connectivity{T}, i = 1) where {T <: Integer}
    if i > length(c)
        return nothing
    else
        return c[i], i + 1
    end
end

function Base.show(io::IO, c::Connectivity)
    println(typeof(c))
    for (i, _c) in enumerate(c)
        println(io, i, "=>", _c)
    end
end

#Base.keys(c::Connectivity) = LinearIndices(1:length(c))

"""
    inverse_connectivity(c::Connectivity{T}) where {T}

Returns the "inverse" of the connectivity 'c' and the corresponding 'keys'.
'keys' are provided because indices in 'c' could be sparse in the general case.

# Example
```julia
mesh = basic_mesh()
c2n = connectivities_indices(mesh,:c2n)
n2c, keys = inverse_connectivity(c2n)
```
Here, 'n2c' is the node->cell graph of connectivity and,
'n2c[i]' contains the indices of the cells connected to the node of index 'keys[i]'.
If 'c2n' is dense, 'keys' is not necessary (because keys[i]==i, ∀i)
"""
function inverse_connectivity(c::Connectivity{T}) where {T}
    dict = Dict{T, Vector{T}}()
    for (i, cᵢ) in enumerate(c)
        for k in cᵢ
            get(dict, k, nothing) === nothing ? dict[k] = [i] : push!(dict[k], i)
        end
    end
    _keys = collect(keys(dict))
    _vals = collect(values(dict))
    numindices = length.(_vals)
    indices = rawcat(_vals)
    Connectivity(numindices, indices), _keys
end
