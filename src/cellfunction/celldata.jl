abstract type AbstractCellData{T} <: AbstractVector{T} end
IndexableOperatorStyle(::Type{<:AbstractCellData}) = IsCellIndexableOperatorStyle()
CallableOperatorStyle(::Type{<:AbstractCellData}) = IsNotCallableOperatorStyle()

struct CellData{T} <: AbstractCellData{T}
    v::Vector{T}
end

CellData(v::AbstractVector{T}) where {T} = CellData{T}(v)

@inline Base.parent(c::CellData) = c.v
Base.size(c::CellData) = size(parent(c))
Base.getindex(c::CellData, i::Int) = parent(c)[i]
Base.getindex(c::CellData, i::CellInfo) = c[cellindex(i)]

Base.setindex!(c::CellData, x, I...) = parent(c)[I...] = x

get_values(c::CellData) = parent(c)
set_values!(c::CellData, x) = c.v .= x

Base.:*(a::CellData, b::CellData) = CellData(get_values(a) .* get_values(b))
