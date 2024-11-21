abstract type AbstractMeshDataLocation end
struct CellData <: AbstractMeshDataLocation end
struct FaceData <: AbstractMeshDataLocation end
struct PointData <: AbstractMeshDataLocation end

"""
Represent a data whose values are known inside each cell/node/face of the mesh.

Note that the "values" can be anything : an vector of scalar (conductivity by cell), an array
of functions, etc.

# Example
```julia
n = 10
mesh = line_mesh(n)
data = MeshCellData(rand(n))
data = MeshCellData([PhysicalFunction(x -> i*x) for i in 1:n])
```
"""
struct MeshData{L <: AbstractMeshDataLocation, T <: AbstractVector} <: AbstractLazy
    values::T
end
function MeshData(location::AbstractMeshDataLocation, values::AbstractVector)
    MeshData{typeof(location), typeof(values)}(values)
end
get_values(data::MeshData) = data.values
set_values!(data::MeshData, values::Union{Number, AbstractVector}) = data.values .= values
get_location(::MeshData{L}) where {L} = L()

function LazyOperators.materialize(data::MeshData{CellData}, cInfo::CellInfo)
    value = get_values(data)[cellindex(cInfo)]
    return _wrap_value(value)
end

function LazyOperators.materialize(
    data::MeshData{CellData},
    side::Side⁻{Nothing, <:Tuple{<:FaceInfo}},
)
    fInfo = get_args(side)[1]
    cInfo_n = get_cellinfo_n(fInfo)
    return materialize(data, cInfo_n)
end

function LazyOperators.materialize(
    data::MeshData{CellData},
    side::Side⁺{Nothing, <:Tuple{<:FaceInfo}},
)
    fInfo = get_args(side)[1]
    cInfo_p = get_cellinfo_p(fInfo)
    return materialize(data, cInfo_p)
end

function LazyOperators.materialize(
    data::MeshData{FaceData},
    side::AbstractSide{Nothing, <:Tuple{<:FaceInfo}},
)
    fInfo = get_args(side)[1]
    value = get_values(data)[faceindex(fInfo)]
    return _wrap_value(value)
end

_wrap_value(value) = value
_wrap_value(value::Union{Number, AbstractArray}) = ReferenceFunction(ξ -> value, Val(1))

MeshCellData(values::AbstractVector) = MeshData(CellData(), values)
MeshFaceData(values::AbstractVector) = MeshData(FaceData(), values)
MeshPointData(values::AbstractVector) = MeshData(PointData(), values)
