abstract type AbstractMeshDataLocation end
struct CellData <: AbstractMeshDataLocation end
struct PointData <: AbstractMeshDataLocation end

"""
Represent a data whose values are known at each cell-center (or at each node) of the mesh
"""
struct MeshData{L <: AbstractMeshDataLocation, T <: AbstractVector} <: AbstractLazy
    values::T
end
function MeshData(location::AbstractMeshDataLocation, values::AbstractVector)
    MeshData{typeof(location), typeof(values)}(values)
end
get_values(data::MeshData) = data.values
set_values!(data::MeshData, values::Union{Number, AbstractVector}) = data.values .= values
get_location(data::MeshData{L}) where {L} = L()

function LazyOperators.materialize(data::MeshData, cInfo::CellInfo)
    PhysicalFunction(x -> get_values(data)[cellindex(cInfo)])
end

MeshCellData(values::AbstractVector) = MeshData(CellData(), values)
MeshPointData(values::AbstractVector) = MeshData(PointData(), values)
