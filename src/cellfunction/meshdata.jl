abstract type AbstractMeshDataLocation end
struct CellData <: AbstractMeshDataLocation end
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
    return get_values(data)[cellindex(cInfo)]
end

MeshCellData(values::AbstractVector) = MeshData(CellData(), values)
MeshPointData(values::AbstractVector) = MeshData(PointData(), values)
