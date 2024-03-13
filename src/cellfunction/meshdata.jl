abstract type AbstractMeshDataLocation end
struct CellData <: AbstractMeshDataLocation end
struct PointData <: AbstractMeshDataLocation end

"""
Represent a data whose values are known inside each cell (or on each node) of the mesh.

Note that the "values" can be anything : an vector of scalar (conductivity by cell), an array
of functions, etc.

# Example
```julia
n = 10
mesh = line_mesh(n)
data = MeshCellData(rand(n))
data = MeshCellData([x -> i*x for i in 1:n])
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
    PhysicalFunction(x -> _make_callable(get_values(data)[cellindex(cInfo)], x))
end

_make_callable(f, x) = f(x)
_make_callable(a::Union{Number, AbstractArray}, x) = a

MeshCellData(values::AbstractVector) = MeshData(CellData(), values)
MeshPointData(values::AbstractVector) = MeshData(PointData(), values)
