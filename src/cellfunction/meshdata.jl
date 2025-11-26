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

"""
    convert_to_lagrange_P1(mesh::AbstractMesh, data::MeshData{PointData})

Return a Lagrange P1 representation of the `MeshPointData`.
"""
function convert_to_lagrange_P1(mesh::AbstractMesh, data::MeshData{PointData})
    # For now, only "scalar" MeshPointData are supported
    @assert ndims(get_values(data)) == 1 "Only scalar data are supported for now"
    @assert length(get_values(data)) == nnodes(mesh)

    # Build the node -> dof numbering
    # Warning : in the mesh, it can exist some nodes that don't belong to any element
    # (especially coming from ill-constructed input file).
    fs = FunctionSpace(:Lagrange, 1)
    U = TrialFESpace(fs, mesh)
    dhl = _get_dhl(U)
    node2idof = zeros(Int, nnodes(mesh))
    for cellInfo in DomainIterator(CellDomain(mesh))
        cshape = shape(celltype(cellInfo))
        icell = cellindex(cellInfo)
        c2n = get_nodes_index(cellInfo)
        for (ivertex_l, idofs_l) in enumerate(idof_by_vertex(fs, cshape))
            ivertex_g = c2n[ivertex_l]
            idof_g = get_dof(dhl, icell, 1, first(idofs_l)) # there is only one dof per vertex with Lagrange P1
            node2idof[ivertex_g] = idof_g
        end
    end

    # Filter to eliminate nodes belonging to no cell
    ind = findall(x -> x > 0, node2idof)
    perm = invperm(node2idof[ind])
    vals = get_values(data)[ind]

    # Reorder the MeshPointData values to match the dof ordering
    return FEFunction(U, vals[perm])
end

_wrap_value(value) = value
_wrap_value(value::Union{Number, AbstractArray}) = ReferenceFunction(ξ -> value, Val(1))

MeshCellData(values::AbstractVector) = MeshData(CellData(), values)
MeshFaceData(values::AbstractVector) = MeshData(FaceData(), values)
MeshPointData(values::AbstractVector) = MeshData(PointData(), values)
