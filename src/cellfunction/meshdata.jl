abstract type AbstractMeshDataLocation end
struct CellData <: AbstractMeshDataLocation end
struct FaceData <: AbstractMeshDataLocation end
struct PointData <: AbstractMeshDataLocation end

"""
    MeshData{L <: AbstractMeshDataLocation, T <: AbstractVector} <: AbstractLazy

Represent a data whose values are known inside each cell/node/face of the mesh.

Note that the "values" can be anything : an vector of scalar (conductivity by cell), an array
of functions, etc.

# Example
```julia
n = 10
mesh = line_mesh(10)
cell_data = MeshCellData(rand(ncells(mesh)))
cell_data = MeshCellData([PhysicalFunction(x -> i*x) for i in 1:ncells(mesh)])
node_data = MeshPointData(rand(nnodes(mesh)))
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

# Examples
Scalar field
```julia
nx = ny = 10
mesh = rectangle_mesh(nx, ny)
node_values = MeshPointData([i+j for i in 1:nx for j in 1:ny])
u_sca = Bcube.convert_to_lagrange_P1(mesh, node_values)
```

Vector field
```julia
nx = ny = 10
mesh = rectangle_mesh(nx, ny)
node_values = MeshPointData([[i,j] for i in 1:nx for j in 1:ny])
u_vec = Bcube.convert_to_lagrange_P1(mesh, node_values)
```
"""
function convert_to_lagrange_P1(mesh::AbstractMesh, data::MeshData{PointData})
    vals = get_values(data)

    @assert length(vals) == nnodes(mesh)

    # Get the size of the FESpace
    ncomps = length(first(vals))

    # Create the FEFunction
    fs = FunctionSpace(:Lagrange, 1)
    U = TrialFESpace(fs, mesh; size = ncomps)
    dhl = get_dhl(U)
    u = FEFunction(U)
    dofValues = get_dof_values(u)

    # Loop over the mesh cells. In each cell, loop over the vertices and the dofs (same number)
    # and set the dof values.
    for cinfo in DomainIterator(CellDomain(mesh))
        cshape = shape(celltype(cinfo))
        icell = cellindex(cinfo)
        c2n = get_nodes_index(cinfo)
        for (ivertex_g, idofs_l) in zip(c2n, idof_by_vertex(fs, cshape))
            idof_l = first(idofs_l) # there is only one dof per vertex with Lagrange P1
            idofs_g = map(icomp -> get_dof(dhl, icell, icomp, idof_l), 1:ncomps)
            dofValues[idofs_g] .= vals[ivertex_g]
        end
    end

    return u
end

_wrap_value(value) = value
_wrap_value(value::Union{Number, AbstractArray}) = ReferenceFunction(ξ -> value, Val(1))

MeshCellData(values::AbstractVector) = MeshData(CellData(), values)
MeshFaceData(values::AbstractVector) = MeshData(FaceData(), values)
MeshPointData(values::AbstractVector) = MeshData(PointData(), values)
