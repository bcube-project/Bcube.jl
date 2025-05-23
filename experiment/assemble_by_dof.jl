module tmp
using Bcube
using KernelAbstractions

struct MyShapeFunction{FE, I} <: Bcube.AbstractLazy where {FE, I}
    feSpace::FE
    iloc::I
end

Bcube.materialize(f::MyShapeFunction, ::Bcube.CellInfo) = f

function Bcube.materialize(f::MyShapeFunction, cPoint::Bcube.CellPoint)
    cInfo = Bcube.get_cellinfo(cPoint)
    cType = Bcube.get_element_type(cInfo)
    cShape = Bcube.shape(cType)
    fs = Bcube.get_function_space(f.feSpace)
    ξ = Bcube.get_coords(cPoint)
    return Bcube._scalar_shape_functions(fs, cShape, ξ)[f.iloc]
end

"""
Build the connectivity <global dof index> -> <cells surrounding this dof, local index of this dof
in the cells>
"""
function build_dof_to_cells(mesh, U)
    dof_to_cells = [Tuple{Int, Int}[] for _ in 1:get_ndofs(U)]
    dhl = Bcube._get_dhl(U)
    # dof_to_cells = [Int[] for _ in 1:get_ndofs(U)]
    for icell in 1:ncells(mesh)
        # foreach(idof -> push!(dof_to_cells[idof], icell), Bcube.get_dofs(U, icell))
        for iloc in 1:get_ndofs(dhl, icell)
            idof = Bcube.get_dof(dhl, icell, 1, iloc) # comp = 1
            push!(dof_to_cells[idof], (icell, iloc))
        end
    end
    return dof_to_cells
end

@kernel function gpu_assemble_kernel(
    b,
    @Const(f),
    @Const(V),
    @Const(measure),
    @Const(dof_to_cells)
)
    I = @index(Global)

    # Alias
    domain = Bcube.get_domain(measure)
    quadrature = Bcube.get_quadrature(measure)
    mesh = Bcube.get_mesh(domain)

    # Loop over cells surrounding this dof
    for (icell, iloc) in dof_to_cells[I]
        cellInfo = Bcube._get_cellinfo(mesh, icell)
        φ = MyShapeFunction(V, iloc)
        fᵥ = Bcube.materialize(f(φ), cellInfo)
        value = Bcube.integrate_on_ref_element(fᵥ, cellInfo, quadrature)
        b[I] += value
    end
end

function gpu_assemble(backend, b, f, V, measure, dof_to_cells)
    gpu_assemble_kernel(backend, 64)(b, f, V, measure, dof_to_cells; ndrange = size(b))
    synchronize(backend)
end

function run()
    mesh = rectangle_mesh(3, 4)
    dΩ = Measure(CellDomain(mesh), 1)

    fs = FunctionSpace(:Lagrange, 1)
    U = TrialFESpace(fs, mesh)
    V = TestFESpace(U)
    u = FEFunction(U, (1.0 * 1):get_ndofs(U))

    dof_to_cells = build_dof_to_cells(mesh, U)

    b = zeros(get_ndofs(U))
    f(φ) = u * φ

    backend = CPU()
    gpu_assemble(backend, b, f, V, dΩ, dof_to_cells)
    @show b

    @show assemble_linear(φ -> ∫(f(φ))dΩ, V)
end

run()

end