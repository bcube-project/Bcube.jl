module BcubeGPU
# using Cthulhu
using Bcube
using StaticArrays
import KernelAbstractions as KA
using GPUArrays
using Adapt
import AcceleratedKernels as AK
using BenchmarkTools
using Profile
using InteractiveUtils
using CUDA
using Atomix
using GPUArrays
import Bcube:
    AbstractCellDomain,
    AbstractFaceDomain,
    AllFaceDomain,
    CellInfo,
    CellPoint,
    Connectivity,
    DofHandler,
    DomainIterator,
    FaceInfo,
    FacePoint,
    FaceSidePair,
    LazyMapOver,
    Mesh,
    MeshConnectivity,
    NullOperator,
    PhysicalDomain,
    ReferenceDomain,
    Side⁻,
    Side⁺,
    SingleFESpace,
    SingleFieldFEFunction,
    boundary_faces,
    boundary_nodes,
    connectivities,
    entities,
    get_args,
    get_cellinfo_n,
    get_cellinfo_p,
    get_cell_shape_functions,
    get_cell_side_n,
    get_cell_side_p,
    get_coords,
    get_dof,
    get_dirichlet_boundary_tags,
    get_element_index,
    get_element_type,
    get_function_space,
    get_metadata,
    get_quadrature,
    idof_by_face_with_bounds,
    indices,
    integrate_on_ref_element,
    is_continuous,
    nfaces,
    nlayers,
    shape,
    _get_dhl,
    _get_index,
    _scalar_shape_functions

const WORKGROUP_SIZE = 256
const BENCH = true

include("./adapt.jl")

struct MyShapeFunction{FE, I} <: Bcube.AbstractLazy
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

@kernel function AK_assemble_kernel!(y::Y, f::F, V::TV, quadrature, domain) where {Y, F, TV}
    I = @index(Global)
    elementInfo = Bcube._get_index(domain, I)
    vₑ = Bcube.blockmap_shape_functions(V, elementInfo)
    fᵥ = Bcube.materialize(f(vₑ), elementInfo)
    values = Bcube.integrate_on_ref_element(fᵥ, elementInfo, quadrature)
    Bcube._update_b!(y, V, values, elementInfo, domain)
    nothing
end

# function _update_b!(b::B, V, values, elementInfo::CellInfo) where {B}
#     idofs = Bcube.get_dofs(V, Bcube.cellindex(elementInfo))
#     unwrapValues = Bcube._unwrap_cell_integrate(V, values)
#     Bcube._update_b!(b, idofs, unwrapValues)
# end

## TODO : AVOID TYPE PIRACY !!!!
## and avoid atomic on single-threaded CPU cases with a stable dispatch
function Bcube._update_b!(b::AbstractVector, dofs::D, vals::V) where {D, V}#, withAtomic = true)
    withAtomic = true
    if withAtomic
        for (i, val) in zip(dofs, vals)
            Atomix.@atomic b[i] += val
        end
    else
        for (i, val) in zip(dofs, vals)
            b[i] += val
        end
    end
    nothing
end

function Bcube.__assemble_linear!(b::AbstractGPUArray, f, V, measure::Measure)
    AK_assemble!(b, f, nothing, V, measure)
end

function AK_assemble!(y, f, elts, V, measure)
    quadrature = Bcube.get_quadrature(measure)
    domain = get_domain(measure)
    backend = get_backend(y)
    ndrange = length(DomainIterator(domain))
    AK_assemble_kernel!(backend, WORKGROUP_SIZE)(
        y,
        f,
        V,
        quadrature,
        domain;
        ndrange = ndrange,
    )
    KA.synchronize(backend)
    return nothing
end

@kernel function AK_assemble_bilinear_kernel!(
    imat,
    jmat,
    vmat,
    offsets,
    f::F,
    elts::E,
    U::TU,
    V::TV,
    quadrature,
    domain,
) where {F, E, TU, TV}
    I = @index(Global)
    elementInfo = Bcube._get_index(domain, I)
    λu, λv = Bcube.blockmap_bilinear_shape_functions(U, V, elementInfo)
    g1 = Bcube.materialize(f(λu, λv), elementInfo)
    values = Bcube.integrate_on_ref_element(g1, elementInfo, quadrature)
    Bcube._append_contribution!(
        (offsets[I], vmat),
        imat,
        jmat,
        U,
        V,
        values,
        elementInfo,
        domain,
    )
    nothing
end

function Bcube._append_bilinear!(I, J, _X::Tuple, row, col, vals)
    offset, X = _X
    _rows, _cols = Bcube._cartesian_product(row, col)
    for k in eachindex(_rows)
        I[offset + k] = _rows[k]
        J[offset + k] = _cols[k]
        X[offset + k] = vals[k]
    end
end

@kernel function _ndofs_element_bilinear_kernel!(ndofs, U, V, domain::D) where {D}
    I = @index(Global)
    elementInfo = @inline Bcube._get_index(domain, I)
    nU = Val(Bcube.get_ndofs(U, shape(Bcube.celltype(elementInfo))))
    nV = Val(Bcube.get_ndofs(V, shape(Bcube.celltype(elementInfo))))
    Udofs = Bcube.get_dofs(U, I, nU) # columns correspond to the TrialFunction
    Vdofs = Bcube.get_dofs(V, I, nV) # lines correspond to the TestFunction
    rows, = Bcube._cartesian_product(Vdofs, Udofs)
    ndofs[I] = length(rows)
end

function Bcube.assemble_bilinear!(I, J, X::AbstractGPUArray, f, measure::Measure, U, V)
    AK_assemble_bilinear(I, J, X, f, nothing, U, V, measure)
end

function Bcube.allocate_bilinear(backend::CUDABackend, a, U, V, T)
    integration = a(Bcube._null_operator(U), Bcube._null_operator(V))
    domain = Bcube.get_domain(Bcube.get_measure(integration))
    ndofs = KernelAbstractions.zeros(backend, Int, length(Bcube.indices(domain)))
    _ndofs_element_bilinear_kernel!(backend, WORKGROUP_SIZE)(
        ndofs,
        U,
        V,
        domain;
        ndrange = size(ndofs),
    )
    buffersize = AK.reduce(+, ndofs; init = zero(eltype(ndofs)))
    I = KernelAbstractions.zeros(backend, Int, buffersize)
    J = KernelAbstractions.zeros(backend, Int, buffersize)
    X = KernelAbstractions.zeros(backend, Float64, buffersize)
    return I, J, X
end

function AK_assemble_bilinear(I, J, X, f, elts::E, U, V, measure) where {E}
    quadrature = Bcube.get_quadrature(measure)
    domain = get_domain(measure)
    backend = get_backend(X)

    ndofs = KernelAbstractions.zeros(backend, Int, length(Bcube.indices(domain)))
    _ndofs_element_bilinear_kernel!(backend, WORKGROUP_SIZE)(
        ndofs,
        U,
        V,
        domain;
        ndrange = size(ndofs),
    )
    offsets = AK.accumulate(+, ndofs; init = zero(eltype(ndofs)), inclusive = false)
    ndrange = length(DomainIterator(domain))
    AK_assemble_bilinear_kernel!(backend, WORKGROUP_SIZE)(
        I,
        J,
        X,
        offsets,
        f,
        elts,
        U,
        V,
        quadrature,
        domain;
        ndrange = ndrange,
    )
    KA.synchronize(backend)
    return nothing
end

f(λ) = PhysicalFunction(x -> 1.0) * λ #

solve(backend, nx, ny) = _solve(backend, nx, ny)

function _solve(backend, nx = 300, ny = 300)
    mesh_cpu = rectangle_mesh(nx, ny)
    mesh = adapt(backend, mesh_cpu)
    test_arg(backend, mesh)
    println("mesh on GPU!")

    Ω_cpu = CellDomain(mesh_cpu)
    Ω = CellDomain(mesh)
    degree = 1
    degquad = 2 * degree + 1

    Γ_cpu = InteriorFaceDomain(mesh_cpu)
    Γ = InteriorFaceDomain(mesh)
    # test_arg(backend, Γ)
    println("Γ on GPU!")

    dΓ = Measure(Γ, degquad)
    dΩ = Measure(Ω, degquad)
    # test_arg(backend, dΓ)
    println("dΩ on GPU!")

    # Build TrialFESpace and TestFESpace
    # The TrialFESpace must be first built on the CPU for now because the
    # underlying DofHandler constructor uses scalar indexing
    g(x, t) = 3x[1]
    h(x, t) = 5x[1] + 2

    U_cpu = TrialFESpace(
        FunctionSpace(:Lagrange, degree),
        mesh_cpu,
        :discontinuous,
        Dict("xmin" => 5.0, "xmax" => g, "ymin" => h),
    )
    U = adapt(backend, U_cpu)
    test_arg(backend, U)
    println("U on GPU!")

    V = TestFESpace(U)
    test_arg(backend, V)
    println("V on GPU!")

    # Build FEFunction
    u = FEFunction(U, KernelAbstractions.ones(backend, Float64, get_ndofs(U)))
    test_arg(backend, u)
    println("u on GPU!")

    _cells_cpu = collect(Bcube.DomainIterator(Ω_cpu))
    cells_cpu = typeof(first(_cells_cpu))[_cells_cpu...]
    cells = adapt(backend, cells_cpu)

    _faces_cpu = collect(Bcube.DomainIterator(Γ_cpu))
    faces_cpu = typeof(first(_faces_cpu))[_faces_cpu...]
    faces = adapt(backend, faces_cpu)

    # Define linear form and assemble
    f(φ) = u * φ
    f_face(φ) = side_n(u) * jump(φ)
    y = KernelAbstractions.zeros(backend, Float64, get_ndofs(U))
    AK_assemble!(y, f, cells, V, dΩ)

    y_face = KernelAbstractions.zeros(backend, Float64, get_ndofs(U))
    AK_assemble!(y_face, f_face, faces, V, dΓ)
    # AK_assemble_linear_elemental!(y, f, cells, Iloc, V, dΩ)
    # synchronize(backend)
    y .= 0
    if backend isa typeof(CPU())
        @time AK_assemble!(y, f, cells, V, dΩ)
        # @btime AK_assemble!($y, $f, $cells, $V, $dΩ, $rdhl)
        # Profile.init(; n = 10^7) # returns the current settings
        # Profile.clear()
        # Profile.clear_malloc_data()
        # @profile begin
        #     for i in 1:100
        #         AK_assemble!(y, f, cells, V, dΩ, rdhl)
        #     end
        # end
    else
        if BENCH # time CELLDOMAIN
            y .= 0
            CUDA.@time CUDA.@sync AK_assemble!(y, f, cells, V, dΩ)
            y .= 0
            @btime CUDA.@sync begin
                AK_assemble!($y, $f, $cells, $V, $dΩ)
            end
            y .= 0
            AK_assemble!(y, f, cells, V, dΩ)
        end
        if BENCH # time INTERIORFACEDOMAIN
            println("--face")
            y_face .= 0
            @btime CUDA.@sync begin
                AK_assemble!($y_face, $f_face, $faces, $V, $dΓ)
            end
        end
        CUDA.@profile AK_assemble!(y_face, f_face, faces, V, dΓ)
    end

    # ref solution :
    u_cpu = FEFunction(U_cpu, ones(get_ndofs(U)))
    dΩ_cpu = Measure(CellDomain(mesh_cpu), degquad)
    dΓ_cpu = Measure(InteriorFaceDomain(mesh_cpu), degquad)
    f_cpu(φ) = ∫(u_cpu * φ) * dΩ_cpu
    f_face_cpu(φ) = ∫(side_n(u_cpu) * jump(φ)) * dΓ_cpu
    y_ref = assemble_linear(f_cpu, TestFESpace(U_cpu))
    y_ref .= 0.0
    println("--- REF timing ----")
    assemble_linear!(y_ref, f_cpu, TestFESpace(U_cpu))
    if BENCH
        @time assemble_linear!(y_ref, f_cpu, TestFESpace(U_cpu))
        y_ref .= 0.0
        @btime assemble_linear!($y_ref, $f_cpu, $(TestFESpace(U_cpu)))
        println("--face--")
        y_ref .= 0.0
        @btime assemble_linear!($y_ref, $f_face_cpu, $(TestFESpace(U_cpu)))
    end

    y_ref .= 0.0
    assemble_linear!(y_ref, f_cpu, TestFESpace(U_cpu))
    # @show Array(y)[1:20]
    #  @show y_ref[1:20]
    @assert all(Array(y) .≈ y_ref)
    println("assert all(Array(y) .≈ y_ref) --> OK")

    # display(y)
end
function run(backend, nx, ny)
    solve(backend, nx, ny)
end

@kernel function test_arg_kernel(x, @Const(arg))
    I = @index(Global)
    x[I] += 1
end

function test_arg(backend, arg)
    x = KernelAbstractions.zeros(backend, Float32, 10)
    test_arg_kernel(backend, WORKGROUP_SIZE)(x, arg; ndrange = size(x))
end

end
