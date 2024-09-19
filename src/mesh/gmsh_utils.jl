import gmsh_jll
include(gmsh_jll.gmsh_api)
import .gmsh

# Constants
const GMSHTYPE = Dict(
    1 => Bar2_t(),
    2 => Tri3_t(),
    3 => Quad4_t(),
    4 => Tetra4_t(),
    5 => Hexa8_t(),
    6 => Penta6_t(),
    7 => Pyra5_t(),
    9 => Tri6_t(),
    10 => Quad9_t(),
    21 => Tri9_t(),
    21 => Tri10_t(),
    36 => Quad16_t(),
)

"""
    read_msh(path::String, spaceDim::Int = 0; verbose::Bool = false)

Read a .msh file designated by its `path`.

See `read_msh()` for more details.
"""
function read_msh(path::String, spaceDim::Int = 0; verbose::Bool = false)
    isfile(path) ? nothing : error("File does not exist ", path)

    # Read file using gmsh lib
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", Int(verbose))
    gmsh.open(path)

    # build mesh
    mesh = _read_msh(spaceDim, verbose)

    # free gmsh
    gmsh.finalize()

    return mesh
end

"""
    _read_msh(spaceDim::Int, verbose::Bool)

To use this function, the `gmsh` file must have been opened already (see `read_msh(path::String)`
for instance).

The number of topological dimensions is given by the highest dimension found in the file. The
number of space dimensions is deduced from the axis dimensions if `spaceDim = 0`.
If `spaceDim` is set to a positive number, this number is used as the number of space dimensions.

# Implementation
Global use of `gmsh` module. Do not try to improve this function by passing an argument
such as `gmsh` or `gmsh.model` : it leads to problems.
"""
function _read_msh(spaceDim::Int, verbose::Bool)
    # Spatial dimension of the mesh
    dim = gmsh.model.getDimension()

    # Read nodes
    ids, xyz = gmsh.model.mesh.getNodes()

    # Create a node number remapping to ensure a dense numbering
    absolute_node_indices = [convert(Int, i) for i in ids]
    _, glo2loc_node_indices = densify(absolute_node_indices; permute_back = true)

    # Build nodes coordinates
    xyz = reshape(xyz, 3, :)
    _spaceDim = spaceDim > 0 ? spaceDim : _compute_space_dim(verbose)
    nodes = [Node(xyz[1:_spaceDim, i]) for i in axes(xyz, 2)]

    # Read cells
    elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(dim)

    # Create a cell number remapping to ensure a dense numbering
    absolute_cell_indices = Int.(reduce(vcat, elementTags))
    _, glo2loc_cell_indices = densify(absolute_cell_indices; permute_back = true)

    # Read boundary conditions
    bc_tags = gmsh.model.getPhysicalGroups(-1)
    bc_names = [gmsh.model.getPhysicalName(_dim, _tag) for (_dim, _tag) in bc_tags]
    # keep only physical groups of dimension "dim-1" with none-empty names.
    # bc is a vector of (tag,name) for all valid boundary conditions
    bc = [
        (_tag, _name) for
        ((_dim, _tag), _name) in zip(bc_tags, bc_names) if _dim == dim - 1 && _name ≠ ""
    ]

    bc_names = Dict(convert(Int, _tag) => _name for (_tag, _name) in bc)
    bc_nodes = Dict(
        convert(Int, _tag) => Int[
            glo2loc_node_indices[i] for
            i in gmsh.model.mesh.getNodesForPhysicalGroup(dim - 1, _tag)[1]
        ] for (_tag, _name) in bc
    )

    # Fill type of each cell
    celltypes = [
        GMSHTYPE[k] for (i, k) in enumerate(elementTypes) for t in 1:length(elementTags[i])
    ]

    # Build cell->node connectivity (with Gmsh internal numbering convention)
    c2n_gmsh = Connectivity(
        Int[nnodes(k) for k in reduce(vcat, celltypes)],
        Int[glo2loc_node_indices[k] for k in reduce(vcat, nodeTags)],
    )

    # Convert to CGNS numbering
    c2n = _c2n_gmsh2cgns(celltypes, c2n_gmsh)

    mesh = Mesh(nodes, celltypes, c2n; bc_names = bc_names, bc_nodes = bc_nodes)
    add_absolute_indices!(mesh, :node, absolute_node_indices)
    add_absolute_indices!(mesh, :cell, absolute_cell_indices)
    return mesh
end

"""
    read_msh_with_cell_names(path::String, spaceDim = 0; verbose = false)

Read a .msh file designated by its `path` and also return names and tags
"""
function read_msh_with_cell_names(path::String, spaceDim = 0; verbose = false)
    isfile(path) ? nothing : error("File does not exist ", path)

    # Read file using gmsh lib
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", Int(verbose))
    gmsh.open(path)

    # build mesh
    _spaceDim = spaceDim > 0 ? spaceDim : _compute_space_dim(verbose)
    mesh = _read_msh(_spaceDim, verbose)

    # Read volumic physical groups (build a dict tag -> name)
    el_tags = gmsh.model.getPhysicalGroups(_spaceDim)
    _el_names = [gmsh.model.getPhysicalName(_dim, _tag) for (_dim, _tag) in el_tags]
    el = [
        (_tag, _name) for ((_dim, _tag), _name) in zip(el_tags, _el_names) if
        _dim == _spaceDim && _name ≠ ""
    ]
    el_names = Dict(convert(Int, _tag) => _name for (_tag, _name) in el)
    el_names_inv = Dict(_name => convert(Int, _tag) for (_tag, _name) in el)

    # Read cell indices associated to each volumic physical group
    el_cells = Dict{Int, Array{Int}}()
    for (_dim, _tag) in el_tags
        v = Int[]

        for iEntity in gmsh.model.getEntitiesForPhysicalGroup(_dim, _tag)
            tmpTypes, tmpTags, tmpNodeTags = gmsh.model.mesh.getElements(_dim, iEntity)

            # Notes : a PhysicalGroup "entity" can contain different types of elements.
            # So `tmpTags` is an array of the cell indices of each type in the Physical group.
            for _tmpTags in tmpTags
                v = vcat(v, Int.(_tmpTags))
            end
        end
        el_cells[_tag] = v
    end

    absolute_cell_indices = absolute_indices(mesh, :cell)
    _, glo2loc_cell_indices = densify(absolute_cell_indices; permute_back = true)

    # free gmsh
    gmsh.finalize()

    return mesh, el_names, el_names_inv, el_cells, glo2loc_cell_indices
end

"""
Deduce the number of space dimensions from the mesh : if one (or more) dimension of the bounding
box is way lower than the other dimensions, the number of space dimension is decreased.

Currently, having for instance (x,z) is not supported. Only (x), or (x,y), or (x,y,z).
"""
function _compute_space_dim(verbose::Bool)
    tol = 1e-15

    topodim = gmsh.model.getDimension()

    # Bounding box
    box = gmsh.model.getBoundingBox(-1, -1)
    lx = box[4] - box[1]
    ly = box[5] - box[2]
    lz = box[6] - box[3]

    return _compute_space_dim(topodim, lx, ly, lz, tol, verbose)
end

function _apply_gmsh_options(;
    split_files = false,
    create_ghosts = false,
    msh_format = 0,
    verbose = false,
)
    gmsh.option.setNumber("General.Terminal", Int(verbose))
    gmsh.option.setNumber("Mesh.PartitionSplitMeshFiles", Int(split_files))
    gmsh.option.setNumber("Mesh.PartitionCreateGhostCells", Int(create_ghosts))
    (msh_format > 0) && gmsh.option.setNumber("Mesh.MshFileVersion", msh_format)
end

"""
    gen_line_mesh(
        output;
        nx = 2,
        lx = 1.0,
        xc = 0.0,
        order = 1,
        bnd_names = ("LEFT", "RIGHT"),
        n_partitions = 0,
        kwargs...
    )

Generate a 1D mesh of a segment and write to "output".

Available kwargs are
* `verbose` : `true` or `false` to enable gmsh verbose
* `msh_format` : floating number indicating the output msh format (for instance : `2.2`)
* `split_files` : if `true`, create one file by partition
* `create_ghosts` : if `true`, add a layer of ghost cells at every partition boundary
"""
function gen_line_mesh(
    output;
    nx = 2,
    lx = 1.0,
    xc = 0.0,
    order = 1,
    bnd_names = ("LEFT", "RIGHT"),
    n_partitions = 0,
    kwargs...,
)
    gmsh.initialize()
    _apply_gmsh_options(; kwargs...)
    lc = 1e-1

    # Points
    A = gmsh.model.geo.addPoint(xc - lx / 2, 0, 0, lc)
    B = gmsh.model.geo.addPoint(xc + lx / 2, 0, 0, lc)

    # Line
    AB = gmsh.model.geo.addLine(A, B)

    # Mesh settings
    gmsh.model.geo.mesh.setTransfiniteCurve(AB, nx)

    # Define boundaries (`0` stands for 0D, i.e nodes)
    # ("-1" to create a new tag)
    gmsh.model.addPhysicalGroup(0, [A], -1, bnd_names[1])
    gmsh.model.addPhysicalGroup(0, [B], -1, bnd_names[2])
    gmsh.model.addPhysicalGroup(1, [AB], -1, "Domain")

    # Gen mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(order) # Mesh order
    gmsh.model.mesh.partition(n_partitions)

    # Write result
    gmsh.write(output)

    # End
    gmsh.finalize()
end

"""
    gen_rectangle_mesh(
        output,
        type;
        transfinite = false,
        nx = 2,
        ny = 2,
        lx = 1.0,
        ly = 1.0,
        xc = -1.0,
        yc = -1.0,
        order = 1,
        bnd_names = ("North", "South", "East", "West"),
        n_partitions = 0,
        write_geo = false,
        transfinite_lines = true,
        lc = 1e-1,
        kwargs...
    )

Generate a 2D mesh of a rectangle domain and write the mesh to `output`. Use `type` to specify the element types:
`:tri` or `:quad`.

For kwargs, see [`gen_line_mesh`](@ref).
"""
function gen_rectangle_mesh(
    output,
    type;
    transfinite = false,
    nx = 2,
    ny = 2,
    lx = 1.0,
    ly = 1.0,
    xc = -1.0,
    yc = -1.0,
    order = 1,
    bnd_names = ("North", "South", "East", "West"),
    n_partitions = 0,
    write_geo = false,
    transfinite_lines = true,
    lc = 1e-1,
    kwargs...,
)
    #         North
    #      D ------- C
    #      |         |
    # West |         | East
    #      |         |
    #      A ------- B
    #         South
    gmsh.initialize()
    _apply_gmsh_options(; kwargs...)

    # Points
    A = gmsh.model.geo.addPoint(xc - lx / 2, yc - ly / 2, 0, lc)
    B = gmsh.model.geo.addPoint(xc + lx / 2, yc - ly / 2, 0, lc)
    C = gmsh.model.geo.addPoint(xc + lx / 2, yc + ly / 2, 0, lc)
    D = gmsh.model.geo.addPoint(xc - lx / 2, yc + ly / 2, 0, lc)

    # Lines
    AB = gmsh.model.geo.addLine(A, B)
    BC = gmsh.model.geo.addLine(B, C)
    CD = gmsh.model.geo.addLine(C, D)
    DA = gmsh.model.geo.addLine(D, A)

    # Contour
    ABCD = gmsh.model.geo.addCurveLoop([AB, BC, CD, DA])

    # Surface
    surf = gmsh.model.geo.addPlaneSurface([ABCD])

    # Mesh settings
    if transfinite_lines
        gmsh.model.geo.mesh.setTransfiniteCurve(AB, nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(BC, ny)
        gmsh.model.geo.mesh.setTransfiniteCurve(CD, nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(DA, ny)
    end

    (transfinite || type == :quad) && gmsh.model.geo.mesh.setTransfiniteSurface(surf)

    (type == :quad) && gmsh.model.geo.mesh.setRecombine(2, surf)

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define boundaries (`1` stands for 1D, i.e lines)
    south = gmsh.model.addPhysicalGroup(1, [AB])
    east = gmsh.model.addPhysicalGroup(1, [BC])
    north = gmsh.model.addPhysicalGroup(1, [CD])
    west = gmsh.model.addPhysicalGroup(1, [DA])
    domain = gmsh.model.addPhysicalGroup(2, [ABCD])
    gmsh.model.setPhysicalName(1, south, bnd_names[2])
    gmsh.model.setPhysicalName(1, east, bnd_names[3])
    gmsh.model.setPhysicalName(1, north, bnd_names[1])
    gmsh.model.setPhysicalName(1, west, bnd_names[4])
    gmsh.model.setPhysicalName(2, domain, "Domain")

    # Gen mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.partition(n_partitions)

    # Write result
    rm(output; force = true)
    gmsh.write(output)
    if write_geo
        output_geo = first(splitext(output)) * ".geo_unrolled"
        rm(output_geo; force = true)
        gmsh.write(output_geo)
    end

    # End
    gmsh.finalize()
end

"""
    gen_mesh_around_disk(
        output,
        type;
        r_in = 1.0,
        r_ext = 10.0,
        nθ = 360,
        nr = 100,
        nr_prog = 1.05,
        order = 1,
        recombine = true,
        bnd_names = ("Farfield", "Wall"),
        n_partitions = 0,
        kwargs...
    )

Mesh the 2D domain around a disk. `type` can be `:quad` or `:tri`.

For kwargs, see [`gen_line_mesh`](@ref).
"""
function gen_mesh_around_disk(
    output,
    type;
    r_int = 1.0,
    r_ext = 10.0,
    nθ = 360,
    nr = 100,
    nr_prog = 1.05,
    order = 1,
    recombine = true,
    bnd_names = ("Farfield", "Wall"),
    n_partitions = 0,
    write_geo = false,
    kwargs...,
)
    @assert type ∈ (:tri, :quad)

    gmsh.initialize()
    _apply_gmsh_options(; kwargs...)

    lc_ext = 2π * r_ext / (nθ - 1)
    lc_int = 2π * r_int / (nθ - 1)

    # Points
    O = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)

    Ae = gmsh.model.geo.addPoint(r_ext * cos(-3π / 4), r_ext * sin(-3π / 4), 0.0, lc_ext)
    Be = gmsh.model.geo.addPoint(r_ext * cos(-π / 4), r_ext * sin(-π / 4), 0.0, lc_ext)
    Ce = gmsh.model.geo.addPoint(r_ext * cos(π / 4), r_ext * sin(π / 4), 0.0, lc_ext)
    De = gmsh.model.geo.addPoint(r_ext * cos(3π / 4), r_ext * sin(3π / 4), 0.0, lc_ext)

    Ai = gmsh.model.geo.addPoint(r_int * cos(-3π / 4), r_int * sin(-3π / 4), 0.0, lc_int)
    Bi = gmsh.model.geo.addPoint(r_int * cos(-π / 4), r_int * sin(-π / 4), 0.0, lc_int)
    Ci = gmsh.model.geo.addPoint(r_int * cos(π / 4), r_int * sin(π / 4), 0.0, lc_int)
    Di = gmsh.model.geo.addPoint(r_int * cos(3π / 4), r_int * sin(3π / 4), 0.0, lc_int)

    # Curves
    AOBe = gmsh.model.geo.addCircleArc(Ae, O, Be)
    BOCe = gmsh.model.geo.addCircleArc(Be, O, Ce)
    CODe = gmsh.model.geo.addCircleArc(Ce, O, De)
    DOAe = gmsh.model.geo.addCircleArc(De, O, Ae)
    AOBi = gmsh.model.geo.addCircleArc(Ai, O, Bi)
    BOCi = gmsh.model.geo.addCircleArc(Bi, O, Ci)
    CODi = gmsh.model.geo.addCircleArc(Ci, O, Di)
    DOAi = gmsh.model.geo.addCircleArc(Di, O, Ai)

    # Surfaces
    if type == :quad
        # Curves
        AiAe = gmsh.model.geo.addLine(Ai, Ae)
        BiBe = gmsh.model.geo.addLine(Bi, Be)
        CiCe = gmsh.model.geo.addLine(Ci, Ce)
        DiDe = gmsh.model.geo.addLine(Di, De)

        # Contours
        _AB = gmsh.model.geo.addCurveLoop([AiAe, AOBe, -BiBe, -AOBi])
        _BC = gmsh.model.geo.addCurveLoop([BiBe, BOCe, -CiCe, -BOCi])
        _CD = gmsh.model.geo.addCurveLoop([CiCe, CODe, -DiDe, -CODi])
        _DA = gmsh.model.geo.addCurveLoop([DiDe, DOAe, -AiAe, -DOAi])

        # Surfaces
        AB = gmsh.model.geo.addPlaneSurface([_AB])
        BC = gmsh.model.geo.addPlaneSurface([_BC])
        CD = gmsh.model.geo.addPlaneSurface([_CD])
        DA = gmsh.model.geo.addPlaneSurface([_DA])

        # Mesh settings
        for arc in [AOBe, BOCe, CODe, DOAe, AOBi, BOCi, CODi, DOAi]
            gmsh.model.geo.mesh.setTransfiniteCurve(arc, round(Int, nθ / 4))
        end
        for rad in [AiAe, BiBe, CiCe, DiDe]
            gmsh.model.geo.mesh.setTransfiniteCurve(rad, nr, "Progression", nr_prog)
        end
        for surf in [AB, BC, CD, DA]
            gmsh.model.geo.mesh.setTransfiniteSurface(surf)
            recombine && gmsh.model.geo.mesh.setRecombine(2, surf)
        end

        surfaces = [AB, BC, CD, DA]

    elseif type == :tri
        # Contours
        loop_ext = gmsh.model.geo.addCurveLoop([AOBe, BOCe, CODe, DOAe])
        loop_int = gmsh.model.geo.addCurveLoop([AOBi, BOCi, CODi, DOAi])

        # Surface
        surfaces = [gmsh.model.geo.addPlaneSurface([loop_ext, loop_int])]
    end

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define boundaries (`1` stands for 1D, i.e lines)
    farfield = gmsh.model.addPhysicalGroup(1, [AOBe, BOCe, CODe, DOAe])
    wall = gmsh.model.addPhysicalGroup(1, [AOBi, BOCi, CODi, DOAi])
    domain = gmsh.model.addPhysicalGroup(2, surfaces)
    gmsh.model.setPhysicalName(1, farfield, bnd_names[1])
    gmsh.model.setPhysicalName(1, wall, bnd_names[2])
    gmsh.model.setPhysicalName(2, domain, "Domain")

    # Gen mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.partition(n_partitions)

    # Write result
    gmsh.write(output)
    if write_geo
        output_geo = first(splitext(output)) * ".geo_unrolled"
        rm(output_geo; force = true)
        gmsh.write(output_geo)
    end

    # End
    gmsh.finalize()
end

"""
    gen_rectangle_mesh_with_tri_and_quad(
        output;
        nx = 2,
        ny = 2,
        lx = 1.0,
        ly = 1.0,
        xc = -1.0,
        yc = -1.0,
        order = 1,
        n_partitions = 0,
        kwargs...
    )

Generate a 2D mesh of a rectangle domain and write the mesh to `output`. The domain is split vertically in two parts:
the upper part is composed of 'quad' cells and the lower part with 'tri'.
          North
         D ------- C
         |   :quad |
  West M₁|---------|M₂  East
         |   :tri  |
         A ------- B
           South

For kwargs, see [`gen_line_mesh`](@ref).
"""
function gen_rectangle_mesh_with_tri_and_quad(
    output;
    nx = 2,
    ny = 2,
    lx = 1.0,
    ly = 1.0,
    xc = -1.0,
    yc = -1.0,
    order = 1,
    n_partitions = 0,
    kwargs...,
)

    #         South
    gmsh.initialize()
    _apply_gmsh_options(; kwargs...)
    lc = 1e-1

    # Points
    A = gmsh.model.geo.addPoint(xc - lx / 2, yc - ly / 2, 0, lc)
    B = gmsh.model.geo.addPoint(xc + lx / 2, yc - ly / 2, 0, lc)
    C = gmsh.model.geo.addPoint(xc + lx / 2, yc + ly / 2, 0, lc)
    D = gmsh.model.geo.addPoint(xc - lx / 2, yc + ly / 2, 0, lc)

    M₁ = gmsh.model.geo.addPoint(xc - lx / 2, yc, 0, lc)
    M₂ = gmsh.model.geo.addPoint(xc + lx / 2, yc, 0, lc)

    # Lines
    AB = gmsh.model.geo.addLine(A, B)
    BM₂ = gmsh.model.geo.addLine(B, M₂)
    M₂C = gmsh.model.geo.addLine(M₂, C)
    CD = gmsh.model.geo.addLine(C, D)
    DM₁ = gmsh.model.geo.addLine(D, M₁)
    M₁A = gmsh.model.geo.addLine(M₁, A)
    M₁M₂ = gmsh.model.geo.addLine(M₁, M₂)

    # Contour
    ABM₂M₁ = gmsh.model.geo.addCurveLoop([AB, BM₂, -M₁M₂, M₁A])
    M₁M₂CD = gmsh.model.geo.addCurveLoop([M₁M₂, M₂C, CD, DM₁])

    # Surface
    lower_surf = gmsh.model.geo.addPlaneSurface([ABM₂M₁])
    upper_surf = gmsh.model.geo.addPlaneSurface([M₁M₂CD])

    # Mesh settings
    gmsh.model.geo.mesh.setTransfiniteCurve(AB, nx)
    gmsh.model.geo.mesh.setTransfiniteCurve(CD, nx)
    gmsh.model.geo.mesh.setTransfiniteCurve(M₁M₂, nx)
    gmsh.model.geo.mesh.setTransfiniteCurve(BM₂, floor(Int, ny / 2) + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(M₂C, floor(Int, ny / 2) + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(DM₁, floor(Int, ny / 2) + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(M₁A, floor(Int, ny / 2) + 1)

    gmsh.model.geo.mesh.setTransfiniteSurface(upper_surf)
    gmsh.model.geo.mesh.setRecombine(2, upper_surf)

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define boundaries (`1` stands for 1D, i.e lines)
    south = gmsh.model.addPhysicalGroup(1, [AB])
    east = gmsh.model.addPhysicalGroup(1, [BM₂, M₂C])
    north = gmsh.model.addPhysicalGroup(1, [CD])
    west = gmsh.model.addPhysicalGroup(1, [DM₁, M₁A])
    domain = gmsh.model.addPhysicalGroup(2, [ABM₂M₁, M₁M₂CD])
    gmsh.model.setPhysicalName(1, south, "South")
    gmsh.model.setPhysicalName(1, east, "East")
    gmsh.model.setPhysicalName(1, north, "North")
    gmsh.model.setPhysicalName(1, west, "West")
    gmsh.model.setPhysicalName(2, domain, "Domain")

    # Gen mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.partition(n_partitions)

    # Write result
    gmsh.write(output)

    # End
    gmsh.finalize()
end

"""
    gen_hexa_mesh(
        output,
        type;
        transfinite = false,
        nx = 2,
        ny = 2,
        nz = 2,
        lx = 1.0,
        ly = 1.0,
        lz = 1.0,
        xc = -1.0,
        yc = -1.0,
        zc = -1.0,
        order = 1,
        bnd_names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
        n_partitions = 0,
        write_geo = false,
        transfinite_lines = true,
        lc = 1e-1,
        kwargs...,
    )

Generate a 3D mesh of a hexahedral domain and write the mesh to `output`. Use `type` to specify the element types:
`:tetra` or `:hexa`.

For kwargs, see [`gen_line_mesh`](@ref).
"""
function gen_hexa_mesh(
    output,
    type;
    transfinite = false,
    nx = 2,
    ny = 2,
    nz = 2,
    lx = 1.0,
    ly = 1.0,
    lz = 1.0,
    xc = -1.0,
    yc = -1.0,
    zc = -1.0,
    order = 1,
    bnd_names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
    n_partitions = 0,
    write_geo = false,
    transfinite_lines = true,
    lc = 1e-1,
    kwargs...,
)
    @assert type ∈ (:tetra, :hexa) "`type` can only be :tetra or :hexa"
    @assert !((type == :hexa) && !transfinite_lines) "Cannot mix :hexa option with transfinite_lines=false"

    isHexa = type == :hexa

    gmsh.initialize()
    gmsh.model.add("model") # helps debugging
    _apply_gmsh_options(; kwargs...)

    # Points
    A = gmsh.model.geo.addPoint(xc - lx / 2, yc - ly / 2, zc - lz / 2, lc)
    B = gmsh.model.geo.addPoint(xc + lx / 2, yc - ly / 2, zc - lz / 2, lc)
    C = gmsh.model.geo.addPoint(xc + lx / 2, yc + ly / 2, zc - lz / 2, lc)
    D = gmsh.model.geo.addPoint(xc - lx / 2, yc + ly / 2, zc - lz / 2, lc)

    # Lines
    AB = gmsh.model.geo.addLine(A, B)
    BC = gmsh.model.geo.addLine(B, C)
    CD = gmsh.model.geo.addLine(C, D)
    DA = gmsh.model.geo.addLine(D, A)

    # Contour
    loop = gmsh.model.geo.addCurveLoop([AB, BC, CD, DA])

    # Surface
    ABCD = gmsh.model.geo.addPlaneSurface([loop])

    # Extrusion
    nlayers = (transfinite || isHexa || transfinite_lines) ? [nz - 1] : []
    recombine = (transfinite || isHexa)
    out = gmsh.model.geo.extrude([(2, ABCD)], 0, 0, lz, nlayers, [], recombine)

    # Identification
    zmin = ABCD
    zmax = out[1][2]
    vol = out[2][2]
    ymin = out[3][2]
    xmax = out[4][2]
    ymax = out[5][2]
    xmin = out[6][2]

    # Mesh settings
    if transfinite_lines
        gmsh.model.geo.mesh.setTransfiniteCurve(AB, nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(BC, ny)
        gmsh.model.geo.mesh.setTransfiniteCurve(CD, nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(DA, ny)
    end

    (transfinite || isHexa) && gmsh.model.geo.mesh.setTransfiniteSurface(ABCD)

    isHexa && gmsh.model.geo.mesh.setRecombine(2, ABCD)

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define boundaries (`1` stands for 1D, i.e lines)
    gmsh.model.addPhysicalGroup(2, [xmin], -1, bnd_names[1])
    gmsh.model.addPhysicalGroup(2, [xmax], -1, bnd_names[2])
    gmsh.model.addPhysicalGroup(2, [ymin], -1, bnd_names[3])
    gmsh.model.addPhysicalGroup(2, [ymax], -1, bnd_names[4])
    gmsh.model.addPhysicalGroup(2, [zmin], -1, bnd_names[5])
    gmsh.model.addPhysicalGroup(2, [zmax], -1, bnd_names[6])
    gmsh.model.addPhysicalGroup(3, [vol], -1, "Domain")

    # Gen mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.partition(n_partitions)

    # Write result
    rm(output; force = true)
    gmsh.write(output)
    if write_geo
        output_geo = first(splitext(output)) * ".geo_unrolled"
        rm(output_geo; force = true)
        gmsh.write(output_geo)
    end

    # End
    gmsh.finalize()
end

"""
    gen_disk_mesh(
        output;
        radius = 1.0,
        lc = 1e-1,
        order = 1,
        n_partitions = 0,
        kwargs...
    )

Generate a 2D mesh of a disk domain and write the mesh to `output`.

For kwargs, see [`gen_line_mesh`](@ref).
"""
function gen_disk_mesh(
    output;
    radius = 1.0,
    lc = 1e-1,
    order = 1,
    n_partitions = 0,
    kwargs...,
)
    gmsh.initialize()
    _apply_gmsh_options(; kwargs...)

    # Points
    O = gmsh.model.geo.addPoint(0, 0, 0, lc)
    A = gmsh.model.geo.addPoint(radius, 0, 0, lc)
    B = gmsh.model.geo.addPoint(-radius, 0, 0, lc)

    # Lines
    AOB = gmsh.model.geo.addCircleArc(A, O, B)
    BOA = gmsh.model.geo.addCircleArc(B, O, A)

    # Contour
    circle = gmsh.model.geo.addCurveLoop([AOB, BOA])

    # Surface
    disk = gmsh.model.geo.addPlaneSurface([circle])

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define boundaries (`1` stands for 1D, i.e lines)
    border = gmsh.model.addPhysicalGroup(1, [AOB, BOA])
    domain = gmsh.model.addPhysicalGroup(2, [disk])
    gmsh.model.setPhysicalName(1, border, "BORDER")
    gmsh.model.setPhysicalName(2, domain, "Domain")

    # Gen mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.partition(n_partitions)

    # Write result
    gmsh.write(output)

    # End
    gmsh.finalize()
end

"""
    gen_star_disk_mesh(
        output,
        ε,
        m;
        nθ = 360,
        radius = 1.0,
        lc = 1e-1,
        order = 1,
        n_partitions = 0,
        kwargs...,
    )

Generate a 2D mesh of a star domain and write the mesh to `output`. The "star" wall is defined by
``r_{wall} = R \\left( 1 + \\varepsilon \\cos(m \\theta) \\right)``.

For kwargs, see [`gen_line_mesh`](@ref).
"""
function gen_star_disk_mesh(
    output,
    ε,
    m;
    nθ = 360,
    radius = 1.0,
    lc = 1e-1,
    order = 1,
    n_partitions = 0,
    kwargs...,
)
    gmsh.initialize()
    _apply_gmsh_options(; kwargs...)

    # Alias
    R = radius

    # Points
    pts = []
    sizehint!(pts, nθ)
    for θ in range(0, 2π; length = nθ + 1) # 0 and 2π are identical so 2π will be removed
        rw = R * (1 + ε * cos(m * θ))
        push!(pts, gmsh.model.geo.addPoint(rw * cos(θ), rw * sin(θ), 0, lc))
    end
    pop!(pts) # remove duplicated point created by '2π'
    push!(pts, pts[1]) # repeat first point index -> needed for closed lines

    # Lines
    #spline = gmsh.model.geo.addSpline(pts) # Polyline seems not available in gmsh 4.0.5
    #polyline = gmsh.model.geo.addPolyline(pts) # Polyline seems not available in gmsh 4.0.5
    lines = []
    sizehint!(lines, nθ)
    for (p1, p2) in zip(pts, pts[2:end])
        push!(lines, gmsh.model.geo.addLine(p1, p2)) # the line [pn, p1] is taken into account
    end

    # # Contour
    #loop = gmsh.model.geo.addCurveLoop([spline])
    #loop = gmsh.model.geo.addCurveLoop([polyline])
    loop = gmsh.model.geo.addCurveLoop(lines)

    # # Surface
    disk = gmsh.model.geo.addPlaneSurface([loop])

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define boundaries (`1` stands for 1D, i.e lines)
    wall = gmsh.model.addPhysicalGroup(1, lines)
    domain = gmsh.model.addPhysicalGroup(2, [disk])
    gmsh.model.setPhysicalName(1, wall, "WALL")
    gmsh.model.setPhysicalName(2, domain, "Domain")

    # Gen mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.partition(n_partitions)

    # Write result
    gmsh.write(output)

    # End
    gmsh.finalize()
end

"""
    gen_cylinder_mesh(
        output,
        Lz,
        nz;
        radius = 1.0,
        lc = 1e-1,
        order = 1,
        n_partitions = 0,
        kwargs...
    )

Generate a 3D mesh of a cylindrical domain and length `L` and write the mesh to `output`.

For kwargs, see [`gen_line_mesh`](@ref).
"""
function gen_cylinder_mesh(
    output,
    Lz,
    nz;
    radius = 1.0,
    lc = 1e-1,
    order = 1,
    n_partitions = 0,
    kwargs...,
)
    gmsh.initialize()
    _apply_gmsh_options(; kwargs...)

    # Points -> need for 4 arcs for extrusion otherwise crash
    O = gmsh.model.geo.addPoint(0, 0, 0, lc)
    A = gmsh.model.geo.addPoint(radius, 0, 0, lc)
    B = gmsh.model.geo.addPoint(0, radius, 0, lc)
    C = gmsh.model.geo.addPoint(-radius, 0, 0, lc)
    D = gmsh.model.geo.addPoint(0, -radius, 0, lc)

    # Lines
    AOB = gmsh.model.geo.addCircleArc(A, O, B)
    BOC = gmsh.model.geo.addCircleArc(B, O, C)
    COD = gmsh.model.geo.addCircleArc(C, O, D)
    DOA = gmsh.model.geo.addCircleArc(D, O, A)

    # Contour
    circle = gmsh.model.geo.addCurveLoop([AOB, BOC, COD, DOA])

    # Surface
    disk = gmsh.model.geo.addPlaneSurface([circle])
    gmsh.model.geo.synchronize()

    # Extrude : 1 layer divided in `nz` heights
    out = gmsh.model.geo.extrude([(2, disk)], 0.0, 0.0, Lz, [nz], [], true)
    # out[1] = (dim, tag) is the top of disk extruded
    # out[2] = (dim, tag) is the volume created by the extrusion
    # out[3] = (dim, tag) is the side created by the extrusion of AOB, (1st first element of contour `circle`)
    # out[4] = (dim, tag) is the side created by the extrusion of BOC, (2nd first element of contour `circle`)
    # out[5] = (dim, tag) is the side created by the extrusion of COD, (3rd first element of contour `circle`)
    # out[6] = (dim, tag) is the side created by the extrusion of DOA, (4fh first element of contour `circle`)

    # Synchronize
    gmsh.model.geo.synchronize()

    # Define boundaries (`1` stands for 1D, i.e lines)
    border = gmsh.model.addPhysicalGroup(2, [out[3][2], out[4][2], out[5][2], out[6][2]])
    inlet = gmsh.model.addPhysicalGroup(2, [disk])
    outlet = gmsh.model.addPhysicalGroup(2, [out[1][2]])
    domain = gmsh.model.addPhysicalGroup(3, [out[2][2]])
    gmsh.model.setPhysicalName(2, border, "BORDER")
    gmsh.model.setPhysicalName(2, inlet, "INLET")
    gmsh.model.setPhysicalName(2, outlet, "OUTLET")
    gmsh.model.setPhysicalName(3, domain, "Domain")

    # Gen mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.partition(n_partitions)

    # Write result
    gmsh.write(output)

    # End
    gmsh.finalize()
end

"""
    gen_sphere_mesh(
        output;
        radius = 1.0,
        lc = 1e-1,
        order = 1,
        n_partitions = 0,
        kwargs...,
    )

Generate the mesh of a sphere (surface of topological dimension 2, spatial dimension 3).
"""
function gen_sphere_mesh(
    output;
    radius = 1.0,
    lc = 1e-1,
    order = 1,
    n_partitions = 0,
    kwargs...,
)
    gmsh.initialize()
    _apply_gmsh_options(; kwargs...)

    # Points
    P1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    P2 = gmsh.model.geo.addPoint(radius, 0, 0, lc)
    P3 = gmsh.model.geo.addPoint(0, radius, 0, lc)
    P4 = gmsh.model.geo.addPoint(0, 0, radius, lc)
    P5 = gmsh.model.geo.addPoint(-radius, 0, 0, lc)
    P6 = gmsh.model.geo.addPoint(0, -radius, 0, lc)
    P7 = gmsh.model.geo.addPoint(0, 0, -radius, lc)

    # Line
    C1 = gmsh.model.geo.addCircleArc(P2, P1, P3)
    C2 = gmsh.model.geo.addCircleArc(P3, P1, P5)
    C3 = gmsh.model.geo.addCircleArc(P5, P1, P6)
    C4 = gmsh.model.geo.addCircleArc(P6, P1, P2)
    C5 = gmsh.model.geo.addCircleArc(P2, P1, P7)
    C6 = gmsh.model.geo.addCircleArc(P7, P1, P5)
    C7 = gmsh.model.geo.addCircleArc(P5, P1, P4)
    C8 = gmsh.model.geo.addCircleArc(P4, P1, P2)
    C9 = gmsh.model.geo.addCircleArc(P6, P1, P7)
    C10 = gmsh.model.geo.addCircleArc(P7, P1, P3)
    C11 = gmsh.model.geo.addCircleArc(P3, P1, P4)
    C12 = gmsh.model.geo.addCircleArc(P4, P1, P6)

    # Loops
    LL1 = gmsh.model.geo.addCurveLoop([C1, C11, C8])
    LL2 = gmsh.model.geo.addCurveLoop([C2, C7, -C11])
    LL3 = gmsh.model.geo.addCurveLoop([C3, -C12, -C7])
    LL4 = gmsh.model.geo.addCurveLoop([C4, -C8, C12])
    LL5 = gmsh.model.geo.addCurveLoop([C5, C10, -C1])
    LL6 = gmsh.model.geo.addCurveLoop([-C2, -C10, C6])
    LL7 = gmsh.model.geo.addCurveLoop([-C3, -C6, -C9])
    LL8 = gmsh.model.geo.addCurveLoop([-C4, C9, -C5])

    # Surfaces
    RS = map([LL1, LL2, LL3, LL4, LL5, LL6, LL7, LL8]) do LL
        gmsh.model.geo.addSurfaceFilling([LL])
    end

    # Domains
    sphere = gmsh.model.addPhysicalGroup(2, RS)
    gmsh.model.setPhysicalName(2, sphere, "Sphere")

    # Gen mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.partition(n_partitions)

    # Write result
    gmsh.write(output)

    # End
    gmsh.finalize()
end

"""
    _gen_2cubes_mesh(output)

Only for testing purpose.

D------E------F
|      |      |
|      |      |
A------B------C
"""
function _gen_2cubes_mesh(output)
    lc = 1.0

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    # Points
    A = gmsh.model.geo.addPoint(0, 0, 0, lc)
    B = gmsh.model.geo.addPoint(1, 0, 0, lc)
    C = gmsh.model.geo.addPoint(2, 0, 0, lc)
    D = gmsh.model.geo.addPoint(0, 1, 0, lc)
    E = gmsh.model.geo.addPoint(1, 1, 0, lc)
    F = gmsh.model.geo.addPoint(2, 1, 0, lc)

    # Line
    AB = gmsh.model.geo.addLine(A, B)
    BC = gmsh.model.geo.addLine(B, C)
    DE = gmsh.model.geo.addLine(D, E)
    EF = gmsh.model.geo.addLine(E, F)
    AD = gmsh.model.geo.addLine(A, D)
    BE = gmsh.model.geo.addLine(B, E)
    CF = gmsh.model.geo.addLine(C, F)

    # Surfaces
    ABED = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([AB, BE, -DE, -AD])])
    BCFE = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([BC, CF, -EF, -BE])])

    # Extrusion
    gmsh.model.geo.extrude([(2, ABED), (2, BCFE)], 0, 0, 1, [1], [], true)

    for l in (AB, BC, DE, EF, AD, BE, CF)
        gmsh.model.geo.mesh.setTransfiniteCurve(l, 2)
    end

    for s in (ABED, BCFE)
        gmsh.model.geo.mesh.setTransfiniteSurface(s)
        gmsh.model.geo.mesh.setRecombine(2, s)
    end
    # Gen mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    # Write result
    gmsh.write(output)

    # End
    gmsh.finalize()
end

"""
    gen_cylinder_shell_mesh(
        output,
        nθ,
        nz;
        radius = 1.0,
        lz = 1.0,
        lc = 1e-1,
        order = 1,
        n_partitions = 0,
        recombine = false,
        transfinite = false,
        kwargs...,
    )

# Implementation
Extrusion is not used to enable "random" tri filling (whereas with extrusion we can at worse obtain regular rectangle triangle)
"""
function gen_cylinder_shell_mesh(
    output,
    nθ,
    nz;
    radius = 1.0,
    lz = 1.0,
    lc = 1e-1,
    order = 1,
    n_partitions = 0,
    recombine = false,
    transfinite = false,
    kwargs...,
)
    gmsh.initialize()
    gmsh.model.add("model") # helps debugging
    _apply_gmsh_options(; kwargs...)

    # Points
    O1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    O2 = gmsh.model.geo.addPoint(0, 0, lz, lc)

    A1 = gmsh.model.geo.addPoint(radius, 0, 0, lc)
    B1 = gmsh.model.geo.addPoint(radius * cos(2π / 3), radius * sin(2π / 3), 0, lc)
    C1 = gmsh.model.geo.addPoint(radius * cos(4π / 3), radius * sin(4π / 3), 0, lc)

    A2 = gmsh.model.geo.addPoint(radius, 0, lz, lc)
    B2 = gmsh.model.geo.addPoint(radius * cos(2π / 3), radius * sin(2π / 3), lz, lc)
    C2 = gmsh.model.geo.addPoint(radius * cos(4π / 3), radius * sin(4π / 3), lz, lc)

    # Lines
    AOB1 = gmsh.model.geo.addCircleArc(A1, O1, B1)
    BOC1 = gmsh.model.geo.addCircleArc(B1, O1, C1)
    COA1 = gmsh.model.geo.addCircleArc(C1, O1, A1)
    AOB2 = gmsh.model.geo.addCircleArc(A2, O2, B2)
    BOC2 = gmsh.model.geo.addCircleArc(B2, O2, C2)
    COA2 = gmsh.model.geo.addCircleArc(C2, O2, A2)

    A1A2 = gmsh.model.geo.addLine(A1, A2)
    B1B2 = gmsh.model.geo.addLine(B1, B2)
    C1C2 = gmsh.model.geo.addLine(C1, C2)

    # Surfaces
    loops = [
        gmsh.model.geo.addCurveLoop([AOB1, B1B2, -AOB2, -A1A2]),
        gmsh.model.geo.addCurveLoop([BOC1, C1C2, -BOC2, -B1B2]),
        gmsh.model.geo.addCurveLoop([COA1, A1A2, -COA2, -C1C2]),
    ]

    surfs = map(loop -> gmsh.model.geo.addSurfaceFilling([loop]), loops)

    # Mesh settings
    if transfinite
        _nθ = round(Int, nθ / 3)
        foreach(
            line -> gmsh.model.geo.mesh.setTransfiniteCurve(line, _nθ),
            (AOB1, BOC1, COA1, AOB2, BOC2, COA2),
        )
        foreach(
            line -> gmsh.model.geo.mesh.setTransfiniteCurve(line, nz),
            (A1A2, B1B2, C1C2),
        )
        foreach(gmsh.model.geo.mesh.setTransfiniteSurface, surfs)
    end
    recombine && map(surf -> gmsh.model.geo.mesh.setRecombine(2, surf), surfs)

    gmsh.model.geo.synchronize()

    # Define boundaries (`1` stands for 1D, i.e lines)
    domain = gmsh.model.addPhysicalGroup(2, surfs)
    bottom = gmsh.model.addPhysicalGroup(1, [AOB1, BOC1, COA1])
    top = gmsh.model.addPhysicalGroup(1, [AOB2, BOC2, COA2])
    gmsh.model.setPhysicalName(1, bottom, "zmin")
    gmsh.model.setPhysicalName(1, top, "zmax")
    gmsh.model.setPhysicalName(2, domain, "domain")

    # Gen mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.partition(n_partitions)

    # Write result
    gmsh.write(output)

    # End
    gmsh.finalize()
end

"""
    _gen_cube_pile(output)

Only for testing purpose.

       G------H
       |      |
       |      |
D------E------F
|      |      |
|      |      |
A------B------C
"""
function _gen_cube_pile(output)
    lc = 1.0

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    # Points
    A = gmsh.model.geo.addPoint(0, 0, 0, lc)
    B = gmsh.model.geo.addPoint(1, 0, 0, lc)
    C = gmsh.model.geo.addPoint(2, 0, 0, lc)
    D = gmsh.model.geo.addPoint(0, 1, 0, lc)
    E = gmsh.model.geo.addPoint(1, 1, 0, lc)
    F = gmsh.model.geo.addPoint(2, 1, 0, lc)
    G = gmsh.model.geo.addPoint(1, 2, 0, lc)
    H = gmsh.model.geo.addPoint(2, 2, 0, lc)

    # Line
    AB = gmsh.model.geo.addLine(A, B)
    BC = gmsh.model.geo.addLine(B, C)
    DE = gmsh.model.geo.addLine(D, E)
    EF = gmsh.model.geo.addLine(E, F)
    GH = gmsh.model.geo.addLine(G, H)
    AD = gmsh.model.geo.addLine(A, D)
    BE = gmsh.model.geo.addLine(B, E)
    CF = gmsh.model.geo.addLine(C, F)
    EG = gmsh.model.geo.addLine(E, G)
    FH = gmsh.model.geo.addLine(F, H)

    # Surfaces
    ABED = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([AB, BE, -DE, -AD])])
    BCFE = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([BC, CF, -EF, -BE])])
    EFHG = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([EF, FH, -GH, -EG])])

    # Extrusion
    out = gmsh.model.geo.extrude([(2, ABED), (2, BCFE), (2, EFHG)], 0, 0, 1, [1], [], true)

    gmsh.model.geo.extrude([out[7]], 0, 0, 1, [1], [], true)

    for l in (AB, BC, DE, EF, GH, AD, BE, CF, EG, FH)
        gmsh.model.geo.mesh.setTransfiniteCurve(l, 2)
    end

    for s in (ABED, BCFE, EFHG)
        gmsh.model.geo.mesh.setTransfiniteSurface(s)
        gmsh.model.geo.mesh.setRecombine(2, s)
    end
    # Gen mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    # Write result
    gmsh.write(output)

    # End
    gmsh.finalize()
end

"""
    gen_torus_shell_mesh(output, rint, rext; order = 1, lc = 0.1, write_geo = false, kwargs...)

Generate the mesh of the shell of a torus, defined by its inner radius `rint` and exterior radius `rext`.

The torus revolution axis is the z-axis.
"""
function gen_torus_shell_mesh(
    output,
    rint,
    rext;
    order = 1,
    lc = 0.1,
    write_geo = false,
    n_partitions = 0,
    kwargs...,
)
    gmsh.initialize()
    gmsh.model.add("model") # helps debugging
    _apply_gmsh_options(; kwargs...)

    # Points
    xc = (rint + rext) / 2
    r = (rext - rint) / 2
    O = gmsh.model.geo.addPoint(xc, 0, 0, lc)
    A = gmsh.model.geo.addPoint(xc + r, 0, 0, lc)
    B = gmsh.model.geo.addPoint(xc + r * cos(2π / 3), 0, r * sin(2π / 3), lc)
    C = gmsh.model.geo.addPoint(xc + r * cos(4π / 3), 0, r * sin(4π / 3), lc)

    # Lines
    AOB = gmsh.model.geo.addCircleArc(A, O, B)
    BOC = gmsh.model.geo.addCircleArc(B, O, C)
    COA = gmsh.model.geo.addCircleArc(C, O, A)

    # Surfaces
    opts = (0, 0, 0, 0, 0, 1, 2π / 3)
    out = gmsh.model.geo.revolve([(1, AOB), (1, BOC), (1, COA)], opts...)
    rev1_AOB = out[1:4]
    rev1_BOC = out[5:8]
    rev1_COA = out[9:12]
    out = gmsh.model.geo.revolve([rev1_AOB[1], rev1_BOC[1], rev1_COA[1]], opts...)
    rev2_AOB = out[1:4]
    rev2_BOC = out[5:8]
    rev2_COA = out[9:12]
    out = gmsh.model.geo.revolve([rev2_AOB[1], rev2_BOC[1], rev2_COA[1]], opts...)
    rev3_AOB = out[1:4]
    rev3_BOC = out[5:8]
    rev3_COA = out[9:12]

    # Gen mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.partition(n_partitions)

    # Write result
    gmsh.write(output)
    if write_geo
        output_geo = first(splitext(output)) * ".geo_unrolled"
        gmsh.write(output_geo)
    end

    # End
    gmsh.finalize()
end

"""
    nodes_gmsh2cgns(entity::AbstractEntityType, nodes::AbstractArray)

Reorder `nodes` of a given `entity` from the Gmsh format to CGNS format.

See https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
"""
function nodes_gmsh2cgns(entity::AbstractEntityType, nodes::AbstractArray)
    map(i -> nodes[i], nodes_gmsh2cgns(entity))
end

nodes_gmsh2cgns(e) = nodes_gmsh2cgns(typeof(e))
function nodes_gmsh2cgns(::Type{<:T}) where {T <: AbstractEntityType}
    error("Function nodes_gmsh2cgns is not defined for type $T")
end
nodes_gmsh2cgns(e::Type{Node_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Bar2_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Bar3_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Bar4_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Bar5_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Tri3_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Tri6_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Tri9_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Tri10_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Tri12_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Quad4_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Quad8_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Quad9_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Quad16_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Tetra4_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Tetra10_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Hexa8_t}) = nodes(e) #same numbering between CGNS and Gmsh
function nodes_gmsh2cgns(::Type{Hexa27_t})
    SA[
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        17,
        18,
        19,
        20,
        13,
        14,
        15,
        16,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
    ]
end
nodes_gmsh2cgns(e::Type{Penta6_t}) = nodes(e) #same numbering between CGNS and Gmsh
nodes_gmsh2cgns(e::Type{Pyra5_t}) = nodes(e) #same numbering between CGNS and Gmsh

"""
Convert a cell->node connectivity with gmsh numbering convention to a cell->node connectivity
with CGNs numbering convention.
"""
function _c2n_gmsh2cgns(celltypes, c2n_gmsh)
    n = Int[]
    indices = Int[]
    for (ct, c2nᵢ) in zip(celltypes, c2n_gmsh)
        append!(n, length(c2nᵢ))
        append!(indices, nodes_gmsh2cgns(ct, c2nᵢ))
    end
    return Connectivity(n, indices)
end
