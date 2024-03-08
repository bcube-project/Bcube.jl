"""
    basic_mesh()

Generate a toy mesh of two quads and one triangle.

 v1       v2       v3       v4
 +---e1-->+---e5-->+---e8-->+
 ^        |        |  c3  /
e4   c1   e2   c2  e6   e9
 |        |        |  /
 +<--e3---+<--e7---+/
 v5      v6        v7
"""
function basic_mesh(; coef_x = 1.0, coef_y = 1.0)
    x = coef_x .* [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0]
    y = coef_y .* [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    nodes = [Node([_x, _y]) for (_x, _y) in zip(x, y)]
    celltypes = Union{Quad4_t, Tri3_t}[Quad4_t(), Quad4_t(), Tri3_t()]
    # cell->node connectivity indices :
    cell2node = Connectivity([4, 4, 3], [1, 2, 6, 5, 2, 3, 7, 6, 3, 4, 7])

    # Prepare boundary nodes
    bnd_name = "BORDER"
    tag2name = Dict(1 => bnd_name)
    tag2nodes = Dict(1 => collect(1:7))

    return Mesh(nodes, celltypes, cell2node; bc_names = tag2name, bc_nodes = tag2nodes)
end

"""
    one_cell_mesh(type::Symbol, order = 1)

Generate a mesh of one cell. `type` can be `:line`, `:quad`, `:tri` or `:hexa`.

The argument `order` refers to the geometry order. It has the same effect
as the `-order` parameter in gmsh.
"""
function one_cell_mesh(
    type::Symbol;
    xmin = -1.0,
    xmax = 1.0,
    ymin = -1,
    ymax = 1,
    zmin = -1,
    zmax = 1.0,
    order = 1,
)
    if (type == :line || type == :bar)
        return one_line_mesh(Val(order), xmin, xmax)
    elseif (type == :quad || type == :square)
        return one_quad_mesh(Val(order), xmin, xmax, ymin, ymax)
    elseif (type == :tri || type == :triangle)
        return one_tri_mesh(Val(order), xmin, xmax, ymin, ymax)
    elseif (type == :tetra)
        return one_tetra_mesh(Val(order), xmin, xmax, ymin, ymax, zmin, zmax)
    elseif (type == :hexa || type == :cube)
        return one_hexa_mesh(Val(order), xmin, xmax, ymin, ymax, zmin, zmax)
    elseif (type == :prism || type == :penta)
        return one_prism_mesh(Val(order), xmin, xmax, ymin, ymax, zmin, zmax)
    else
        throw(ArgumentError("Expected type :line, :quad, :tri, :hexa"))
    end
end

"""
    ncube_mesh(n::Vector{Int}; order = 1)

Generate either a line mesh, a rectangle mesh, a cubic mesh... depending on the dimension of `n`.

# Argument
- `n` number of vertices in each spatial directions

# Example
```julia-repl
mesh_of_a_line   = ncube_mesh([10])
mesh_of_a_square = ncube_mesh([4, 5])
```
"""
function ncube_mesh(n::Vector{Int}; order = 1)
    if (length(n) == 1)
        return line_mesh(n[1]; order = order)
    elseif (length(n) == 2)
        return rectangle_mesh(n[1], n[2]; order = order)
    else
        throw(ArgumentError("mesh not available for R^" * string(length(n))))
    end
end

"""
    line_mesh(n; xmin = 0., xmax = 1., order = 1, names = ("LEFT", "RIGHT"))

Generate a mesh of a line of `n` vertices.

# Example
```julia-repl
julia> mesh = line_mesh(5)
```
"""
function line_mesh(n; xmin = 0.0, xmax = 1.0, order = 1, names = ("LEFT", "RIGHT"))
    l = xmax - xmin # line length
    nelts = n - 1 # Number of cells

    # Linear elements
    if (order == 1)
        Δx = l / (n - 1)

        # Nodes
        nodes = [Node([xmin + (i - 1) * Δx]) for i in 1:n]

        # Cell type is constant
        celltypes = [Bar2_t() for ielt in 1:nelts]

        # Cell -> nodes connectivity
        cell2node = zeros(Int, 2 * nelts)
        for ielt in 1:nelts
            cell2node[2 * ielt - 1] = ielt
            cell2node[2 * ielt]     = ielt + 1
        end

        # Number of nodes of each cell : always 2
        cell2nnodes = 2 * ones(Int, nelts)

        # Boundaries
        bc_names, bc_nodes = one_line_bnd(1, n, names)

        # Mesh
        return Mesh(
            nodes,
            celltypes,
            Connectivity(cell2nnodes, cell2node);
            bc_names,
            bc_nodes,
        )

        # Quadratic elements
    elseif (order == 2)
        Δx = l / 2 / (n - 1)

        # Cell type is constant
        celltypes = [Bar3_t() for ielt in 1:nelts]

        # Nodes + Cell -> nodes connectivity
        nodes = Array{Node{1, Float64}}(undef, 2 * n - 1)
        cell2node = zeros(Int, 3 * nelts)
        nodes[1] = Node([xmin]) # First node
        i = 1 # init counter
        for ielt in 1:nelts
            # two new nodes : middle one and then right boundary
            nodes[i + 1] = xmin + i * Δx
            nodes[i + 2] = xmin + (i + 1) * Δx

            # cell -> node
            cell2node[3 * ielt - 2] = i
            cell2node[3 * ielt - 1] = i + 2
            cell2node[3 * ielt - 0] = i + 1 # middle node

            # Update counter
            i += 2
        end

        # Number of nodes of each cell : always 3
        cell2nnodes = 3 * ones(Int, nelts)

        # Boundaries
        bc_names, bc_nodes = one_line_bnd(1, n, names)

        # Mesh
        return Mesh(
            nodes,
            celltypes,
            Connectivity(cell2nnodes, cell2node);
            bc_names,
            bc_nodes,
        )

    else
        throw(
            ArgumentError(
                "`order` must be <= 2 (but feel free to implement higher orders)",
            ),
        )
    end
end

"""
    rectangle_mesh(
        nx,
        ny;
        type = :quad,
        xmin = 0.0,
        xmax = 1.0,
        ymin = 0.0,
        ymax = 1.0,
        order = 1,
        bnd_names = ("north", "south", "east", "west"),
    )

Generate a 2D mesh of a rectangle with `nx` and `ny` vertices in the x and y directions respectively.

# Example
```julia-repl
julia> mesh = rectangle_mesh(5, 4)
```
"""
function rectangle_mesh(
    nx,
    ny;
    type = :quad,
    xmin = 0.0,
    xmax = 1.0,
    ymin = 0.0,
    ymax = 1.0,
    order = 1,
    bnd_names = ("north", "south", "east", "west"),
)
    @assert (nx > 1 && ny > 1) "`nx` and `ny`, the number of nodes, must be greater than 1 (nx=$nx, ny=$ny)"

    if (type == :quad)
        return _rectangle_quad_mesh(nx, ny, xmin, xmax, ymin, ymax, Val(order), bnd_names)
    else
        throw(
            ArgumentError("`type` must be :quad (but feel free to implement other types)"),
        )
    end
end

function _rectangle_quad_mesh(nx, ny, xmin, xmax, ymin, ymax, ::Val{1}, bnd_names)
    # Notes
    # 6-----8-----9
    # |     |     |
    # |  3  |  4  |
    # |     |     |
    # 4-----5-----6
    # |     |     |
    # |  1  |  2  |
    # |     |     |
    # 1-----2-----3

    lx = xmax - xmin
    ly = ymax - ymin
    nelts = (nx - 1) * (ny - 1)
    Δx = lx / (nx - 1)
    Δy = ly / (ny - 1)

    # Prepare boundary nodes
    tag2name = Dict(tag => name for (tag, name) in enumerate(bnd_names))
    tag2nodes = Dict(tag => Int[] for tag in 1:length(bnd_names))

    # Nodes
    iglob = 1
    nodes = Array{Node{2, Float64}}(undef, nx * ny)
    for iy in 1:ny
        for ix in 1:nx
            nodes[(iy - 1) * nx + ix] = Node([xmin + (ix - 1) * Δx, ymin + (iy - 1) * Δy])

            # Boundary conditions
            if ix == 1 # West
                push!(tag2nodes[4], iglob)
            elseif ix == nx # East
                push!(tag2nodes[3], iglob)
            elseif iy == 1
                push!(tag2nodes[2], iglob)
            elseif iy == ny
                push!(tag2nodes[1], iglob)
            end

            iglob += 1
        end
    end

    # Cell -> node connectivity
    cell2node = zeros(Int, 4 * nelts)
    for iy in 1:(ny - 1)
        for ix in 1:(nx - 1)
            ielt = (iy - 1) * (nx - 1) + ix
            cell2node[4 * ielt - 3] = (iy + 0 - 1) * nx + ix + 0
            cell2node[4 * ielt - 2] = (iy + 0 - 1) * nx + ix + 1
            cell2node[4 * ielt - 1] = (iy + 1 - 1) * nx + ix + 1
            cell2node[4 * ielt - 0] = (iy + 1 - 1) * nx + ix + 0
        end
    end

    # Cell type is constant
    celltypes = [Quad4_t() for ielt in 1:nelts]

    # Number of nodes of each cell : always 4
    cell2nnodes = 4 * ones(Int, nelts)

    return Mesh(
        nodes,
        celltypes,
        Connectivity(cell2nnodes, cell2node);
        bc_names = tag2name,
        bc_nodes = tag2nodes,
    )
end

# Remark : with a rectangle domain of quadratic elements, we can then apply a mapping on
# this rectangle domain to obtain a curved mesh...
function _rectangle_quad_mesh(nx, ny, xmin, xmax, ymin, ymax, ::Val{2}, bnd_names)
    # Notes
    # 07----08----09
    # |           |
    # |           |
    # 04    05    06
    # |           |
    # |           |
    # 01----02----03
    #
    # 11----12----13----14----15
    # |           |           |
    # |           |           |
    # 06    07    08    09    10
    # |           |           |
    # |           |           |
    # 01----02----03----04----05

    lx = xmax - xmin
    ly = ymax - ymin
    nelts = (nx - 1) * (ny - 1)
    nnodes = nx * ny + (nx - 1) * ny + (ny - 1) * nx + nelts
    Δx = lx / (nx - 1) / 2
    Δy = ly / (ny - 1) / 2

    # Nodes
    # we override some nodes multiple times, but it is easier this way
    nodes = Array{Node{2, Float64}}(undef, nnodes)
    for iy in 1:(ny + (ny - 1))
        for ix in 1:(nx + (nx - 1))
            nodes[(iy - 1) * (nx + (nx - 1)) + ix] =
                Node([xmin + (ix - 1) * Δx, ymin + (iy - 1) * Δy])
        end
    end

    # Cell -> node connectivity
    cell2node = zeros(Int, 9 * nelts)
    for j in 1:(ny - 1)
        for i in 1:(nx - 1)
            ielt = (j - 1) * (nx - 1) + i

            ix = (i - 1) * 2 + 1
            iy = (j - 1) * 2 + 1

            cell2node[9 * ielt - 8] = (iy + 0 - 1) * (nx + (nx - 1)) + ix + 0
            cell2node[9 * ielt - 7] = (iy + 0 - 1) * (nx + (nx - 1)) + ix + 2
            cell2node[9 * ielt - 6] = (iy + 2 - 1) * (nx + (nx - 1)) + ix + 2
            cell2node[9 * ielt - 5] = (iy + 2 - 1) * (nx + (nx - 1)) + ix + 0

            cell2node[9 * ielt - 4] = (iy + 0 - 1) * (nx + (nx - 1)) + ix + 1
            cell2node[9 * ielt - 3] = (iy + 1 - 1) * (nx + (nx - 1)) + ix + 2
            cell2node[9 * ielt - 2] = (iy + 2 - 1) * (nx + (nx - 1)) + ix + 1
            cell2node[9 * ielt - 1] = (iy + 1 - 1) * (nx + (nx - 1)) + ix + 0

            cell2node[9 * ielt - 0] = (iy + 1 - 1) * (nx + (nx - 1)) + ix + 1
        end
    end

    # Cell type is constant
    celltypes = [Quad9_t() for ielt in 1:nelts]

    # Number of nodes of each cell : always 4
    cell2nnodes = 9 * ones(Int, nelts)

    return Mesh(nodes, celltypes, Connectivity(cell2nnodes, cell2node))
end

function one_line_mesh(::Val{1}, xmin, xmax)
    nodes = [Node([xmin]), Node([xmax])]
    celltypes = [Bar2_t()]
    cell2node = Connectivity([2], [1, 2])
    bc_names, bc_nodes = one_line_bnd()
    return Mesh(nodes, celltypes, cell2node; bc_names = bc_names, bc_nodes = bc_nodes)
end

function one_line_mesh(::Val{2}, xmin, xmax)
    nodes = [Node([xmin]), Node([xmax]), Node([(xmin + xmax) / 2])]
    celltypes = [Bar3_t()]
    cell2node = Connectivity([3], [1, 2, 3])
    bc_names, bc_nodes = one_line_bnd()
    return Mesh(nodes, celltypes, cell2node; bc_names, bc_nodes)
end

function one_line_bnd(ileft = 1, iright = 2, names = ("LEFT", "RIGHT"))
    bc_names = Dict(1 => names[1], 2 => names[2])
    bc_nodes = Dict(1 => [ileft], 2 => [iright])
    return bc_names, bc_nodes
end

function one_tri_mesh(::Val{1}, xmin, xmax, ymin, ymax)
    nodes = [Node([xmin, ymin]), Node([xmax, ymin]), Node([xmin, ymax])]
    celltypes = [Tri3_t()]
    cell2node = Connectivity([3], [1, 2, 3])
    return Mesh(nodes, celltypes, cell2node)
end

function one_tri_mesh(::Val{2}, xmin, xmax, ymin, ymax)
    x1 = [xmin, ymin]
    x2 = [xmax, ymin]
    x3 = [xmin, ymax]
    nodes = [
        Node(x1),
        Node(x2),
        Node(x3),
        Node((x1 + x2) / 2),
        Node((x2 + x3) / 2),
        Node((x3 + x1) / 2),
    ]
    celltypes = [Tri6_t()]
    cell2node = Connectivity([6], [1, 2, 3, 4, 5, 6])
    return Mesh(nodes, celltypes, cell2node)
end

function one_quad_mesh(::Val{1}, xmin, xmax, ymin, ymax)
    nodes = [Node([xmin, ymin]), Node([xmax, ymin]), Node([xmax, ymax]), Node([xmin, ymax])]
    celltypes = [Quad4_t()]
    cell2node = Connectivity([4], [1, 2, 3, 4])
    return Mesh(nodes, celltypes, cell2node)
end

function one_quad_mesh(::Val{2}, xmin, xmax, ymin, ymax)
    x1 = [xmin, ymin]
    x2 = [xmax, ymin]
    x3 = [xmax, ymax]
    x4 = [xmin, ymax]
    nodes = [
        Node([xmin, ymin]),
        Node([xmax, ymin]),
        Node([xmax, ymax]),
        Node([xmin, ymax]),
        Node((x1 + x2) / 2),
        Node((x2 + x3) / 2),
        Node((x3 + x4) / 2),
        Node((x4 + x1) / 2),
        Node([(xmin + xmax) / 2, (ymin + ymax) / 2]),
    ]
    celltypes = [Quad9_t()]
    cell2node = Connectivity([9], collect(1:9))
    return Mesh(nodes, celltypes, cell2node)
end

function one_quad_mesh(::Val{3}, xmin, xmax, ymin, ymax)
    nodes = [
        Node([xmin, ymin]), # 1
        Node([xmax, ymin]), # 2
        Node([xmax, ymax]), # 3
        Node([xmin, ymax]), # 4
        Node([xmin + 1 * (xmax - xmin) / 3, ymin]), # 5
        Node([xmin + 2 * (xmax - xmin) / 3, ymin]), # 6
        Node([xmax, ymin + 1 * (ymax - ymin) / 3]), # 7
        Node([xmax, ymin + 2 * (ymax - ymin) / 3]), # 8
        Node([xmin + 2 * (xmax - xmin) / 3, ymax]), # 9
        Node([xmin + 1 * (xmax - xmin) / 3, ymax]), # 10
        Node([xmin, ymin + 2 * (ymax - ymin) / 3]), # 11
        Node([xmin, ymin + 1 * (ymax - ymin) / 3]), # 12
        Node([xmin + 1 * (xmax - xmin) / 3, ymin + 1 * (ymax - ymin) / 3]), # 13
        Node([xmin + 2 * (xmax - xmin) / 3, ymin + 1 * (ymax - ymin) / 3]), # 14
        Node([xmin + 2 * (xmax - xmin) / 3, ymin + 2 * (ymax - ymin) / 3]), # 15
        Node([xmin + 1 * (xmax - xmin) / 3, ymin + 2 * (ymax - ymin) / 3]), # 16
    ]
    celltypes = [Quad16_t()]
    cell2node = Connectivity([16], collect(1:16))
    return Mesh(nodes, celltypes, cell2node)
end

function one_tetra_mesh(::Val{1}, xmin, xmax, ymin, ymax, zmin, zmax)
    nodes = [
        Node([xmin, ymin, zmin]),
        Node([xmax, ymin, zmin]),
        Node([xmin, ymax, zmin]),
        Node([xmin, ymin, zmax]),
    ]
    celltypes = [Tetra4_t()]
    cell2node = Connectivity([4], collect(1:4))

    # Prepare boundary nodes
    bnd_names = ("F1", "F2", "F3", "F4")
    tag2name = Dict(tag => name for (tag, name) in enumerate(bnd_names))
    tag2nodes = Dict(tag => [faces2nodes(Tetra4_t)[tag]...] for tag in 1:length(bnd_names))
    tag2bfaces = Dict(tag => [tag] for tag in 1:length(bnd_names))

    return Mesh(
        nodes,
        celltypes,
        cell2node;
        bc_names = tag2name,
        bc_nodes = tag2nodes,
        bc_faces = tag2bfaces,
    )
end

function one_hexa_mesh(::Val{1}, xmin, xmax, ymin, ymax, zmin, zmax)
    nodes = [
        Node([xmin, ymin, zmin]),
        Node([xmax, ymin, zmin]),
        Node([xmax, ymax, zmin]),
        Node([xmin, ymax, zmin]),
        Node([xmin, ymin, zmax]),
        Node([xmax, ymin, zmax]),
        Node([xmax, ymax, zmax]),
        Node([xmin, ymax, zmax]),
    ]
    celltypes = [Hexa8_t()]
    cell2node = Connectivity([8], collect(1:8))
    return Mesh(nodes, celltypes, cell2node)
end

function one_hexa_mesh(::Val{2}, xmin, xmax, ymin, ymax, zmin, zmax)
    P1 = [xmin, ymin, zmin]
    P2 = [xmax, ymin, zmin]
    P3 = [xmax, ymax, zmin]
    P4 = [xmin, ymax, zmin]
    P5 = [xmin, ymin, zmax]
    P6 = [xmax, ymin, zmax]
    P7 = [xmax, ymax, zmax]
    P8 = [xmin, ymax, zmax]
    nodes = [
        Node(P1),
        Node(P2),
        Node(P3),
        Node(P4),
        Node(P5),
        Node(P6),
        Node(P7),
        Node(P8),
        Node(0.5 .* (P1 .+ P2)), # 9
        Node(0.5 .* (P2 .+ P3)), # 10
        Node(0.5 .* (P3 .+ P4)), # 11
        Node(0.5 .* (P4 .+ P1)), # 12
        Node(0.5 .* (P1 .+ P5)), # 13
        Node(0.5 .* (P2 .+ P6)), # 14
        Node(0.5 .* (P3 .+ P7)), # 15
        Node(0.5 .* (P4 .+ P8)), # 16
        Node(0.5 .* (P5 .+ P6)), # 17
        Node(0.5 .* (P6 .+ P7)), # 18
        Node(0.5 .* (P7 .+ P8)), # 19
        Node(0.5 .* (P8 .+ P5)), # 20
        Node(0.25 .* (P1 .+ P2 .+ P3 .+ P4)), # 21
        Node(0.25 .* (P1 .+ P2 .+ P6 .+ P5)), # 22
        Node(0.25 .* (P2 .+ P3 .+ P7 .+ P6)), # 23
        Node(0.25 .* (P3 .+ P4 .+ P8 .+ P7)), # 24
        Node(0.25 .* (P4 .+ P1 .+ P5 .+ P8)), # 25
        Node(0.25 .* (P5 .+ P6 .+ P7 .+ P8)), # 26
        Node(0.5 .* [xmin + xmax, ymin + ymax, zmin + zmax]), # 27
    ]
    celltypes = [Hexa27_t()]
    cell2node = Connectivity([27], collect(1:27))
    return Mesh(nodes, celltypes, cell2node)
end

function one_prism_mesh(::Val{1}, xmin, xmax, ymin, ymax, zmin, zmax)
    nodes = [
        Node([xmin, ymin, zmin]),
        Node([xmax, ymin, zmin]),
        Node([xmin, ymax, zmin]),
        Node([xmin, ymin, zmax]),
        Node([xmax, ymin, zmax]),
        Node([xmin, ymax, zmax]),
    ]
    celltypes = [Penta6_t()]
    cell2node = Connectivity([6], [1, 2, 3, 4, 5, 6])
    return Mesh(nodes, celltypes, cell2node)
end

"""
    circle_mesh(n; r = 1, order = 1)

Mesh a circle (in 2D) with `n` nodes on the circumference.
"""
function circle_mesh(n; radius = 1.0, order = 1)
    if (order == 1)
        # Nodes
        nodes = [
            Node(radius * [cos(θ), sin(θ)]) for
            θ in range(0, 2π; length = n + 1)[1:(end - 1)]
        ]

        # Cell type is constant
        celltypes = [Bar2_t() for ielt in 1:n]

        # Cell -> nodes connectivity
        cell2node = zeros(Int, 2 * n)
        for ielt in 1:(n - 1)
            cell2node[2 * ielt - 1] = ielt
            cell2node[2 * ielt]     = ielt + 1
        end
        cell2node[2 * n - 1] = n
        cell2node[2 * n]     = 1

        # Number of nodes of each cell : always 2
        cell2nnodes = 2 * ones(Int, n)

        # Mesh
        return Mesh(nodes, celltypes, Connectivity(cell2nnodes, cell2node))

    elseif (order == 2)
        # Nodes
        nodes = [
            Node(radius * [cos(θ), sin(θ)]) for
            θ in range(0, 2π; length = 2 * n + 1)[1:(end - 1)]
        ]

        # Cell type is constant
        celltypes = [Bar3_t() for ielt in 1:n]

        # Cell -> nodes connectivity
        cell2node = zeros(Int, 3 * n)
        i = 1 # init counter
        for ielt in 1:(n - 1)
            cell2node[3 * ielt - 2] = i
            cell2node[3 * ielt - 1] = i + 2
            cell2node[3 * ielt]     = i + 1
            i                       += 2
        end
        cell2node[3 * n - 2] = i
        cell2node[3 * n - 1] = 1
        cell2node[3 * n]     = i + 1

        # Number of nodes of each cell : always 2
        cell2nnodes = 3 * ones(Int, n)

        # Mesh
        return Mesh(nodes, celltypes, Connectivity(cell2nnodes, cell2node))
    else
        error("circle_mesh not implemented for order = ", order)
    end
end

"""
    scale(mesh, factor)

Scale the input mesh nodes coordinates by a given factor and return the resulted mesh. The `factor` can be a number
or a vector.

Usefull for debugging.
"""
scale(mesh::AbstractMesh, factor) = transform(mesh, x -> factor .* x)

scale!(mesh::AbstractMesh, factor) = transform!(mesh, x -> factor .* x)

"""
    transform(mesh::AbstractMesh, fun)

Transform the input mesh nodes coordinates by applying the given `fun` function and return the resulted mesh.

Usefull for debugging.
"""
function transform(mesh::AbstractMesh, fun)
    new_mesh = _duplicate_mesh(mesh)
    transform!(new_mesh, fun)
    return new_mesh
end

function transform!(mesh::AbstractMesh, fun)
    new_nodes = [Node(fun(n.x)) for n in get_nodes(mesh)]
    #mesh.nodes .= new_nodes # bmxam : I don't understand why this is not working
    set_nodes(mesh, new_nodes)
end

"""
    translate(mesh::AbstractMesh, t::AbstractVector)

Translate the input mesh with vector `t`.

Usefull for debugging.
"""
translate(mesh::AbstractMesh, t::AbstractVector) = transform(mesh, x -> x + t)
translate!(mesh::AbstractMesh, t::AbstractVector) = transform!(mesh, x -> x + t)

"""
    _duplicate_mesh(mesh::AbstractMesh)

Make an exact copy of the input mesh.
"""
function _duplicate_mesh(mesh::AbstractMesh)
    Mesh(
        get_nodes(mesh),
        cells(mesh),
        connectivities(mesh, :c2n).indices;
        bc_names = boundary_names(mesh),
        bc_nodes = boundary_nodes(mesh),
        bc_faces = boundary_faces(mesh),
    )
end
