# How to... (FAQ)

## Build your own `LazyOperator`
Imagine that you want some kind of function (~operator) that has a different behavior depending on the cell (or face) it is applied to. The `PhysicalFunction` won't do the job since it is assumed that the provided function applies the same way in all the different cells. What you want is a `LazyOperator`. Here is how to build a custom one.

For the example, let's say that you want an operator whose action is to multiply `x`, the evaluated point, by the index of the cell surrounding `x`. Start importing some Bcube material and by declaring a type corresponding to this operator:
```julia
using Bcube
import Bcube: CellInfo, CellPoint, get_coords
struct DummyOperator <: Bcube.AbstractLazy end
```

Then, specify what happens when `Bcube` asks for the restriction of your operator in a given cell. This is done before applying it to any point. In most case, you don't want to do anything special, so just return the operator itself:
```julia
Bcube.materialize(op::DummyOperator, ::CellInfo) = op
```

Now, specify what to return when `Bcube` wants to apply this operator on a given point in a cell. As said earlier, we want it the return the point, multiplied by the cell index (but it could be anything you want):
```julia
function Bcube.materialize(
    ::DummyOperator,
    cPoint::CellPoint,
)
    x = get_coords(cPoint)
    cInfo = Bcube.get_cellinfo(cPoint)
    index = Bcube.cellindex(cInfo)
    return x * index
end
```

That's it! To see your operator in action, take a look at the related [section](@ref Evaluate-a-LazyOperator-on-a-specific-point).

In this short example, note that we restricted ourselves to `CellPoint` : the `DummyOperator` won't be applicable to a face. To do so, you have to specialize the materialization on a `Side` of a `FaceInfo` and on a `Side` of a `FacePoint`. Checkout the source code for `TangentialProjector` to see this in action. Besides, the `CellPoint` is parametrized by a `DomainStyle`, allowing to specify different behavior depending on if your operator is applied to a point in the `ReferenceDomain` or in the `PhysicalDomain`.

## Evaluate a `LazyOperator` on a specific point
Suppose that you have built a mesh and defined a `LazyOperator` on this mesh and you want, for debug purpose, evaluate this operator on a point of your choice. First, let's define our example operator:
```julia
using Bcube
mesh = circle_mesh(10)
op = Bcube.TangentialProjector()
```
Then, let's define the point where we want to evaluate this operator. For this, we need to create a so-called `CellPoint`. It's structure is quite basic : it needs the coordinates, the mesh cell owning these coordinates, and if the coordinates are given in the `ReferenceDomain` or in the `PhysicalDomain`. Here, we will select the first cell of the mesh, and choose the coordinates `[0.5]` (recall that we are in 1D, hence this vector of one component):
```julia
cInfo = Bcube.CellInfo(mesh, 1)
cPoint = Bcube.CellPoint([0.5], cInfo, Bcube.ReferenceDomain())
```
Now, they are always two steps to evaluate a `LazyOperator`. First we need to materialize it on a cell (or a face) and then to evaluate it on a cell-point (or face-point). The materialization on a cell does not necessarily triggers something, it depends on the operator. For instance, an analytic function will not have a specific behaviour depending on the cell; however a shape function will.
```julia
op_cell = Bcube.materialize(op, cInfo)
```
Finally, we can apply our operator on the cell point defined above and observe the result. It is also called a "materialization":
```julia
@show Bcube.materialize(op_cell, cPoint)
```
Note that before and after the materialization on a cell point, the operator can be displayed as a tree with
```julia
Bcube.show_lazy_operator(op)
Bcube.show_lazy_operator(op_cell)
```

## Get the coordinates of Lagrange dofs
For a **Lagrange** "uniform" function space, the dofs corresponds to vertices. The following `lagrange_dof_to_coords` function returns a matrix : each line contains the coordinates of the dof corresponding to the line number.
```julia
function lagrange_dof_to_coords(mesh, degree)
    U = TrialFESpace(FunctionSpace(:Lagrange, degree), mesh)
    coords = map(1:Bcube.spacedim(mesh)) do i
        f = PhysicalFunction(x -> x[i])
        u = FEFunction(U)
        projection_l2!(u, f, mesh)
        return get_dof_values(u)
    end
    return hcat(coords...)
end
```
For instance:
```julia
using Bcube
mesh = rectangle_mesh(2, 3; xmin = 1, xmax = 2, ymin = 3, ymax = 5)
coords = lagrange_dof_to_coords(mesh, 1)
@show coords[2] # coordinates of dof '2' in the global numbering
```

## Comparing manually the benchmarks with `main`

Let's say you want to compare the performance of your current branch (named "target" hereafter) with the `main` branch (named "baseline" hereafter).

Open from `Bcube.jl/` a REPL and type:

```julia
pkg> activate --temp
pkg> add BenchmarkTools PkgBenchmark StaticArrays WriteVTK UnPack
pkg> dev .
using PkgBenchmark
import Bcube
benchmarkpkg(Bcube, BenchmarkConfig(; env = Dict("JULIA_NUM_THREADS" => "1")); resultfile = joinpath(@__DIR__, "result-target.json"))
```

This will create a `result-target.json` in the current directory.

Then checkout the `main` branch. Start a fresh REPL and type (almost the same):

```julia
pkg> activate --temp
pkg> add BenchmarkTools PkgBenchmark StaticArrays WriteVTK UnPack
pkg> dev .
using PkgBenchmark
import Bcube
benchmarkpkg(Bcube, BenchmarkConfig(; env = Dict("JULIA_NUM_THREADS" => "1")); resultfile = joinpath(@__DIR__, "result-baseline.json"))
```

This will create a `result-baseline.json` in the current directory.

You can now "compare" the two files by running (watch-out for the order):

```julia
target = PkgBenchmark.readresults("result-target.json")
baseline = PkgBenchmark.readresults("result-baseline.json")
judgement = judge(target, baseline)
export_markdown("judgement.md", judgement)
```

This will create the markdown file `judgement.md` with the results.

For more details, once you've built the `judgement` object, you can also type the following code from `https://github.com/tkf/BenchmarkCI.jl`:

```julia
open("detailed-judgement.md", "w") do io
    println(io, "# Judge result")
    export_markdown(io, judgement)
    println(io)
    println(io)
    println(io, "---")
    println(io, "# Target result")
    export_markdown(io, PkgBenchmark.target_result(judgement))
    println(io)
    println(io)
    println(io, "---")
    println(io, "# Baseline result")
    export_markdown(io, PkgBenchmark.baseline_result(judgement))
    println(io)
    println(io)
    println(io, "---")
end
```

## Run the benchmark manually

Let's say you want to run the benchmarks locally (without comparing with `main`)

Open from `Bcube.jl/` a REPL and type:

```julia
pkg> activate --temp
pkg> add BenchmarkTools PkgBenchmark StaticArrays WriteVTK UnPack
pkg> dev .
using PkgBenchmark
import Bcube
results = benchmarkpkg(Bcube, BenchmarkConfig(; env = Dict("JULIA_NUM_THREADS" => "1")); resultfile = joinpath(@__DIR__, "result.json"))
export_markdown("results.md", results)
```

This will create the markdown file `results.md` with the results.
