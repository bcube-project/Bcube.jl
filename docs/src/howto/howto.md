# How to

To be completed to answer common user questions.

## Evaluate a `LazyOperator` on a specific point
Suppose that you have built a mesh and defined a `LazyOperator` on this mesh and you want, for debug purpose, evaluate this operator on a point of your choice. First, let's define our example operator:
```julia
mesh = circle_mesh(10)
op = Bcube.TangentialOperator()
```
Then, let's define the point where we want to evaluate this operator. For this, we need to create a so-called `CellPoint`. It's structure is quite basic : it needs the coordinates, the mesh cell owning these coordinates, and if the coordinates are given in the `ReferenceDomain` or in the `PhysicalDomain`. Here, we will select the first cell of the mesh, and choose the coordinates `[0.5]` (recall that we are in 1D, hence this vector of one component):
```julia
cInfo = Bcube.CellInfo(mesh, 1)
cPoint = Bcube.CellPoint(SA[0.5], cInfo, Bcube.ReferenceDomain())
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

## Comparing manually the benchmarks with `main`

Let's say you want to compare the performance of your current branch (named "target" hereafter) with the `main` branch (named "baseline" hereafter).

Open from `Bcube.jl/` a REPL and type:

```julia
pkg> activate --temp
pkg> add PkgBenchmark BenchmarkTools
pkg> dev .
using PkgBenchmark
import Bcube
benchmarkpkg(Bcube, BenchmarkConfig(; env = Dict("JULIA_NUM_THREADS" => "1")); resultfile = joinpath(@__DIR__, "result-target.json"))
```

This will create a `result-target.json` in the current directory.

Then checkout the `main` branch. Start a fresh REPL and type (almost the same):

```julia
pkg> activate --temp
pkg> add PkgBenchmark BenchmarkTools
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
pkg> add PkgBenchmark
pkg> dev .
using PkgBenchmark
import Bcube
results = benchmarkpkg(Bcube, BenchmarkConfig(; env = Dict("JULIA_NUM_THREADS" => "1")); resultfile = joinpath(@__DIR__, "result.json"))
export_markdown("results.md", results)
```

This will create the markdown file `results.md` with the results.
