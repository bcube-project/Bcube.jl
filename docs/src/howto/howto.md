# How to

To be completed to answer common user questions.

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
