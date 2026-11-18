using BenchmarkTools, Pkg
using Bcube
suite = include("bench_dirichlet.jl")
println("Julia: ", VERSION, "; threads: ", Threads.nthreads())
Pkg.status()
for U in (
    BenchDirichlet.velocity,
    Bcube.MultiFESpace(BenchDirichlet.velocity, BenchDirichlet.pressure),
)
    println("ndofs: ", Bcube.get_ndofs(U))
end
results = run(suite; seconds = 1, samples = 20)
println("| configuration | operation | median ns | bytes | allocations |")
println("|---|---|---:|---:|---:|")
for config in sort(collect(keys(results))),
    operation in sort(collect(keys(results[config])))

    estimate = median(results[config][operation])
    println(
        "| ",
        config,
        " | ",
        operation,
        " | ",
        estimate.time,
        " | ",
        estimate.memory,
        " | ",
        estimate.allocs,
        " |",
    )
end
