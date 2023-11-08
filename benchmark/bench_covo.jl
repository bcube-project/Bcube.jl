module BenchCovo_Quad

using StaticArrays
using BenchmarkTools

ENV["BenchmarkMode"] = "true"

include("../example/covo.jl")
include("./driver_bench_covo.jl")

suite = run_covo()

end  # module

module BenchCovo_TriQuad

using StaticArrays
using BenchmarkTools

ENV["BenchmarkMode"] = "true"
ENV["MeshConfig"] = "triquad"

include("../example/covo.jl")
include("./driver_bench_covo.jl")

suite = run_covo()

end  # module

suiteCovo = BenchmarkGroup()
suiteCovo["Quad"] = BenchCovo_Quad.suite
suiteCovo["TriQuad"] = BenchCovo_TriQuad.suite

suiteCovo
