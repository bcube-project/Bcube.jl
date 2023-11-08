module BenchMesh

using BenchmarkTools
using Bcube

suite = BenchmarkGroup()

suite["todo"] = @benchmarkable a = 1

end  # module
BenchMesh.suite
