module BenchEntities

using StaticArrays
using BenchmarkTools
using Bcube

suite = BenchmarkGroup()
tri = Bcube.Tri3_t()
ind = @SVector [10, 20, 30]
suite["nnodes"] = @benchmarkable Bcube.f2n_from_c2n($tri, $ind)
suite["nodes"] = @benchmarkable Bcube.nodes($tri)
suite["nedges"] = @benchmarkable Bcube.nedges($tri)
suite["edges2nodes"] = @benchmarkable Bcube.edges2nodes($tri)

end  # module
BenchEntities.suite
