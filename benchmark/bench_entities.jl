module BenchEntities

using StaticArrays
using BenchmarkTools
using Bcube

suite = BenchmarkGroup()
tri = Tri3_t()
ind = @SVector [10, 20, 30]
suite["nnodes"] = @benchmarkable f2n_from_c2n($tri, $ind)
suite["nodes"] = @benchmarkable nodes($tri)
suite["nedges"] = @benchmarkable nedges($tri)
suite["edges2nodes"] = @benchmarkable edges2nodes($tri)

end  # module
BenchEntities.suite
