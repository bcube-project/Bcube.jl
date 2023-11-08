# Alias to script directory : helps running this file from anywhere
const dir = string(@__DIR__, "/../") # Bcube dir

# Run examples
include(dir * "example/covo.jl")
include(dir * "example/euler_naca_steady.jl")
include(dir * "example/linear_elasticity.jl")
