abstract type AbstractAffineTransformation end
(::AbstractAffineTransformation)(x) = error("not implemented")
Base.inv(::AbstractAffineTransformation) = error("not implemented")

struct Translation{T} <: AbstractAffineTransformation
    vec::T
end
(t::Translation)(x) = x .+ t.vec
Base.inv(t::Translation) = Translation(-t.vec)

struct Rotation{T} <: AbstractAffineTransformation
    A::T
end

"""
    Rotation(u::AbstractVector, θ::Number)

Return a `Rotation` operator build from a given
rotation axis `u` and an angle of rotation `θ`.

# Example :
```julia
julia> rot = Rotation([0,0,1], π/4);
julia> x = [3.0,0,0];
julia> x_rot_ref = 3√(2)/2 .*[1,1,0] # expected result
julia> all(rot(x) .≈ x_rot_ref)
true
```
"""
function Rotation(u::AbstractVector, θ::Number)
    Rotation((normalize(u), sincos(θ)))
end

function (r::Rotation)(x)
    u, (sinθ, cosθ) = r.A
    rx = cosθ * x + (1 - cosθ) * (u ⋅ x) * u + sinθ * cross(u, x)
    return rx
end
