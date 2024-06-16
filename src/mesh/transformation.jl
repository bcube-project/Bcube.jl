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

function Rotation(u::AbstractVector, θ::Number)
    Rotation((normalize(u), sincos(θ)))
end

function (r::Rotation)(x)
    u, (sinθ, cosθ) = r.A
    rx = cosθ * x + (1 - cosθ) * (u ⋅ x) * u + sinθ * cross(u, x)
    return rx
end