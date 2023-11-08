abstract type AbstractAffineTransformation end
(::AbstractAffineTransformation)(x) = error("not implemented")
Base.inv(::AbstractAffineTransformation) = error("not implemented")

struct Translation{T} <: AbstractAffineTransformation
    vec::T
end
(t::Translation)(x) = x .+ t.vec
Base.inv(t::Translation) = Translation(-t.vec)
