otimes(x::AbstractVector, y::AbstractVector) = x * y'
otimes(x::AbstractVector) = otimes(x, x)

"""
    otimes(A,B)

Tensors product between second order tensors

# Implementation
A[i,j] ⊗ B[k,l] = C[i,j,k,l]
such as
C[i,j,k,l] = A[i,j] * B[k,l]
"""
function otimes(A::AbstractMatrix{T1}, B::AbstractMatrix{T2}) where {T1, T2}
    sA = size(A)
    sB = size(B)
    C = zeros(promote_type(T1, T2), sA..., sB...)
    C = [A[i, j] * B[k, l] for i in 1:sA[1], j in 1:sA[2], k in 1:sB[1], l in 1:sB[2]]
    return C
end

# otimes for static arrays
function otimes(
    A::SMatrix{I1, I2, T1, L1},
    B::SMatrix{I3, I4, T2, L2},
) where {I1, I2, I3, I4, T1, T2, L1, L2}
    M = (A[i, j] * B[k, l] for i in 1:I1, j in 1:I2, k in 1:I3, l in 1:I4)
    if I1 * I2 * I3 * I4 > 3^4
        # add collect to avoid stack overflow 
        return SArray{Tuple{I1, I2, I3, I4}}(collect(M))
    else
        return SArray{Tuple{I1, I2, I3, I4}}(M)
    end
end

const ⊗ = otimes

"""
Tensors double contraction between third order tensor and second order tensor

# Implementation
Only valid for a tensor of dimension 3 with a tensor of dimension 2 for now:
A[i,j,k] : B[l,m] = C[i]
such as
C[i] = A[i,j,k] * B[j,k] (Einstein sum)
"""
function dcontract(A::AbstractArray{T1, 3}, B::AbstractArray{T2, 2}) where {T1, T2}
    [sum(A[i, :, :] .* B) for i in 1:size(A)[1]]
end
dcontract(A::AbstractArray{T1, 2}, B::AbstractArray{T2, 3}) where {T1, T2} = dcontract(B, A)

"""
Tensors double contraction between second order tensors

# Implementation
A[i,j] : B[l,m] = c
such as
c = A[i,j] * B[i,j] (Einstein sum)
"""
dcontract(A::AbstractMatrix, B::AbstractMatrix) = sum(A .* B)

"""
Tensors double contraction between the identity tensor and a second order tensor

# Implementation
I : B = dot(I, B)
B : I = dot(B, I)
"""
dcontract(I::UniformScaling, B::AbstractMatrix) = dot(I, B)
dcontract(B::AbstractMatrix, I::UniformScaling) = dot(B, I)

"""
Tensors double contraction for third order tensors

# Implementation
A[i,j,k] : B[l,m,n] = C[i,l]
such as
C[i,l] = A[i,j,k] * B[l,j,k] (Einstein sum)
"""
function dcontract(A::AbstractArray{T1, 3}, B::AbstractArray{T2, 3}) where {T1, T2}
    sA = size(A)
    sB = size(B)
    C = zeros(sA[1], sB[1])
    for i in 1:sA[1]
        for j in 1:sB[1]
            C[i, j] = sum(A[i, :, :] .* B[j, :, :])
        end
    end
    return C
end

"""
    dcontract(A,B)

Tensors double contraction between fourth order tensor and second order tensor

# Implementation
A[i,j,k,l] : B[m,n] = C[i,j]
such as
C[i,j] = A[i,j,k,l] * B[k,l] (Einstein sum)
"""
function dcontract(A::AbstractArray{T1, 4}, B::AbstractArray{T2, 2}) where {T1, T2}
    sA = size(A)
    C = zeros(promote_type(eltype(A), eltype(B)), sA[1], sA[2])
    C = [sum(view(A, i, j, :, :) .* B) for i in 1:sA[1], j in 1:sA[2]]
    return C
end

# dcontract for static arrays
function dcontract(
    A::SArray{<:Tuple{I1, J, K}, T1, 3, L1},
    B::SMatrix{J, K, T2, L2},
) where {I1, J, K, T1, T2, L1, L2}
    return SVector{I1}(sum(A[i, :, :] .* B) for i in 1:I1)
end
function dcontract(
    B::SMatrix{J, K, T2, L2},
    A::SArray{<:Tuple{I1, J, K}, T1, 3, L1},
) where {I1, J, K, T1, T2, L1, L2}
    dcontract(A, B)
end

function dcontract(
    A::SArray{<:Tuple{I1, J, K}, T1, 3, L1},
    B::SArray{<:Tuple{I2, J, K}, T2, 3, L2},
) where {I1, J, K, I2, T1, T2, L1, L2}
    return SMatrix{I1, I2}(sum(A[i, :, :] .* B[j, :, :]) for i in 1:I1, j in 1:I2)
end

function dcontract(
    A::SArray{<:Tuple{I1, I2, K, L}, T1, 4, L1},
    B::SMatrix{K, L, T2, L2},
) where {I1, I2, K, L, T1, L1, T2, L2}
    return SMatrix{I1, I2}(sum(A[i, j, :, :] .* B) for i in 1:I1, j in 1:I2)
end

const ⊡ = dcontract

###############################################################
# Extend LazyOperators behaviors with newly defined operators
###############################################################

# GENERIC PART (it can be applied to all newly defined operators)
for f in (:otimes, :dcontract)
    # deal with `AbstractLazy`
    @eval ($f)(a::AbstractLazy, b::AbstractLazy) = LazyOperator($f, a, b)
    @eval ($f)(a::AbstractLazy, b) = ($f)(a, LazyWrap(b))
    @eval ($f)(a, b::AbstractLazy) = ($f)(LazyWrap(a), b)

    # deal with `MapOver`
    @eval ($f)(a::MapOver, b::MapOver) = LazyOperators.map_over($f, a, b)
    @eval ($f)(a::MapOver, b) = LazyOperators.map_over(Base.Fix2($f, b), a)
    @eval ($f)(a, b::MapOver) = LazyOperators.map_over(Base.Fix1($f, a), b)
end

# SPECIFIC PART : rules depend on how `NullOperator` acts with each operator.

# Both `otimes` and `dcontract` follow the same
# rule when they are applied to a `NullOperator`,
# which is a "absorbing element" in this case.
for f in (:otimes, :dcontract)
    @eval ($f)(a, ::NullOperator) = NullOperator()
    @eval ($f)(::NullOperator, b) = NullOperator()
    @eval ($f)(::NullOperator, ::NullOperator) = NullOperator()
    @eval ($f)(::AbstractLazy, ::NullOperator) = NullOperator()
    @eval ($f)(::NullOperator, ::AbstractLazy) = NullOperator()
end
