"""
    DofMap{D, T, L}

Container of the values of all the shape functions (dofs) of a FESpace, on which the
operators are applied dof by dof, lazily and with fusion (see `dof_broadcasted`).

- `D` is the tuple of the (static) axes: `(n,)` for the `n` dofs of a linear form,
  `(1, n)` for the trial functions and `(m, 1)` for the test functions of a bilinear form,
  `(m, n)` for their combination (the broadcast rules of `Base` combine the axes).
- `T` is the type of the values, which must be the same for all the dofs.
- `L = prod(D)` is the number of values, stored column-major in `data`.

The trial and test functions of a bilinear form share the same tuple of values: only the
axes differ, the values are not duplicated.

# Dev notes
The operators on a `DofMap` return a `Broadcasted{DofMapStyle}` instead of computing a new
`DofMap`: the whole integrand is fused and evaluated in a single loop by `dof_copy`. This
loop has a dynamic index, so the compilation time does not grow with the number of dofs
(unlike an unrolled evaluation such as `map_over_leaves`).
"""
struct DofMap{D, T, L}
    data::NTuple{L, T}

    function DofMap{D, T, L}(data::NTuple{L, T}) where {D, T, L}
        D isa Tuple{Vararg{Int}} && length(D) <= 2 && all(>=(0), D) ||
            throw(ArgumentError("DofMap: invalid axes $D"))
        prod(D) == L || throw(DimensionMismatch("DofMap: axes $D for $L values"))
        new{D, T, L}(data)
    end
end

DofMap{D}(data::NTuple{L, T}) where {D, T, L} = DofMap{D, T, L}(data)
DofMap(data::NTuple{L, T}) where {L, T} = DofMap{(L,)}(data)

Base.eltype(::Type{<:DofMap{D, T}}) where {D, T} = T
unwrap(a::DofMap) = a.data

"""
    BlockMap{A <: Tuple}

Container of a few blocks of possibly different types, on which the operators are applied
block by block: the sides of a face (for instance `(nn, pn, np, pp)` for a bilinear form)
or the FESpaces of a `MultiFESpace`. A block is usually a `DofMap` (or a lazy
`Broadcasted{DofMapStyle}`), or a `NullOperator`.

A `BlockMap` whose blocks are all `NullOperator`s is collapsed into a `NullOperator` by the
operators.
"""
struct BlockMap{A <: Tuple}
    blocks::A
end

BlockMap(blocks::Vararg{Any, N}) where {N} = BlockMap(blocks)

unwrap(a::BlockMap) = a.blocks

_blockmap(blocks::Tuple) = BlockMap(blocks)
_blockmap(::Tuple{NullOperator, Vararg{NullOperator}}) = NullOperator()

function _map_blocks(
    f::F,
    a::BlockMap{<:NTuple{N, Any}},
    b::BlockMap{<:NTuple{N, Any}},
) where {F, N}
    _blockmap(map(f, a.blocks, b.blocks))
end
function _map_blocks(f, a::BlockMap, b::BlockMap)
    throw(DimensionMismatch("BlockMaps of different lengths"))
end

"""
    DofMapStyle <: Base.Broadcast.BroadcastStyle

Style of the lazy `Broadcasted` built by the operators on `DofMap`s.
"""
struct DofMapStyle <: Base.Broadcast.BroadcastStyle end

const DofBroadcasted = Base.Broadcast.Broadcasted{DofMapStyle}
const DofLike = Union{DofMap, DofBroadcasted}

"""
    FixedArg{T}

Wrap an argument which is not a `DofMap` in a `Broadcasted{DofMapStyle}`: it is passed
unchanged to each dof. It prevents the broadcasting inside the argument (for instance an
`SVector`). `Base.RefValue` is not used because it is mutable: it escapes in `dof_copy` and
allocates.
"""
struct FixedArg{T}
    x::T
end
Base.eltype(::Type{FixedArg{T}}) where {T} = T

_dof_wrap(x::DofLike) = x
_dof_wrap(x) = FixedArg(x)

"""
    dof_broadcasted(f, args...)

Build the lazy `Broadcasted{DofMapStyle}` applying `f` dof by dof on `args`. The arguments
which are not `DofMap`s (or lazy `Broadcasted{DofMapStyle}`) are passed unchanged to each
call of `f`.
"""
@inline function dof_broadcasted(f::F, args::Vararg{Any, N}) where {F, N}
    Base.Broadcast.Broadcasted{DofMapStyle}(f, map(_dof_wrap, args))
end

###############################################################
# Static axes
###############################################################

function _combine_dof_axes(A::Tuple, B::Tuple)
    L = max(length(A), length(B))
    a = (A..., ntuple(_ -> 1, L - length(A))...)
    b = (B..., ntuple(_ -> 1, L - length(B))...)
    ntuple(L) do k
        if a[k] == b[k] || b[k] == 1
            a[k]
        elseif a[k] == 1
            b[k]
        else
            throw(DimensionMismatch("DofMap: incompatible axes $A and $B"))
        end
    end
end

@generated function _combine_dof_axes(::Val{A}, ::Val{B}) where {A, B}
    D = try
        _combine_dof_axes(A, B)
    catch e
        e isa DimensionMismatch || rethrow()
        return :(throw($e))
    end
    return :(Val($D))
end

"""
    dof_axes(a)

Return the static axes of `a` as a `Val`: `Val(D)` for a `DofMap{D}`, the combination of the
axes of the arguments for a `Broadcasted{DofMapStyle}`, `Val(())` for a `FixedArg`.
"""
@inline dof_axes(::DofMap{D}) where {D} = Val(D)
@inline dof_axes(::FixedArg) = Val(())
@inline dof_axes(bc::DofBroadcasted) = _dof_axes(bc.args)
@inline _dof_axes(args::Tuple{Any}) = dof_axes(first(args))
@inline function _dof_axes(args::Tuple)
    _combine_dof_axes(dof_axes(first(args)), _dof_axes(Base.tail(args)))
end

_size2(D::Tuple{}) = (1, 1)
_size2(D::Tuple{Int}) = (D[1], 1)
_size2(D::Tuple{Int, Int}) = D

###############################################################
# Fused evaluation
###############################################################

@inline function _dof_getindex(a::DofMap{D}, i, j) where {D}
    m, n = _size2(D)
    k = (m == 1 ? 1 : i) + (n == 1 ? 0 : (j - 1) * m)
    @inbounds a.data[k]
end
@inline _dof_getindex(a::FixedArg, i, j) = a.x
@inline _dof_getindex(bc::DofBroadcasted, i, j) = bc.f(_dof_getindex_args(bc.args, i, j)...)
@inline _dof_getindex_args(::Tuple{}, i, j) = ()
@inline function _dof_getindex_args(args::Tuple, i, j)
    (_dof_getindex(first(args), i, j), _dof_getindex_args(Base.tail(args), i, j)...)
end

"""
Minimal mutable buffer of `L` values of the `isbits` type `T`, filled by `dof_copy`. It
does not escape `dof_copy`, so the compiler does not allocate it (as for a `MVector`,
but without depending on StaticArrays).
"""
mutable struct _DofBuffer{L, T}
    data::NTuple{L, T}
    _DofBuffer{L, T}() where {L, T} = new{L, T}()
end

@inline function Base.setindex!(b::_DofBuffer{L, T}, x, i::Int) where {L, T}
    GC.@preserve b begin
        ptr = Base.unsafe_convert(Ptr{T}, pointer_from_objref(b))
        unsafe_store!(ptr, convert(T, x), i)
    end
    return b
end

"""
    dof_copy(a)

Evaluate the lazy `Broadcasted{DofMapStyle}` `a` and return a `DofMap`, or a `NullOperator`
if `f` returns `NullOperator`s. If `f` returns `Tuple`s, a `Tuple` of `DofMap`s (or
`NullOperator`s) is returned, one per component. A `BlockMap` is evaluated block by block
(and collapsed into a `NullOperator` if all its blocks are). If all its blocks give `Tuple`s
of the same length, a `Tuple` of `BlockMap`s is returned, as if the operators of `f` were
applied on the `BlockMap`. Other arguments are returned unchanged.

# Dev notes
The values are computed by a loop with a dynamic index into a `_DofBuffer`, so that the
compilation time does not depend on the number of dofs. If the type of the values is not a
concrete `isbits` type (for instance if the inference of `f` fails), the values are computed
by an unrolled `ntuple` instead, whose compilation time grows with the number of dofs.
"""
dof_copy(a) = a
dof_copy(a::BlockMap) = _blockmap_from_copies(map(dof_copy, a.blocks))
@inline function dof_copy(bc::DofBroadcasted)
    T = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    _dof_copy(T, dof_axes(bc), bc)
end

_dof_copy(::Type{T}, ::Val{D}, bc) where {T, D} = _dof_copy_values(T, Val(D), bc)
_dof_copy(::Type{NullOperator}, ::Val, bc) = NullOperator()
function _dof_copy(::Type{T}, ::Val{D}, bc) where {T <: Tuple, D}
    a = _dof_copy_values(T, Val(D), bc)
    isconcretetype(T) || return a
    return _split_components(a, Val(fieldcount(T)))
end

function _dof_copy_values(::Type{T}, ::Val{D}, bc) where {T, D}
    m, n = _size2(D)
    L = m * n
    if isbitstype(T)
        out = _DofBuffer{L, T}()
        for j in 1:n, i in 1:m
            out[i + (j - 1) * m] = _dof_getindex(bc, i, j)
        end
        return DofMap{D, T, L}(out.data)
    else
        data = ntuple(k -> _dof_getindex(bc, (k - 1) % m + 1, (k - 1) ÷ m + 1), Val(L))
        return DofMap{D, T, L}(data)
    end
end

struct _Component{k} end
@inline (::_Component{k})(x) where {k} = getfield(x, k)

"Split a `DofMap` of `Tuple`s of length `K` into a `Tuple` of `K` `DofMap`s."
@generated function _split_components(a::DofMap, ::Val{K}) where {K}
    exprs = [:(dof_copy(dof_broadcasted(_Component{$k}(), a))) for k in 1:K]
    return :(($(exprs...),))
end

_blockmap_from_copies(blocks::Tuple) = _blockmap(blocks)
"Blocks that are all `Tuple`s of length `K`: return a `Tuple` of `K` `BlockMap`s."
@generated function _blockmap_from_copies(
    blocks::Tuple{Tuple{Vararg{Any, K}}, Vararg{Tuple{Vararg{Any, K}}}},
) where {K}
    exprs = [:(_blockmap_from_copies(map(_Component{$k}(), blocks))) for k in 1:K]
    return :(($(exprs...),))
end

Base.copy(bc::DofBroadcasted) = dof_copy(bc)
Base.Broadcast.instantiate(bc::DofBroadcasted) = bc

###############################################################
# Operators
###############################################################

"""
    dofmap_binary_operator_rules(f)

Return the `Expr` defining the binary operator `f` on `DofMap`s and `BlockMap`s (except the
rules with `NullOperator`, which depend on `f`). The `Expr` must be evaluated in the module
owning `f`.
"""
function dofmap_binary_operator_rules(f)
    quote
        ($f)(a::$DofLike, b::$DofLike) = $dof_broadcasted($f, a, b)
        ($f)(a::$DofLike, b) = $dof_broadcasted($f, a, b)
        ($f)(a, b::$DofLike) = $dof_broadcasted($f, a, b)

        ($f)(a::$BlockMap, b::$BlockMap) = $_map_blocks($f, a, b)
        ($f)(a::$BlockMap, b) = $_blockmap(map(Base.Fix2($f, b), a.blocks))
        ($f)(a, b::$BlockMap) = $_blockmap(map(Base.Fix1($f, a), b.blocks))
        # the blocks are the outer level
        ($f)(a::$BlockMap, b::$DofLike) = $_blockmap(map(Base.Fix2($f, b), a.blocks))
        ($f)(a::$DofLike, b::$BlockMap) = $_blockmap(map(Base.Fix1($f, a), b.blocks))

        # remove ambiguities with `AbstractLazy` (lazy operators win)
        ($f)(a::$AbstractLazy, b::$DofLike) = ($f)(a, $LazyWrap(b))
        ($f)(a::$DofLike, b::$AbstractLazy) = ($f)($LazyWrap(a), b)
        ($f)(a::$AbstractLazy, b::$BlockMap) = ($f)(a, $LazyWrap(b))
        ($f)(a::$BlockMap, b::$AbstractLazy) = ($f)($LazyWrap(a), b)
    end
end

"""
    dofmap_absorbing_null_rules(f)

Return the `Expr` defining `f(a, ::NullOperator) = f(::NullOperator, b) = NullOperator()`
for `DofMap`s and `BlockMap`s (`*`, `dot`, `otimes`, …).
"""
function dofmap_absorbing_null_rules(f)
    quote
        ($f)(::$DofLike, ::$NullOperator) = $NullOperator()
        ($f)(::$NullOperator, ::$DofLike) = $NullOperator()
        ($f)(::$BlockMap, ::$NullOperator) = $NullOperator()
        ($f)(::$NullOperator, ::$BlockMap) = $NullOperator()
    end
end

for f in LazyBinaryOp
    eval(dofmap_binary_operator_rules(f))
end

for f in (:*, :dot)
    eval(dofmap_absorbing_null_rules(f))
end

for T in (DofLike, BlockMap)
    @eval begin
        Base.:+(a::$T, ::NullOperator) = a
        Base.:+(::NullOperator, b::$T) = b
        Base.:-(a::$T, ::NullOperator) = a
        Base.:-(::NullOperator, b::$T) = -b
        Base.:/(::$T, ::NullOperator) = error("Division by a NullOperator is not allowed.")
        Base.:/(::NullOperator, ::$T) = NullOperator()
        Base.:^(::$T, ::NullOperator) = error("Undefined")
        Base.:^(::NullOperator, ::$T) = NullOperator()
        Base.max(a::$T, ::NullOperator) = a
        Base.max(::NullOperator, b::$T) = b
        Base.min(a::$T, ::NullOperator) = a
        Base.min(::NullOperator, b::$T) = b
    end
end

for f in LazyUnaryOp
    @eval ($f)(a::DofLike) = dof_broadcasted($f, a)
    @eval ($f)(a::BlockMap) = _blockmap(map($f, a.blocks))
end

###############################################################
# Composition
###############################################################

"""
    dofwise(f, args...)

Apply `f` dof by dof (lazily) on the arguments: block by block on the `BlockMap`s, then
dof by dof on the `DofMap`s (see `dof_broadcasted`), the other arguments being passed
unchanged to each call of `f`. Without `BlockMap` nor `DofMap` argument, `f(args...)` is
returned. It is used to materialize a composition `f ∘ (a, b, ...)`.

All the `BlockMap` arguments must have the same number of blocks.
"""
@generated function dofwise(f::F, args::Vararg{Any, N}) where {F, N}
    ib = findfirst(T -> T <: BlockMap, args)
    if ib !== nothing
        K = fieldcount(fieldtype(args[ib], :blocks))
        if any(T -> T <: BlockMap && fieldcount(fieldtype(T, :blocks)) != K, args)
            return :(throw(DimensionMismatch("dofwise: BlockMaps of different lengths")))
        end
        blocks = map(1:K) do k
            _args = [args[j] <: BlockMap ? :(args[$j].blocks[$k]) : :(args[$j]) for j in 1:N]
            :(dofwise(f, $(_args...)))
        end
        return :(_blockmap(($(blocks...),)))
    elseif any(T -> T <: DofLike, args)
        return :(dof_broadcasted(f, args...))
    else
        return :(f(args...))
    end
end
