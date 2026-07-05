# Certification that a function does not mutate a given argument, by running it on a write-
# recording wrapper array: every `setindex!` sets a flag, raw-memory escape hatches (`pointer`,
# `unsafe_convert`) throw, and operations the wrapper does not support (`resize!`, `push!`,
# BLAS-specific fast paths) raise `MethodError` -- any of which abandons the certificate.
# `hasbranching` additionally proves the probed path is the only path.

mutable struct _WriteFlag
    hit::Bool
end

struct _WriteProbe{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::A
    flag::_WriteFlag
end

Base.size(w::_WriteProbe) = size(w.data)
Base.IndexStyle(::Type{<:_WriteProbe{T, N, A}}) where {T, N, A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(w::_WriteProbe, i::Int...) = getindex(w.data, i...)
Base.@propagate_inbounds function Base.setindex!(w::_WriteProbe, v, i::Int...)
    w.flag.hit = true
    return setindex!(w.data, v, i...)
end
Base.similar(w::_WriteProbe, ::Type{S}, dims::Dims) where {S} = similar(w.data, S, dims)
Base.pointer(::_WriteProbe, i::Integer...) = throw(ArgumentError("_WriteProbe blocks raw memory access"))
Base.unsafe_convert(::Type{Ptr{T}}, ::_WriteProbe) where {T} =
    throw(ArgumentError("_WriteProbe blocks raw memory access"))

"""
    hasmutation(f, x...; arg = :) -> Bool

Conservatively check whether calling `f(x...)` may mutate the argument(s) selected by `arg`
(an index, collection of indices, or `:` for all mutable arguments): `false` *certifies* that
the selected arguments are not written through their argument position -- `f` is run on
write-recording wrappers with raw-memory access blocked, and [`hasbranching`](@ref) proves the
probed path is the only path. `true` means mutation was observed or could not be ruled out.

For an in-place SciML right-hand side this answers e.g. whether the state is written:
`hasmutation(f!, du, u, p, t; arg = 2)`.

The certificate covers writes through the selected argument reference and rejects (returns
`true` for) arrays with non-`isbits` elements, whose interior could be mutated without a
recorded write. Aliases of the selected array reachable through the *other* arguments are
detected one structural level deep (arguments themselves and elements of tuples or arrays of
arrays); deeper aliasing is outside the certificate.
"""
function hasmutation(f, x...; arg = (:))
    idx = _wrt_indices(arg, length(x))
    all(i -> 1 <= i <= length(x), idx) || throw(ArgumentError("arg index out of range"))
    hasbranching(f, x...) && return true
    flags = _WriteFlag[]
    probed = ntuple(length(x)) do i
        xi = x[i]
        if i in idx && xi isa AbstractArray
            isbitstype(eltype(xi)) || return nothing
            _shallow_alias(xi, x, i) && return nothing
            flag = _WriteFlag(false)
            push!(flags, flag)
            _WriteProbe(copy(xi), flag)
        else
            xi
        end
    end
    any(isnothing, probed) && return true
    ok = try
        f(probed...)
        true
    catch
        false
    end
    return !(ok && !any(flag -> flag.hit, flags))
end

function _shallow_alias(target, args, self)
    for (j, a) in enumerate(args)
        j == self && continue
        a === target && return true
        if a isa Union{Tuple, AbstractArray}
            for el in a
                el === target && return true
            end
        end
    end
    return false
end
