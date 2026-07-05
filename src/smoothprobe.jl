# Certification of smoothness by abstract interpretation: the tracked arguments are seeded with
# `SmoothProbe` tracer numbers, every C^∞-on-its-open-domain primitive propagates the tracer, and
# every non-smooth primitive (`abs`, `sign`, rounding, `mod`, and -- via the throwing comparisons
# -- `max`/`min` and all value-dependent branching) aborts the trace. `hasbranching` additionally
# proves the traced path is the only path. The certificate is therefore: `f` is a composition of
# real-analytic primitives on the interior of its natural domain.

struct SmoothProbe <: Real end

struct SmoothProbeError <: Exception
    op::Symbol
end
function Base.showerror(io::IO, e::SmoothProbeError)
    return print(
        io, "SmoothProbeError: `", e.op, "` is not smooth (or needs the value of a traced ",
        "number); the smoothness certificate is abandoned."
    )
end

Base.promote_rule(::Type{SmoothProbe}, ::Type{<:Real}) = SmoothProbe
Base.convert(::Type{SmoothProbe}, x::SmoothProbe) = x
Base.convert(::Type{SmoothProbe}, x::Real) = SmoothProbe()
SmoothProbe(::SmoothProbe) = SmoothProbe()
SmoothProbe(::Base.TwicePrecision) = SmoothProbe()
SmoothProbe(::Complex) = SmoothProbe()
SmoothProbe(::AbstractChar) = SmoothProbe()
Base.zero(::Type{SmoothProbe}) = SmoothProbe()
Base.zero(::SmoothProbe) = SmoothProbe()
Base.one(::Type{SmoothProbe}) = SmoothProbe()
Base.one(::SmoothProbe) = SmoothProbe()
Base.oneunit(::Type{SmoothProbe}) = SmoothProbe()
Base.float(x::SmoothProbe) = x
Base.widen(::Type{SmoothProbe}) = SmoothProbe

for fn in (:+, :-, :*, :/, :^)
    @eval Base.$fn(::SmoothProbe, ::SmoothProbe) = SmoothProbe()
end
Base.:\(::SmoothProbe, ::SmoothProbe) = SmoothProbe()
# Not smooth on the interior of their domains: `hypot` has a kink at the origin, two-argument
# `atan` jumps across its branch cut, `mod`/`rem` are staircases.
for fn in (:atan, :hypot, :mod, :rem)
    @eval Base.$fn(::SmoothProbe, ::SmoothProbe) = throw(SmoothProbeError($(QuoteNode(fn))))
end
Base.:+(x::SmoothProbe) = x
Base.:-(x::SmoothProbe) = x
Base.inv(::SmoothProbe) = SmoothProbe()
Base.muladd(::SmoothProbe, ::SmoothProbe, ::SmoothProbe) = SmoothProbe()
Base.fma(::SmoothProbe, ::SmoothProbe, ::SmoothProbe) = SmoothProbe()
Base.abs2(::SmoothProbe) = SmoothProbe()
Base.conj(x::SmoothProbe) = x
Base.real(x::SmoothProbe) = x
Base.:^(::SmoothProbe, n::Integer) = SmoothProbe()

# C^∞ on the interior of their natural domains.
for fn in (
        :sqrt, :exp, :exp2, :exp10, :expm1, :log, :log2, :log10, :log1p,
        :sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh, :asinh, :acosh,
        :atanh, :sinpi, :cospi, :sec, :csc, :cot,
    )
    @eval Base.$fn(::SmoothProbe) = SmoothProbe()
end

# Not smooth: kinks, jumps, and staircase functions.
for fn in (:abs, :sign, :cbrt)
    @eval Base.$fn(::SmoothProbe) = throw(SmoothProbeError($(QuoteNode(fn))))
end
for fn in (:floor, :ceil, :trunc, :round)
    @eval Base.$fn(::SmoothProbe) = throw(SmoothProbeError($(QuoteNode(fn))))
end

# Value-needing predicates abort the trace, exactly as for the degree tracer.
for fn in (:isless, :(==), :<, :(<=), :isequal)
    @eval Base.$fn(::SmoothProbe, ::SmoothProbe) = throw(SmoothProbeError($(QuoteNode(fn))))
end
for fn in (:isnan, :isinf, :isfinite, :iszero, :isone, :signbit, :isinteger)
    @eval Base.$fn(::SmoothProbe) = throw(SmoothProbeError($(QuoteNode(fn))))
end

_smooth_seed(x::Real) = SmoothProbe()
_smooth_seed(x::AbstractArray{<:Real}) = map(_ -> SmoothProbe(), x)
_smooth_seed(x) = x   # non-numeric arguments pass through and stay fixed

"""
    issmooth(f, x...; wrt = :) -> Bool

Attempt to *prove* that `f` is smooth (infinitely differentiable) in the arguments selected by
`wrt` (default: all of them), on the interior of its natural domain, holding the remaining
arguments fixed at the values given. `true` certifies that `f` is a composition of
real-analytic primitives along its (only) execution path: the trace aborts on `abs`, `sign`,
rounding, `mod`, `max`/`min`, and any value-dependent comparison, and [`hasbranching`](@ref)
proves the traced path is the only path. `false` means *not proven*.

Domain boundaries are not modeled: `sqrt` and `log` count as smooth because they are smooth on
the interior of their domains, even though they are singular at its edge.

```jldoctest
julia> using FunctionProperties

julia> issmooth((u, p, t) -> exp(u[1]) * sin(t), [1.0], nothing, 0.0)
true

julia> issmooth(u -> abs(u[1]), [1.0])
false
```
"""
function issmooth(f, x...; wrt = (:))
    idx = _wrt_indices(wrt, length(x))
    all(i -> 1 <= i <= length(x), idx) || throw(ArgumentError("wrt index out of range"))
    ok = try
        targs = ntuple(i -> i in idx ? _smooth_seed(x[i]) : x[i], length(x))
        f(targs...)
        true
    catch
        false
    end
    return ok && !hasbranching(f, x...)
end
