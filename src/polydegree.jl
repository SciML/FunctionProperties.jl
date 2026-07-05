# Certification by abstract interpretation over polynomial degrees, in the style of tracer-type
# sparsity detection (Gowda et al., "Sparsity Programming", NeurIPS 2019 program-transformations
# workshop): the tracked arguments are seeded with degree-1 tracer numbers and the program is
# executed on them, propagating a sound upper bound on the real-arithmetic polynomial degree.
# Non-polynomial operations on non-constant values poison the result, and every value-inspecting
# predicate on a tracer (comparisons, `isnan`, rounding, conversion) throws, so any computation
# whose control flow or value flow would need the actual numbers aborts the trace instead of
# producing an unsound certificate. `hasbranching` is additionally required to prove the traced
# path is the only path.

# Degrees saturate at `_NOTPOLY` ("not a polynomial of any bounded degree").
const _NOTPOLY = typemax(Int) ÷ 2

struct PolyDegree <: Real
    d::Int
end

_satadd(a::Int, b::Int) = a >= _NOTPOLY || b >= _NOTPOLY ? _NOTPOLY : a + b
_satmul(a::Int, b::Int) = a >= _NOTPOLY || b >= _NOTPOLY ? _NOTPOLY : min(a * b, _NOTPOLY)

Base.promote_rule(::Type{PolyDegree}, ::Type{<:Real}) = PolyDegree
Base.convert(::Type{PolyDegree}, x::PolyDegree) = x
Base.convert(::Type{PolyDegree}, x::Real) = PolyDegree(0)
PolyDegree(x::PolyDegree) = x
# Disambiguators against Base's cross-family `Number` constructors: any value converted into the
# tracer domain is by definition not a seeded variable, hence a constant (degree 0).
PolyDegree(::Base.TwicePrecision) = PolyDegree(0)
PolyDegree(::Complex) = PolyDegree(0)
PolyDegree(::AbstractChar) = PolyDegree(0)
Base.zero(::Type{PolyDegree}) = PolyDegree(0)
Base.zero(::PolyDegree) = PolyDegree(0)
Base.one(::Type{PolyDegree}) = PolyDegree(0)
Base.one(::PolyDegree) = PolyDegree(0)
Base.oneunit(::Type{PolyDegree}) = PolyDegree(0)
Base.float(x::PolyDegree) = x
Base.widen(::Type{PolyDegree}) = PolyDegree

Base.:+(a::PolyDegree, b::PolyDegree) = PolyDegree(max(a.d, b.d))
Base.:-(a::PolyDegree, b::PolyDegree) = PolyDegree(max(a.d, b.d))
Base.:-(a::PolyDegree) = a
Base.:+(a::PolyDegree) = a
Base.:*(a::PolyDegree, b::PolyDegree) = PolyDegree(_satadd(a.d, b.d))
Base.:/(a::PolyDegree, b::PolyDegree) = b.d == 0 ? a : PolyDegree(_NOTPOLY)
Base.:\(a::PolyDegree, b::PolyDegree) = a.d == 0 ? b : PolyDegree(_NOTPOLY)
Base.inv(a::PolyDegree) = a.d == 0 ? a : PolyDegree(_NOTPOLY)
Base.muladd(a::PolyDegree, b::PolyDegree, c::PolyDegree) = a * b + c
Base.fma(a::PolyDegree, b::PolyDegree, c::PolyDegree) = a * b + c
Base.abs2(a::PolyDegree) = a * a
Base.conj(a::PolyDegree) = a
Base.real(a::PolyDegree) = a

function Base.:^(a::PolyDegree, n::Integer)
    n == 0 && return PolyDegree(0)
    n > 0 && return PolyDegree(_satmul(a.d, Int(n)))
    return a.d == 0 ? a : PolyDegree(_NOTPOLY)
end
Base.:^(a::PolyDegree, b::PolyDegree) =
    a.d == 0 && b.d == 0 ? PolyDegree(0) : PolyDegree(_NOTPOLY)

# Non-polynomial scalar functions: constants map to constants; anything else poisons.
for fn in (
        :sqrt, :cbrt, :exp, :exp2, :exp10, :expm1, :log, :log2, :log10, :log1p,
        :sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh, :asinh, :acosh,
        :atanh, :sinpi, :cospi, :sec, :csc, :cot, :abs, :sign,
    )
    @eval Base.$fn(a::PolyDegree) = a.d == 0 ? a : PolyDegree(_NOTPOLY)
end
Base.atan(a::PolyDegree, b::PolyDegree) =
    a.d == 0 && b.d == 0 ? PolyDegree(0) : PolyDegree(_NOTPOLY)
Base.hypot(a::PolyDegree, b::PolyDegree) =
    a.d == 0 && b.d == 0 ? PolyDegree(0) : PolyDegree(_NOTPOLY)
Base.mod(a::PolyDegree, b::PolyDegree) = PolyDegree(_NOTPOLY)
Base.rem(a::PolyDegree, b::PolyDegree) = PolyDegree(_NOTPOLY)

struct DegreeTracerError <: Exception
    op::Symbol
end
function Base.showerror(io::IO, e::DegreeTracerError)
    return print(
        io, "DegreeTracerError: `", e.op, "` needs the value of a traced number, which a ",
        "degree tracer does not carry; the polynomial-degree certificate is abandoned."
    )
end

# Every predicate or conversion that would need the traced VALUE aborts the trace: allowing any
# of these to answer would let value-dependent control or value flow leak into the certificate.
for fn in (:isless, :(==), :<, :(<=), :isequal)
    @eval Base.$fn(a::PolyDegree, b::PolyDegree) = throw(DegreeTracerError($(QuoteNode(fn))))
end
for fn in (:isnan, :isinf, :isfinite, :iszero, :isone, :signbit, :isinteger)
    @eval Base.$fn(a::PolyDegree) = throw(DegreeTracerError($(QuoteNode(fn))))
end
for fn in (:floor, :ceil, :trunc, :round)
    @eval Base.$fn(a::PolyDegree) = throw(DegreeTracerError($(QuoteNode(fn))))
end

_seed(x::Real) = PolyDegree(1)
_seed(x::AbstractArray{<:Real}) = map(_ -> PolyDegree(1), x)
_seed(x) = x   # non-numeric arguments pass through and stay fixed

_max_degree(y::PolyDegree) = y.d
_max_degree(y::Real) = 0
_max_degree(y::AbstractArray) = isempty(y) ? 0 : maximum(_max_degree, y)
_max_degree(y::Tuple) = isempty(y) ? 0 : maximum(_max_degree, y)
_max_degree(@nospecialize(y)) = _NOTPOLY

_wrt_indices(wrt::Integer, n) = (Int(wrt),)
_wrt_indices(wrt::Colon, n) = ntuple(identity, n)
_wrt_indices(wrt, n) = Tuple(Int.(collect(wrt)))

# Certified upper bound on the real-arithmetic polynomial degree of `f` in the arguments selected
# by `wrt`, or `_NOTPOLY` when no certificate can be produced (non-polynomial operations reached
# non-constant values, the trace aborted, or `f` branches on values).
function _degree_bound(f, args, wrt)
    idx = _wrt_indices(wrt, length(args))
    all(i -> 1 <= i <= length(args), idx) || throw(ArgumentError("wrt index out of range"))
    d = try
        targs = ntuple(i -> i in idx ? _seed(args[i]) : args[i], length(args))
        _max_degree(f(targs...))
    catch
        _NOTPOLY
    end
    d < _NOTPOLY || return _NOTPOLY
    # The trace certifies the executed path; `hasbranching` certifies it is the only path.
    return hasbranching(f, args...) ? _NOTPOLY : d
end

"""
    islinear(f, x...; wrt = 1) -> Bool

Attempt to *prove* that `f` is an affine (polynomial degree ≤ 1) function of the arguments
selected by `wrt` (an index, collection of indices, or `:` for all; default the first argument),
holding the remaining arguments fixed at the values given. Arrays are tracked elementwise.

`true` is a certificate under real arithmetic: the degree bound is established by abstract
interpretation with degree-tracking tracer numbers, and [`hasbranching`](@ref) additionally
proves the traced path is the only path. `false` means *not proven* -- `f` may still be linear
(e.g. the bound does not model cancellation: `x^2 - x^2 + x` is not certified), so use `false`
as "fall back to the general path", never as a proof of nonlinearity.

```jldoctest
julia> using FunctionProperties

julia> islinear((u, p, t) -> p[1] * u[1] + p[2], [1.0], [2.0, 3.0], 0.0)
true

julia> islinear((u, p, t) -> u[1] * u[2], [1.0, 2.0], nothing, 0.0)
false
```
"""
function islinear(f, x...; wrt = 1)
    return _degree_bound(f, x, wrt) <= 1
end

"""
    isquadratic(f, x...; wrt = 1) -> Bool

Attempt to *prove* that `f` is a polynomial of degree ≤ 2 in the arguments selected by `wrt`,
holding the remaining arguments fixed. Same certification semantics and conservatism as
[`islinear`](@ref): `true` is a proof under real arithmetic, `false` means not proven.

```jldoctest
julia> using FunctionProperties

julia> isquadratic((u, p, t) -> u[1] * u[2] + p[1] * u[1], [1.0, 2.0], [3.0], 0.0)
true

julia> isquadratic((u, p, t) -> exp(u[1]), [1.0], nothing, 0.0)
false
```
"""
function isquadratic(f, x...; wrt = 1)
    return _degree_bound(f, x, wrt) <= 2
end

"""
    isautonomous(f, x...; wrt = length(x)) -> Bool

Attempt to *prove* that `f` does not depend on the argument selected by `wrt` (by SciML
convention the trailing time argument, the default). Holding the other arguments fixed at the
values given, `true` certifies that the selected argument cannot influence the output: it is
seeded with a degree tracer, and the output is certified constant (degree `0`) in it, with
[`hasbranching`](@ref) proving the traced path is the only path. `false` means *not proven*.

```jldoctest
julia> using FunctionProperties

julia> isautonomous((u, p, t) -> p[1] * u[1], [1.0], [2.0], 0.0)
true

julia> isautonomous((u, p, t) -> u[1] * sin(t), [1.0], [2.0], 0.0)
false
```
"""
function isautonomous(f, x...; wrt = length(x))
    return _degree_bound(f, x, wrt) == 0
end
