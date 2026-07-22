# Thin, capability-gated certificates over the compiler's own analyses: `ispure` wraps the
# effects system and `isinferable` the typed-IR return type. As with the constant-argument
# machinery in `hasbranching.jl`, the compiler internals involved change shape across Julia
# versions, so `ispure` is gated by a functional probe that verifies the effect queries work
# (a known-pure function must certify, a known-impure one must not); where they do not, the
# answer is the conservative "not proven".

const _EFFECTS_CAPABLE = Ref{Union{Nothing, Bool}}(nothing)

function _pure_effects(f, argtypes)
    eff = Base.infer_effects(f, argtypes)
    return _CC.is_consistent(eff) && _CC.is_effect_free(eff)
end

_probe_pure(x) = x + one(x)
const _PROBE_IMPURE_STATE = Ref(0)
_probe_impure(x) = (_PROBE_IMPURE_STATE[] += 1; x)

function _effects_capable()
    v = _EFFECTS_CAPABLE[]
    if v === nothing
        v = try
            _pure_effects(_probe_pure, Tuple{Int}) && !_pure_effects(_probe_impure, Tuple{Int})
        catch
            false
        end
        _EFFECTS_CAPABLE[] = v
    end
    return v
end

"""
    ispure(f, x...) -> Bool

Attempt to *prove* that calling `f` with arguments of the given types is pure: the compiler's
effects analysis must establish both consistency (equal inputs give equal outputs, no
dependence on or mutation of external state) and effect freeness (no observable side effects).
`false` means *not proven*. The underlying effect queries are compiler internals, so they are
verified functionally at first use; where they are unavailable the answer is always `false`.

## Example

```jldoctest
julia> using FunctionProperties

julia> ispure(x -> x^2 + 1, 2.0)
true
```
"""
function ispure(f, x...)
    _effects_capable() || return false
    return try
        _pure_effects(f, Tuple{map(Core.Typeof, x)...})
    catch
        false
    end
end

"""
    isinferable(f, x...) -> Bool

Check whether the return type of `f` for arguments of the given types is inferred as a single
concrete type. `false` means inference produced an abstract or `Union` result -- or that the
call could not be analyzed at all.

## Example

```jldoctest
julia> using FunctionProperties

julia> isinferable(x -> x + 1, 2.0)
true
```
"""
function isinferable(f, x...)
    sig = Tuple{Core.Typeof(f), Core.Typeof.(x)...}
    results = try
        Base.code_typed_by_type(sig; optimize = false)
    catch
        return false
    end
    isempty(results) && return false
    return all(pair -> isconcretetype(_widen(last(pair))), results)
end
