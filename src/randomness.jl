# Conservative detection of randomness by walking the same statically-resolved call graph as
# `hasbranching` (a sibling walk kept separate so the branch analysis stays untouched): any
# reachable call into the `Random` standard library is reported. The polarity matches the rest
# of the package: `true` means "may use randomness" (a call was found, the walk hit its budget,
# or the entry was unanalyzable), `false` is the certificate. The same visibility boundaries as
# `hasbranching` apply (calls hidden behind dynamic dispatch are not seen).

"""
    hasrandomness(f, x...) -> Bool

Conservatively check whether `f` can reach the random number generator: `true` when any call
into the `Random` standard library (`rand`, `randn`, `shuffle`, seeding, ...) is statically
reachable from `f`, or when the analysis cannot rule it out. `false` certifies that no such
call is reachable through statically resolvable calls.

This matters wherever a trace of `f` is replayed, e.g. compiled ReverseDiff tapes freeze the
random draws of the recording pass, silently changing the semantics of a stochastic `f`.

## Example

```jldoctest
julia> using FunctionProperties

julia> hasrandomness(x -> x + 1, 2.0)
false
```
"""
function hasrandomness(f, x...)
    sig = Tuple{Core.Typeof(f), Core.Typeof.(x)...}
    return _uses_randomness(sig, Set{Any}(), 0)
end

function _uses_randomness(@nospecialize(sig), seen, depth)
    depth > RECURSION_LIMIT && return true
    sig in seen && return false
    push!(seen, sig)
    results = try
        Base.code_typed_by_type(sig; optimize = false)
    catch
        depth == 0 || return false
        delete!(seen, sig)
        return true
    end
    scanned_any = false
    for pair in results
        ci = first(pair)
        ci isa Core.CodeInfo || continue
        scanned_any = true
        for stmt in ci.code
            if _rng_recurse_call(stmt, ci, seen, depth)
                delete!(seen, sig)
                return true
            end
        end
    end
    if depth == 0 && !scanned_any
        delete!(seen, sig)
        return true
    end
    return false
end

# Mirrors `_recurse_call`'s dissection of `:invoke`/`:call`/splat statements, with the callee
# policy inverted for the Random stdlib: it is the poison rather than a leaf.
function _rng_recurse_call(@nospecialize(stmt), ci, seen, depth)
    call = Meta.isexpr(stmt, :(=)) ? stmt.args[2] : stmt

    if Meta.isexpr(call, :invoke)
        mi = call.args[1]
        callsig = mi isa Core.MethodInstance ? mi.specTypes :
            (
                isdefined(mi, :def) && getfield(mi, :def) isa Core.MethodInstance ?
                getfield(mi, :def).specTypes : nothing
            )
        callsig === nothing && return false
        if callsig isa DataType
            for at in callsig.parameters
                _rng_function_arg(at, seen, depth) && return true
            end
        end
        return _rng_visit(callsig, seen, depth)
    end

    Meta.isexpr(call, :call) || return false
    if _is_apply(call.args[1])
        args = call.args
        fpos = args[1].name === :_apply_iterate ? 3 : 2
        length(args) >= fpos || return false
        ftype, _ = _resolve_callee(args[fpos], ci)
        ftype === nothing && return false
        argtypes = Any[]
        for a in @view args[(fpos + 1):end]
            at = _value_type(a, ci)
            if at isa DataType && at <: Tuple && Base.isconcretetype(at)
                append!(argtypes, at.parameters)
            else
                return false
            end
        end
        any(at -> _rng_function_arg(at, seen, depth), argtypes) && return true
        return _rng_visit(Tuple{ftype, argtypes...}, seen, depth)
    end
    ftype, _ = _resolve_callee(call.args[1], ci)
    ftype === nothing && return false
    argtypes = Any[_value_type(a, ci) for a in @view call.args[2:end]]
    any(at -> _rng_function_arg(at, seen, depth), argtypes) && return true
    return _rng_visit(Tuple{ftype, argtypes...}, seen, depth)
end

# A function VALUE passed as an argument -- `rand.()`, `map(x -> x + rand(), u)` -- never
# appears as a visible call in the caller's IR, so the callee walk alone misses it
# (demonstrated with both shapes). Poison Random-owned function arguments directly, walk the
# methods of user-defined singleton functions and closures, and treat closures with captures
# (no singleton instance to inspect) as "could be random".
const _RNG_NAMES = (
    :rand, :randn, :randexp, :rand!, :randn!, :randexp!, :randstring, :randcycle,
    :randcycle!, :randperm, :randperm!, :randsubseq, :randsubseq!, :shuffle, :shuffle!,
    :bitrand, :seed!,
)

function _rng_function_arg(@nospecialize(at), seen, depth)
    at isa DataType && at <: Function || return false
    root = Base.moduleroot(parentmodule(at))
    nameof(root) === :Random && return true
    if isdefined(at, :instance) && at.instance isa Function
        # `rand`/`randn` are Base-owned functions that the Random stdlib merely extends, so
        # ownership rules miss them: match the family by name (a user function that shadows one
        # of these names is flagged too, erring conservative).
        nameof(at.instance) in _RNG_NAMES && return true
        if root === Base || root === Core ||
                (Base.pkgdir(root) !== nothing && startswith(Base.pkgdir(root), Sys.STDLIB))
            return false
        end
        for meth in methods(at.instance)
            _uses_randomness(meth.sig, seen, depth + 1) && return true
        end
        return false
    end
    if root === Base || root === Core ||
            (Base.pkgdir(root) !== nothing && startswith(Base.pkgdir(root), Sys.STDLIB))
        return false
    end
    # A user closure with captured state has no singleton instance to inspect: conservative.
    return true
end

function _rng_visit(@nospecialize(callsig), seen, depth)
    m = try
        Base.which(callsig)
    catch
        return false
    end
    root = Base.moduleroot(m.module)
    # The name comparison avoids taking a dependency on the Random stdlib just to compare module
    # identity; a user package that happens to be named `Random` is flagged too, which errs on
    # the conservative side.
    nameof(root) === :Random && return true
    _is_library_method(m) && return false
    return _uses_randomness(callsig, seen, depth + 1)
end
