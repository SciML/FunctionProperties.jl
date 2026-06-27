module FunctionProperties

using Core: GotoIfNot

# Backstop against pathological recursion depth; real call trees that matter here are shallow.
const RECURSION_LIMIT = 256

"""
    is_leaf(f, args...) -> Bool

Override this to exempt a function from `hasbranching` analysis.
Return `true` to treat `f` as branch-free regardless of its implementation, which also
stops [`hasbranching`](@ref) from recursing into its callees.

## Example

```julia
FunctionProperties.is_leaf(::typeof(my_fn)) = true
```
"""
is_leaf(f, args...) = false

"""
    hasbranching(f, x...)

Checks whether the function `f` has branches (if statements) that are dependent on the
value `x` that would be taken in a tracing system, such as during AD tracing by a package
like ReverseDiff.jl.

## Arguments

  - `f`: the function to inspect.
  - `x`: test arguments. These values don't need to match the actual call values, but their
    *types* must match — they are used to select the right method specialization.

## Outputs

Boolean for whether the function contains a value-dependent conditional branch
(`GotoIfNot`). The type-inferred IR (`code_typed(...; optimize = false)`) of `f` is scanned,
and the scan recurses through statically resolved calls into other user-defined methods so
that branches living behind a non-inlined call boundary are still detected. Calls into
`Base`, `Core`, and the standard libraries are treated as leaves: their internal branches
are structural/compile-time rather than value-dependent user logic, and recursing into them
(e.g. matrix multiply, broadcasting, `getindex` bounds checks) produces false positives.

## Customizing and Removing Functions from the Checks

Some functions may produce false positives because their internal branches are compile-time
constants. Override [`is_leaf`](@ref) to opt them out (this also prevents recursion into
them):

```julia
FunctionProperties.is_leaf(::typeof(my_fn)) = true
```
"""
function hasbranching(f, x...)
    is_leaf(f, x...) && return false
    sig = Tuple{Core.Typeof(f), Core.Typeof.(x)...}
    return _hasbranching(sig, Set{Any}(), 0)
end

function _hasbranching(@nospecialize(sig), seen, depth)
    depth > RECURSION_LIMIT && return false
    sig in seen && return false
    push!(seen, sig)

    results = try
        Base.code_typed_by_type(sig; optimize = false)
    catch
        return false
    end

    for pair in results
        ci = first(pair)
        # Generated functions that were not expanded come back as `Method`, not `CodeInfo`;
        # there is no body to scan, so treat them as leaves.
        ci isa Core.CodeInfo || continue
        any(stmt -> isa(stmt, GotoIfNot), ci.code) && return true
        for stmt in ci.code
            _recurse_call(stmt, ci, seen, depth) && return true
        end
    end
    return false
end

# Inspect a single IR statement: if it is a statically resolvable call into a non-library
# method, recurse into that method's IR. Returns `true` if a branch is found downstream.
function _recurse_call(@nospecialize(stmt), ci, seen, depth)
    call = Meta.isexpr(stmt, :(=)) ? stmt.args[2] : stmt

    if Meta.isexpr(call, :invoke)
        mi = call.args[1]
        callsig = mi isa Core.MethodInstance ? mi.specTypes :
            (
                isdefined(mi, :def) && getfield(mi, :def) isa Core.MethodInstance ?
                getfield(mi, :def).specTypes : nothing
            )
        callsig === nothing && return false
        return _recurse_sig(callsig, nothing, seen, depth)
    end

    Meta.isexpr(call, :call) || return false
    ftype, fval = _resolve_callee(call.args[1], ci)
    ftype === nothing && return false
    argtypes = Any[_value_type(a, ci) for a in @view call.args[2:end]]
    return _recurse_sig(Tuple{ftype, argtypes...}, fval, seen, depth)
end

function _recurse_sig(@nospecialize(callsig), @nospecialize(fval), seen, depth)
    # Honor user `is_leaf` overrides when the concrete function value is recoverable.
    fval !== nothing && is_leaf(fval) && return false
    m = try
        Base.which(callsig)
    catch
        return false
    end
    _is_library_method(m) && return false
    return _hasbranching(callsig, seen, depth + 1)
end

# Library code (`Base`, `Core`, stdlibs) is treated as a leaf: its branches are structural or
# compile-time, not the value-dependent user logic `hasbranching` is meant to surface.
function _is_library_method(m::Method)
    root = Base.moduleroot(m.module)
    (root === Base || root === Core) && return true
    pkgdir = Base.pkgdir(root)
    pkgdir === nothing && return false
    return startswith(pkgdir, Sys.STDLIB)
end

function _resolve_callee(@nospecialize(fexpr), ci)
    if fexpr isa GlobalRef
        if isdefined(fexpr.mod, fexpr.name) && isconst(fexpr.mod, fexpr.name)
            v = getglobal(fexpr.mod, fexpr.name)
            return (Core.Typeof(v), v)
        end
        return (nothing, nothing)
    elseif fexpr isa QuoteNode
        return (Core.Typeof(fexpr.value), fexpr.value)
    elseif fexpr isa Core.SSAValue
        t = ci.ssavaluetypes[fexpr.id]
        t isa Core.Const && return (Core.Typeof(t.val), t.val)
        return (_widen(t), nothing)
    else
        return (_value_type(fexpr, ci), nothing)
    end
end

function _value_type(@nospecialize(a), ci)
    if a isa Core.SSAValue
        return _widen(ci.ssavaluetypes[a.id])
    elseif a isa Core.Argument
        st = ci.slottypes
        return st === nothing ? Any : _widen(st[a.n])
    elseif a isa Core.SlotNumber
        st = ci.slottypes
        return st === nothing ? Any : _widen(st[a.id])
    elseif a isa GlobalRef
        return (isdefined(a.mod, a.name) && isconst(a.mod, a.name)) ?
            Core.Typeof(getglobal(a.mod, a.name)) : Any
    elseif a isa QuoteNode
        return Core.Typeof(a.value)
    else
        return Core.Typeof(a)
    end
end

_widen(@nospecialize t) =
    t isa Core.Const ? Core.Typeof(t.val) :
    t isa Core.PartialStruct ? t.typ :
    isa(t, Type) ? t : Any

export hasbranching, is_leaf

end
