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
    is_leaf_sig(sig::Type{<:Tuple}) -> Bool

Signature-level counterpart to [`is_leaf`](@ref), consulted while recursing through statically
resolved calls. `sig` is the call's `Tuple{typeof(f), argtypes...}`. Return `true` to treat the
call as branch-free and stop recursing into it.

Use this (instead of `is_leaf`) when the exemption depends on the *argument types*, not just the
function. The motivating case is value-independent plumbing whose branch is on an index/type
rather than on traced values — e.g. selecting a buffer by integer index inside a parameter
container, where each real call site passes a literal index that constant-folds the branch away,
but the recursion only sees the widened argument type.

## Example

```julia
FunctionProperties.is_leaf_sig(::Type{<:Tuple{typeof(getindex), <:MyParamContainer, Vararg}}) = true
```
"""
is_leaf_sig(@nospecialize(sig)) = false

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
        for stmt in ci.code
            if isa(stmt, GotoIfNot)
                _is_const_gotoifnot(stmt, ci) || return true
            elseif _recurse_call(stmt, ci, seen, depth)
                return true
            end
        end
    end
    return false
end

# A `GotoIfNot` whose condition type inference has *proven* constant is a compile-time branch,
# not a value-dependent one: e.g. an `x isa T` test on a concretely-typed field (the SciML
# `ODEFunction` wrapper) or the device/type-introspection dispatch inside ML library layers
# (SciML/FunctionProperties.jl#46). Such a branch can never be taken differently under a tracing
# AD, so it is not the branching `hasbranching` is meant to surface. A condition that is a literal
# `true`/`false` written directly into the IR is deliberately *not* skipped: that is a genuine
# syntactic branch in user code (e.g. `true ? a : b`). Only conditions inference resolved to a
# `Core.Const` value are dropped; anything we cannot positively prove constant is kept.
function _is_const_gotoifnot(stmt::GotoIfNot, ci)
    cond = stmt.cond
    t = if cond isa Core.SSAValue
        types = ci.ssavaluetypes
        types isa AbstractVector && checkbounds(Bool, types, cond.id) ? types[cond.id] : nothing
    elseif cond isa Core.Argument
        ci.slottypes === nothing ? nothing : get(ci.slottypes, cond.n, nothing)
    elseif cond isa Core.SlotNumber
        ci.slottypes === nothing ? nothing : get(ci.slottypes, cond.id, nothing)
    else
        nothing
    end
    return t isa Core.Const
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
    if _is_apply(call.args[1])
        return _recurse_apply(call, ci, seen, depth)
    end
    ftype, fval = _resolve_callee(call.args[1], ci)
    ftype === nothing && return false
    argtypes = Any[_value_type(a, ci) for a in @view call.args[2:end]]
    return _recurse_sig(Tuple{ftype, argtypes...}, fval, seen, depth)
end

_is_apply(@nospecialize(f)) =
    f isa GlobalRef && f.mod === Core && (f.name === :_apply_iterate || f.name === :_apply)

# A splatted call `g(a, bs...)` lowers to `Core._apply_iterate(iter, g, groups...)` (or, on
# older lowerings, `Core._apply(g, groups...)`). The real callee `g` is therefore an *argument*
# of a `Core` builtin, so the plain `:call` path would resolve the callee to `_apply_iterate`,
# treat it as library, and dead-end — missing every branch behind the forwarder. SciML/MTK RHS
# objects are exactly such forwarders (`ODEFunction` -> `GeneratedFunctionWrapper` ->
# `RuntimeGeneratedFunction` -> `generated_callfunc`, each `f(args...)`), so the generated body's
# branches only become reachable by following the apply through to `g`. The splatted groups are
# the actual positional arguments; recover their element types from the (concrete) tuple types so
# the right method specialization is selected downstream.
function _recurse_apply(call, ci, seen, depth)
    args = call.args
    fpos = args[1].name === :_apply_iterate ? 3 : 2
    length(args) >= fpos || return false
    ftype, fval = _resolve_callee(args[fpos], ci)
    ftype === nothing && return false
    argtypes = Any[]
    for a in @view args[(fpos + 1):end]
        at = _value_type(a, ci)
        if at isa DataType && at <: Tuple && Base.isconcretetype(at)
            append!(argtypes, at.parameters)
        else
            # Splatted container whose element types we cannot recover statically (e.g. a
            # non-`isbits` `Vararg` tuple or an array): bail rather than guess a wrong signature.
            return false
        end
    end
    return _recurse_sig(Tuple{ftype, argtypes...}, fval, seen, depth)
end

function _recurse_sig(@nospecialize(callsig), @nospecialize(fval), seen, depth)
    # Honor user `is_leaf` overrides when the concrete function value is recoverable.
    fval !== nothing && is_leaf(fval) && return false
    # Signature-level overrides: exemptions that depend on the argument types.
    is_leaf_sig(callsig) && return false
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

export hasbranching, is_leaf, is_leaf_sig

end
