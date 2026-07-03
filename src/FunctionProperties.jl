module FunctionProperties

using Core: GotoIfNot

# Backstop against pathological recursion depth; real call trees that matter here are shallow.
const RECURSION_LIMIT = 256

# ---- experimental: constant-propagation-aware recursion ------------------------------------
#
# When recursing into a call, ordinary analysis widens every argument to its type, discarding
# constants. A branch decided by a constant argument (e.g. selecting a buffer by a literal index
# inside a parameter container) therefore looks value-dependent even though every real call site
# folds it. If we instead preserve the `Core.Const` argument and re-run *inference* (no optimizer,
# so no library/structural branches are inlined into view), such branches fold to `Core.Const`
# conditions that [`_is_const_gotoifnot`](@ref) already skips -- generalizing [`is_leaf_sig`](@ref)
# without any per-container knowledge.
#
# This relies on `Base.Compiler` (`Core.Compiler`) internals whose API churns across Julia versions
# (the `InferenceState` construction and inferred-source extraction already differ between 1.12 and
# 1.13), so it is *functionally* capability-gated -- see `_const_prop_capable` -- and OFF by
# default. Enable with [`enable_const_prop!`](@ref).
const _CC = isdefined(Base, :Compiler) ? Base.Compiler : Core.Compiler

const _CONST_PROP = Ref(false)
# `nothing` until the functional probe has run; then `true`/`false`.
const _CONST_PROP_CAPABLE = Ref{Union{Nothing, Bool}}(nothing)

# Fixture with a branch decided purely by a constant integer index -- the shape the feature must
# fold. Used only by the capability probe.
struct _ProbeContainer
    a::Int
    b::Int
end
@generated function _probe_indexed(x::_ProbeContainer, idx::Int)
    quote
        if idx == 1
            return x.a
        else
            return x.b
        end
    end
end

# Verify, on the running Julia, that constant inference actually folds a constant-decided branch:
# the constant-index call must come back branch-free while the widened-index call must not. If the
# compiler internals we depend on have shifted shape, this returns `false` and the feature stays
# inert (so behaviour is identical to the plain type recursion). Probed once, then cached.
function _probe_const_prop()
    sig = Tuple{typeof(_probe_indexed), _ProbeContainer, Int}
    folded = _const_infer_src(sig, Any[Core.Const(_probe_indexed), _ProbeContainer, Core.Const(1)])
    widened = _const_infer_src(sig, Any[Core.Const(_probe_indexed), _ProbeContainer, Int])
    folded isa Core.CodeInfo || return false
    widened isa Core.CodeInfo || return false
    return _count_nonconst_gotoifnot(folded) == 0 && _count_nonconst_gotoifnot(widened) > 0
end

function _const_prop_capable()
    v = _CONST_PROP_CAPABLE[]
    if v === nothing
        v = try
            _probe_const_prop()
        catch
            false
        end
        _CONST_PROP_CAPABLE[] = v
    end
    return v
end

"""
    enable_const_prop!(on::Bool = true) -> Bool

Experimental. Toggle constant-propagation-aware recursion in [`hasbranching`](@ref). When on (and
the running Julia's compiler internals still fold a constant-decided branch, verified functionally
by [`_const_prop_capable`](@ref)), a call with constant arguments is re-inferred with those
constants preserved, so value-independent branches decided by a constant (e.g. selecting a buffer
by a literal index) fold away instead of being reported. Off by default because it depends on
compiler internals. Returns the effective state.
"""
function enable_const_prop!(on::Bool = true)
    _CONST_PROP[] = on
    return _const_prop_active()
end

_const_prop_active() = _CONST_PROP[] && _const_prop_capable()

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
        _scan_codeinfo(ci, seen, depth) && return true
    end
    return false
end

function _scan_codeinfo(ci, seen, depth)
    for stmt in ci.code
        if isa(stmt, GotoIfNot)
            _is_const_gotoifnot(stmt, ci) || return true
        elseif _recurse_call(stmt, ci, seen, depth)
            return true
        end
    end
    return false
end

# Constant-argument recursion (experimental, see [`enable_const_prop!`](@ref)): re-infer the callee
# with the constant lattice elements preserved so branches decided by a constant argument fold to
# `Core.Const` conditions. Inference is run *without* the optimizer, so no library/structural
# branches are inlined into view. Falls back to the plain type recursion whenever the constant
# inference is unavailable or fails.
function _hasbranching_const(@nospecialize(sig), argtypes, seen, depth)
    depth > RECURSION_LIMIT && return false
    key = (sig, _const_key(argtypes))
    key in seen && return false
    push!(seen, key)
    src = _const_infer_src(sig, argtypes)
    src isa Core.CodeInfo || return _hasbranching(sig, seen, depth)
    return _scan_codeinfo(src, seen, depth)
end

_const_key(argtypes) = map(x -> x isa Core.Const ? (true, x.val) : (false, x), argtypes)

# Run inference on `sig` with the given argument lattice (some `Core.Const`) preserved, and return
# the inferred (unoptimized) `CodeInfo`, or `nothing` if the compiler internals do not cooperate.
# The `InferenceState` construction and the inferred-source location differ across Julia versions:
# 1.12 accepts `InferenceState(result, cache_mode, interp)` and exposes the body on `result.src`,
# while 1.13 wants the uninferred source passed explicitly and exposes the body on `frame.src`. We
# try the explicit-source form first (works on both) with the non-caching `:volatile` mode, then
# fall back, and read whichever of `frame.src`/`result.src` is a `CodeInfo`. Any shape we don't
# recognise simply yields `nothing`, and the functional probe (`_const_prop_capable`) keeps the
# whole feature inert on such versions.
function _const_infer_src(@nospecialize(sig), argtypes)
    m = try
        Base.which(sig)
    catch
        return nothing
    end
    mi = try
        Base.specialize_method(m, sig, Core.svec())
    catch
        return nothing
    end
    overridden = BitVector(x isa Core.Const for x in argtypes)
    src0 = try
        _CC.retrieve_code_info(mi, Base.get_world_counter())
    catch
        nothing
    end
    # A fresh `InferenceResult`/`InferenceState` per attempt: an `InferenceResult` cannot be
    # re-inferred once used.
    for build in (
            interp -> src0 isa Core.CodeInfo ?
                      _CC.InferenceState(_new_result(mi, argtypes, overridden), src0, :volatile, interp) :
                      nothing,
            interp -> _CC.InferenceState(_new_result(mi, argtypes, overridden), :volatile, interp),
        )
        src = try
            interp = _CC.NativeInterpreter()
            frame = build(interp)
            frame === nothing && continue
            _CC.typeinf(interp, frame)
            _inferred_src(frame)
        catch
            nothing
        end
        src isa Core.CodeInfo && return src
    end
    return nothing
end

_new_result(mi, argtypes, overridden) = _CC.InferenceResult(mi, Any[argtypes...], overridden)

function _inferred_src(frame)
    if isdefined(frame, :src) && getfield(frame, :src) isa Core.CodeInfo
        return getfield(frame, :src)
    end
    r = getfield(frame, :result)
    return (r isa _CC.InferenceResult && r.src isa Core.CodeInfo) ? r.src : nothing
end

_count_nonconst_gotoifnot(ci::Core.CodeInfo) =
    count(s -> isa(s, GotoIfNot) && !_is_const_gotoifnot(s, ci), ci.code)

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
        _, fval = _resolve_callee(call.args[2], ci)
        arglat = Any[_arg_lattice(a, ci) for a in @view call.args[3:end]]
        return _recurse_sig(callsig, fval, arglat, seen, depth)
    end

    Meta.isexpr(call, :call) || return false
    if _is_apply(call.args[1])
        return _recurse_apply(call, ci, seen, depth)
    end
    ftype, fval = _resolve_callee(call.args[1], ci)
    ftype === nothing && return false
    arglat = Any[_arg_lattice(a, ci) for a in @view call.args[2:end]]
    return _recurse_sig(Tuple{ftype, (_lat_type(x) for x in arglat)...}, fval, arglat, seen, depth)
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
    # Splatted arguments are recovered from tuple element *types*; constants are not available here.
    return _recurse_sig(Tuple{ftype, argtypes...}, fval, Any[argtypes...], seen, depth)
end

function _recurse_sig(@nospecialize(callsig), @nospecialize(fval), arglat, seen, depth)
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
    if _const_prop_active() && any(x -> x isa Core.Const, arglat)
        funclat = fval !== nothing ? Core.Const(fval) : _first_param(callsig)
        return _hasbranching_const(callsig, Any[funclat, arglat...], seen, depth + 1)
    end
    return _hasbranching(callsig, seen, depth + 1)
end

_first_param(@nospecialize(sig)) =
    (sig isa DataType && !isempty(sig.parameters)) ? sig.parameters[1] : Any
_lat_type(@nospecialize(x)) = x isa Core.Const ? Core.Typeof(x.val) : x

# Argument lattice element: a `Core.Const` when the argument is a compile-time constant, otherwise
# the widened type. Preserving the `Core.Const` is what lets a constant index survive the recursion
# boundary so `_hasbranching_const` can fold the branch it decides.
function _arg_lattice(@nospecialize(a), ci)
    if a isa Core.SSAValue
        t = ci.ssavaluetypes[a.id]
        return t isa Core.Const ? t : _widen(t)
    elseif a isa Core.Argument
        st = ci.slottypes
        st === nothing && return Any
        t = st[a.n]
        return t isa Core.Const ? t : _widen(t)
    elseif a isa Core.SlotNumber
        st = ci.slottypes
        st === nothing && return Any
        t = st[a.id]
        return t isa Core.Const ? t : _widen(t)
    elseif a isa GlobalRef
        return (isdefined(a.mod, a.name) && isconst(a.mod, a.name)) ?
            Core.Const(getglobal(a.mod, a.name)) : Any
    elseif a isa QuoteNode
        return Core.Const(a.value)
    elseif a isa Expr || a isa Core.GotoNode || a isa GotoIfNot ||
            a isa Core.NewvarNode || a isa Core.ReturnNode
        return Any
    else
        # Raw literal constant embedded in the IR (e.g. an `Int` index).
        return Core.Const(a)
    end
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

export hasbranching, is_leaf, is_leaf_sig, enable_const_prop!

end
