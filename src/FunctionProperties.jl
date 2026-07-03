module FunctionProperties

using Core: GotoIfNot

# Backstop against pathological recursion depth; real call trees that matter here are shallow.
const RECURSION_LIMIT = 256

# `hasbranching` recurses through statically resolved calls. Ordinary analysis widens every argument
# to its type, which loses constants: a branch decided by a *constant* argument (e.g. selecting a
# buffer by a literal index inside a parameter container) then looks value-dependent even though
# every real call site folds it. When the running Julia's compiler cooperates, such a call is
# re-inferred with its `Core.Const` arguments preserved (no optimizer, so no library/structural
# branches are inlined into view) and the constant-decided branch folds to a `Core.Const` condition
# that `_is_const_gotoifnot` skips. This depends on `Base.Compiler`/`Core.Compiler` internals whose
# API changes across Julia versions, so it is *functionally* gated (see `_const_prop_capable`): it
# activates only where a probe confirms folding actually works, and otherwise the analysis falls
# back to the plain type recursion.
const _CC = isdefined(Base, :Compiler) ? Base.Compiler : Core.Compiler

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

Branches whose condition inference proves constant are ignored (they are not value-dependent),
and — where the compiler cooperates — a call with constant arguments is re-inferred with those
constants preserved so branches they decide fold away rather than being reported.

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

# A `GotoIfNot` whose condition type inference has *proven* constant is a compile-time branch,
# not a value-dependent one: e.g. an `x isa T` test on a concretely-typed field (the SciML
# `ODEFunction` wrapper), the device/type-introspection dispatch inside ML library layers
# (SciML/FunctionProperties.jl#46), or a branch a constant argument folded via the constant-argument
# recursion below. Such a branch can never be taken differently under a tracing AD. A condition that
# is a literal `true`/`false` written directly into the IR is deliberately *not* skipped: that is a
# genuine syntactic branch in user code (e.g. `true ? a : b`). Only conditions inference resolved to
# a `Core.Const` value are dropped; anything we cannot positively prove constant is kept.
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

# Inspect a single IR statement: if it is a statically resolvable call into a non-library method,
# recurse into that method (with any constant arguments preserved). Returns `true` if a branch is
# found downstream.
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

# A splatted call `g(a, bs...)` lowers to `Core._apply_iterate(iter, g, groups...)` (or, on older
# lowerings, `Core._apply(g, groups...)`). The real callee `g` is therefore an *argument* of a
# `Core` builtin, so the plain `:call` path would resolve the callee to `_apply_iterate`, treat it
# as library, and dead-end — missing every branch behind the forwarder. SciML/MTK RHS objects are
# exactly such forwarders (`ODEFunction` -> `GeneratedFunctionWrapper` -> `RuntimeGeneratedFunction`
# -> `generated_callfunc`, each `f(args...)`), so the generated body's branches only become
# reachable by following the apply. The splatted groups are the actual positional arguments;
# recover their element types from the (concrete) tuple types.
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
            # Splatted container whose element types we cannot recover statically: bail rather than
            # guess a wrong signature.
            return false
        end
    end
    # Splatted arguments are recovered from tuple element *types*; constants are not available here.
    return _recurse_sig(Tuple{ftype, argtypes...}, fval, Any[argtypes...], seen, depth)
end

function _recurse_sig(@nospecialize(callsig), @nospecialize(fval), arglat, seen, depth)
    # Honor user `is_leaf` overrides when the concrete function value is recoverable.
    fval !== nothing && is_leaf(fval) && return false
    m = try
        Base.which(callsig)
    catch
        return false
    end
    _is_library_method(m) && return false
    # The type recursion is the source of truth. If it finds no branch, we are done.
    _hasbranching(callsig, seen, depth + 1) || return false
    # It found a branch. If constant arguments decide that branch, re-inferring with the constants
    # preserved folds it away -- so only then do we let the constant path *refute* the finding. This
    # can only downgrade a reported branch to branch-free, never the reverse, and it is skipped
    # entirely (leaving the branch reported) when there are no constant arguments, when the compiler
    # internals do not cooperate, or when the constant inference errors.
    if _const_prop_capable() && any(x -> x isa Core.Const, arglat)
        funclat = fval !== nothing ? Core.Const(fval) : _first_param(callsig)
        _const_refutes(callsig, Any[funclat, arglat...], depth) && return false
    end
    return true
end

# Re-infer `sig` with the constant lattice elements preserved (no optimizer, so no library or
# structural branches are inlined into view) and report whether the result is branch-free. Returns
# `false` -- i.e. does not refute -- whenever the constant inference is unavailable, fails, or leaves
# a branch, so an inability to fold never suppresses a genuine branch.
function _const_refutes(@nospecialize(sig), argtypes, depth)
    src = _const_infer_src(sig, argtypes)
    src isa Core.CodeInfo || return false
    return try
        !_scan_codeinfo(src, Set{Any}(), depth)
    catch
        false
    end
end

_first_param(@nospecialize(sig)) =
    (sig isa DataType && !isempty(sig.parameters)) ? sig.parameters[1] : Any
_lat_type(@nospecialize(x)) = x isa Core.Const ? Core.Typeof(x.val) : x

# Argument lattice element: a `Core.Const` when the argument is a compile-time constant, otherwise
# the widened type. Preserving the `Core.Const` is what lets a constant index survive the recursion
# boundary so `_const_refutes` can fold the branch it decides.
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

# ---- constant-argument inference -----------------------------------------------------------

# Run inference on `sig` with the given argument lattice (some `Core.Const`) preserved, and return
# the inferred (unoptimized) `CodeInfo`, or `nothing` if the compiler internals do not cooperate.
# The `InferenceState` construction and the inferred-source location differ across Julia versions:
# 1.12 accepts `InferenceState(result, cache_mode, interp)` and exposes the body on `result.src`,
# while 1.13 wants the uninferred source passed explicitly and exposes the body on `frame.src`. We
# try the explicit-source form first (works on both) with the non-caching `:volatile` mode, then
# fall back, and read whichever of `frame.src`/`result.src` is a `CodeInfo`.
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

# `nothing` until the functional probe has run; then `true`/`false`.
const _CONST_PROP_CAPABLE = Ref{Union{Nothing, Bool}}(nothing)

# Fixture with a branch decided purely by a constant integer index -- the shape the constant-argument
# recursion must fold. Used only by the capability probe.
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
# compiler internals we depend on have shifted shape, this returns `false` and the constant-argument
# recursion stays inert (behaviour identical to the plain type recursion). Probed once, then cached.
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

# ---- callee/argument resolution ------------------------------------------------------------

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
