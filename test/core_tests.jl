using FunctionProperties
using ComponentArrays, Random
using Test

@test hasbranching(1, 2) do x, y
    (x < 0 ? -x : x) + exp(y)
end

@test !hasbranching(1, 2) do x, y
    ifelse(x < 0, -x, x) + exp(y)
end

# Test overloading via is_leaf

f_branch() = true ? 1 : 0
@test FunctionProperties.hasbranching(f_branch)
FunctionProperties.is_leaf(::typeof(f_branch)) = true
@test !FunctionProperties.hasbranching(f_branch)

# Branches behind a non-inlined call boundary must still be detected: the value-dependent
# `if` lives in `branchy_helper`, not in the immediate IR of the entry function.
@noinline branchy_helper(x) = x < 0 ? -x : x
nested_branch_rhs(u, p, t) = branchy_helper(u) + p
@test FunctionProperties.hasbranching(nested_branch_rhs, 1.0, 2.0, 0.0)

@noinline branchfree_helper(x) = x * x + one(x)
nested_branchfree_rhs(u, p, t) = branchfree_helper(u) + p
@test !FunctionProperties.hasbranching(nested_branchfree_rhs, 1.0, 2.0, 0.0)

# An `is_leaf` override stops recursion into the marked callee.
@noinline opted_out_helper(x) = x < 0 ? -x : x
opted_out_rhs(u, p, t) = opted_out_helper(u) + p
@test FunctionProperties.hasbranching(opted_out_rhs, 1.0, 2.0, 0.0)
FunctionProperties.is_leaf(::typeof(opted_out_helper)) = true
@test !FunctionProperties.hasbranching(opted_out_rhs, 1.0, 2.0, 0.0)

# Test simple mutating functions
function f(dx, x)
    return @inbounds dx[1] = x[1]
end
x = zeros(1)
dx = zeros(1)
@test !FunctionProperties.hasbranching(f, dx, x)

# Test broadcast
function f(x)
    return cos.(x .+ x .* x)
end
x = [1.0]
@test !FunctionProperties.hasbranching(f, x)

# Neural networks
#
# The relevant scenario is a neural-network-shaped ODE right-hand side (SciML/SciMLSensitivity.jl#997):
# `hasbranching` must report it as branch-free so a tracing AD like ReverseDiff can compile a tape.
# The forward pass is expressed here as explicit affine transforms plus broadcast activations, which
# is the value flow `hasbranching` actually inspects. We deliberately do not trace a real Lux layer:
# modern Lux layer dispatch routes through device-detection / type-introspection helpers that contain
# genuine (but value-independent, compile-time) `GotoIfNot` branches, which this syntactic IR scan
# cannot distinguish from value-dependent branches (SciML/FunctionProperties.jl#46).
rng = Random.default_rng()
W = randn(rng, Float32, 1, 1)
b = randn(rng, Float32, 1)
p = ComponentArray(; weight = W, bias = b)
t = [0.0]

function f(x, ps)
    return ps.weight * x
end
@test !FunctionProperties.hasbranching(f, t, p)

function f(x, ps)
    return x .+ x
end
@test !FunctionProperties.hasbranching(f, t, p)

# Affine transform followed by a broadcast activation (the original `apply_activation` intent).
function f2(x, ps)
    return identity.(ps.weight * x .+ vec(ps.bias))
end
@test !FunctionProperties.hasbranching(f2, t, p)

# A multi-layer perceptron forward pass built from broadcast `tanh` activations.
rng = Random.default_rng()
tspan = (0.0f0, 8.0f0)
W1 = randn(rng, Float32, 32, 2)
b1 = randn(rng, Float32, 32)
W2 = randn(rng, Float32, 32, 32)
b2 = randn(rng, Float32, 32)
W3 = randn(rng, Float32, 1, 32)
b3 = randn(rng, Float32, 1)
p = ComponentArray(; W1, b1, W2, b2, W3, b3)
θ, ax = getdata(p), getaxes(p)

ann(x, p) = p.W3 * tanh.(p.W2 * tanh.(p.W1 * x .+ p.b1) .+ p.b2) .+ p.b3

function dxdt_(dx, x, p, t)
    x1, x2 = x
    dx[1] = x[2] + first(ann(x, p))
    return dx[2] = first(ann([t, t], p))
end
x0 = [-4.0f0, 0.0f0]
ts = Float32.(collect(0.0:0.01:tspan[2]))
@test !FunctionProperties.hasbranching(dxdt_, copy(x0), x0, p, tspan[1])

# ---------------------------------------------------------------------------------------------
# Value-independent (compile-time-constant) branches must not be reported.
#
# A `GotoIfNot` whose condition inference proves `Core.Const` cannot be taken differently under a
# tracing AD, so it is wrapper/dispatch plumbing rather than the value-dependent branching
# `hasbranching` is meant to surface. This is the shape of the SciML `ODEFunction` functor
# (`if f.f isa AbstractSciMLOperator`) and of ML-library device/type-introspection dispatch
# (SciML/FunctionProperties.jl#46). A *literal* `true`/`false` condition is still a genuine branch
# and is kept (covered by the `f_branch` test above).
abstract type FakeOperator end
struct CondWrap{F}
    f::F
end
function (w::CondWrap)(x)
    if w.f isa FakeOperator      # concretely-typed field => `isa` folds to a constant
        return zero(x)
    else
        return w.f(x)
    end
end
branchfree_inner(x) = x * x + one(x)
branchy_inner(x) = x < 0 ? -x : x
@test !FunctionProperties.hasbranching(CondWrap(branchfree_inner), 1.0)   # const `isa` skipped
@test FunctionProperties.hasbranching(CondWrap(branchy_inner), 1.0)       # real inner branch kept

# ---------------------------------------------------------------------------------------------
# Branches behind a *splatted* call boundary must be detected.
#
# `g(args...)` lowers to `Core._apply_iterate(iter, g, args)`, hiding the real callee `g` as an
# argument of a `Core` builtin. The scan must follow the apply through to `g`, otherwise every
# branch behind a splat forwarder is missed. This is the SciML/MTK RHS shape (`ODEFunction` ->
# `GeneratedFunctionWrapper` -> `RuntimeGeneratedFunction` -> `generated_callfunc`, each a
# `f(args...)` forwarder).
@noinline splat_target_branchy(x) = x < 0 ? -x : x
@noinline splat_target_free(x) = x * x
splat_forward_branchy(args...) = splat_target_branchy(args...)
splat_forward_free(args...) = splat_target_free(args...)
@test FunctionProperties.hasbranching(splat_forward_branchy, -1.0)
@test !FunctionProperties.hasbranching(splat_forward_free, -1.0)

# ---------------------------------------------------------------------------------------------
# Constant-decided branches (value-independent) must not be reported.
#
# A branch on an integer index that selects a buffer (the MTK `getindex(::MTKParameters, ::Int)`
# pattern) is value-independent: each real call site passes a literal index that constant-folds the
# branch, but ordinary recursion widens the `Int` and so reports it. The constant-argument recursion
# re-infers the callee with the constant preserved so the branch folds away — where the running
# Julia's compiler cooperates (`_const_prop_capable()`). It stays conservative: a genuinely
# value-dependent branch, and a dynamic (non-constant) index, are always reported.
struct TwoBufferParams
    a::Vector{Float64}
    b::Vector{Float64}
end
@generated function pick_buffer(p::TwoBufferParams, idx::Int)
    quote
        if idx == 1
            return p.a
        elseif idx == 2
            return p.b
        else
            throw(BoundsError(p, idx))
        end
    end
end
cp_relu(x) = x > 0 ? x : zero(x)
rhs_const_index(p) = @inbounds pick_buffer(p, 1)[1]
rhs_dynamic_index(p, i) = @inbounds pick_buffer(p, i)[1]
rhs_real_branch(u, p) = cp_relu(u) + @inbounds pick_buffer(p, 1)[1]
tbp = TwoBufferParams([1.0], [2.0])

@test FunctionProperties.hasbranching(rhs_real_branch, 1.0, tbp)   # genuine branch: always reported
@test FunctionProperties.hasbranching(rhs_dynamic_index, tbp, 1)   # dynamic index: always reported
if FunctionProperties._const_prop_capable()
    @test !FunctionProperties.hasbranching(rhs_const_index, tbp)   # constant index folds away
end

# Refutation must be per-call-site: the same widened callsig can be refutable at one call site
# (constant index) and not at another (dynamic index). Memoizing the sig on the true-returning
# chain produced order-dependent false negatives (const-site first suppressed the dynamic site).
rhs_mixed_cd(p, i) = pick_buffer(p, 1)[1] + pick_buffer(p, i)[1]
rhs_mixed_dc(p, i) = pick_buffer(p, i)[1] + pick_buffer(p, 1)[1]
@test FunctionProperties.hasbranching(rhs_mixed_cd, tbp, 2)
@test FunctionProperties.hasbranching(rhs_mixed_dc, tbp, 2)

# A constant-recursive callee must not send the refutation into unbounded recursion (previously a
# stack overflow inside inference, which the error handling then converted into a false negative).
# The refutation cycle is broken conservatively, so the branch stays reported -- and quickly.
recur_const(p, n) = n == 0 ? p.a : recur_const(p, 5)
rhs_recur(p) = recur_const(p, 5)[1]
mutual_a(p, n) = n == 0 ? p.a : mutual_b(p, 4)
mutual_b(p, n) = n == 1 ? p.b : mutual_a(p, 3)
rhs_mutual(p) = mutual_a(p, 5)[1]
@test FunctionProperties.hasbranching(rhs_recur, tbp)
@test FunctionProperties.hasbranching(rhs_mutual, tbp)

# A value-dependent branch at the base of a constant-recursion tower must never be lost, even when
# the tower exceeds the refutation depth budget: exhausting the budget fails refutation ("cannot
# verify" reports the branch) rather than assuming a leaf. Previously the depth backstop returned
# branch-free, which a refutation cascade silently propagated into a false negative.
hidden_base(p, n, x) = n == 0 ? (x > 0 ? p.a : p.b) : hidden_base(p, n - 1, x)
rhs_hidden5(p, x) = hidden_base(p, 5, x)[1]
rhs_hidden400(p, x) = hidden_base(p, 400, x)[1]
@test FunctionProperties.hasbranching(rhs_hidden5, tbp, 0.5)
@test FunctionProperties.hasbranching(rhs_hidden400, tbp, 0.5)

# Constant-decided recursion folds fully below the depth budget, and is conservatively reported
# above it. A diverging constant recursion (`n + 1`) must terminate via the depth budget.
cnt_const(p, n) = n == 0 ? p.a : cnt_const(p, n - 1)
rhs_cnt5(p) = cnt_const(p, 5)[1]
rhs_cnt400(p) = cnt_const(p, 400)[1]
asc_const(p, n) = n == 0 ? p.a : asc_const(p, n + 1)
rhs_asc(p) = asc_const(p, 5)[1]
if FunctionProperties._const_prop_capable()
    @test !FunctionProperties.hasbranching(rhs_cnt5, tbp)
end
@test FunctionProperties.hasbranching(rhs_cnt400, tbp)
@test FunctionProperties.hasbranching(rhs_asc, tbp)
