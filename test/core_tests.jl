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
    return quote
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

# The refutation path marker keys non-isbits constants by object identity: hashing the value
# itself stack-overflowed on self-referential constants (uncaught, escaping `hasbranching`) and
# was O(length) on large ones. The exact result is version-dependent (whether inference folds
# predicates on a mutable constant); the invariant is that the query completes.
const SELFREF_CONST = Any[]
push!(SELFREF_CONST, SELFREF_CONST)
selref_pick(p, v) = isempty(v) ? p.a : p.b
rhs_selfref(p) = selref_pick(p, SELFREF_CONST)[1]
@test FunctionProperties.hasbranching(rhs_selfref, tbp) isa Bool

# `hasbranching` consulted from inside a generated-function expansion: reflection is restricted
# there, so the IR may be unobtainable -- the answer must then be the conservative "could be
# branching", not a silent branch-free (which made generators emit the wrong arm).
genhb_branchy(x) = x > 0 ? x : -x
@generated function gen_consults_hb(p)
    return FunctionProperties.hasbranching(genhb_branchy, 1.0) ? :(p.a) : :(p.b)
end
@test gen_consults_hb(tbp) == tbp.a

# Constants are keyed by `objectid` in the refutation machinery: user `hash`/`==` overloads must
# never run (a throwing overload previously escaped `hasbranching` as an uncaught exception).
struct EvilHashBits
    x::Int
end
Base.hash(::EvilHashBits, ::UInt) = error("user hash must not be called")
Base.:(==)(::EvilHashBits, ::EvilHashBits) = error("user == must not be called")
evil_pick(p, e) = e.x == 1 ? p.a : p.b
rhs_evilhash(p) = evil_pick(p, EvilHashBits(1))[1]
@test FunctionProperties.hasbranching(rhs_evilhash, tbp) isa Bool

# Loads from const-bound MUTABLES must stay unfolded (Julia's effects system guarantees this;
# lock it in): folding on current contents would turn into a false negative after mutation.
const MUT_FLAG = Ref(true)
mut_pick(p) = MUT_FLAG[] ? p.a : p.b
@test FunctionProperties.hasbranching(mut_pick, tbp)

# An entry with nothing scannable -- e.g. an opaque closure, which has no entry in the method
# tables -- must answer the conservative "could be branching", not a silent branch-free.
const OC_BRANCHY = Base.Experimental.@opaque p -> p.a[1] > 0 ? p.a : p.b
@test FunctionProperties.hasbranching(OC_BRANCHY, tbp)

# Base's callable wrappers delegate through Base-owned helpers, so the library boundary hid user
# branches inside them (`relu ∘ layer` reported branch-free). Known wrappers are unwrapped into
# component signatures under the normal policy: user components are scanned, Base components stay
# library leaves.
wrap_branchy(x) = x > 0 ? x : zero(x)
wrap_cmp(x, t) = x > t ? x : t
@test FunctionProperties.hasbranching(wrap_branchy ∘ identity, 1.0)
@test FunctionProperties.hasbranching(identity ∘ wrap_branchy, 1.0)
@test !FunctionProperties.hasbranching(abs2 ∘ identity, 1.0)
@test !FunctionProperties.hasbranching(sin ∘ identity, 1.0)   # Base components stay leaves
@test FunctionProperties.hasbranching(Base.Fix1(wrap_cmp, 0.0), 1.0)
@test FunctionProperties.hasbranching(Base.Fix2(wrap_cmp, 0.0), 1.0)
@test !FunctionProperties.hasbranching(Base.Fix2(*, 2.0), 1.0)

# ---------------------------------------------------------------------------------------------
# `islinear` / `isquadratic`: degree certification by tracer-type abstract interpretation.
# `true` is a proof under real arithmetic; `false` is only "not proven".
A_lin = [1.0 2.0; 3.0 4.0]
@test islinear((u, p, t) -> p[1] * u[1] + p[2], [1.0], [2.0, 3.0], 0.0)
@test islinear(u -> A_lin * u, [1.0, 2.0])                    # generic matmul certifies
@test islinear(u -> A_lin * u .+ 1.0, [1.0, 2.0])
@test islinear(x -> 2x + 3, 1.0)
@test islinear(x -> 0.0, 1.0)                                 # constants are affine
@test !islinear((u, p, t) -> u[1] * u[2], [1.0, 2.0], nothing, 0.0)
@test isquadratic((u, p, t) -> u[1] * u[2] + p[1] * u[1], [1.0, 2.0], [3.0], 0.0)
@test isquadratic(x -> (x + 1.0)^2, 1.0)
@test !islinear(x -> (x + 1.0)^2, 1.0)
@test !isquadratic(x -> x^3, 1.0)
@test !isquadratic(u -> exp(u[1]), [1.0])
@test !islinear(u -> max.(u, 0.0), [1.0])                     # tracer aborts on comparison
@test !islinear(x -> x > 0 ? x : zero(x), 1.0)                # relu-style branch
@test !islinear(x -> x * x - x * x + x, 1.0)                  # cancellation: conservative, documented
# `wrt` semantics: joint degree in the tracked arguments, others held fixed.
@test islinear((u, v) -> u[1] + v[1], [1.0], [1.0]; wrt = (1, 2))
@test !islinear((u, v) -> u[1] * v[1], [1.0], [1.0]; wrt = (1, 2))
@test islinear((u, p) -> u[1] * p[1], [1.0], [2.0])           # linear in u for fixed p
@test islinear((u, p) -> u[1] * p[1], [1.0], [2.0]; wrt = :) == false
# A branch on an UNTRACKED argument still blocks certification (`hasbranching` guard): the
# function is linear in `u` for the given `p`, but the certificate is conservatively withheld.
@test !islinear((u, p, t) -> p[1] > 0 ? u[1] : 2u[1], [1.0], [1.0], 0.0)
# In-place right-hand sides via the closure pattern.
rhs_ip!(du, u, p, t) = (du[1] = p[1] * u[1]; du[2] = u[1] + u[2]; du)
@test islinear(u -> rhs_ip!(similar(u), u, [2.0], 0.0), [1.0, 2.0])
# Certified results are plain Bools; the tracer type does not leak.
@test islinear(x -> 2x, 1.0) isa Bool

# Ground-truth cross-validation: every `true` linear certificate must have exactly vanishing
# second finite differences over exact rational arithmetic (and third differences for quadratic
# certificates) at random rational points -- a soundness check independent of the tracer rules.
let rng = Random.Xoshiro(0x1517)
    d2(f, x, h1, h2) = f(x + h1 + h2) - f(x + h1) - f(x + h2) + f(x)
    corpusf = [
        (x -> 3 // 2 * x + 7, true),
        (x -> x * (x + 1) - x * x, true),        # cancellation: genuinely linear...
        (x -> (x + 2) * 5 - 3, true),
        (x -> x^2 + x, false),
        (x -> x^3 - x, false),
        (x -> x * x * 2 + 1, false),
    ]
    for (f, lin_truth) in corpusf
        cert = islinear(f, 1.0)
        # soundness: a certificate implies exact linearity at random rational probes
        if cert
            for _ in 1:3
                x, h1, h2 = (Rational{BigInt}(rand(rng, -99:99)) // rand(rng, 1:9) for _ in 1:3)
                @test iszero(d2(f, x, h1, h2))
            end
        end
        # no false certificates on the known-nonlinear corpus
        lin_truth || @test !cert
    end
end

# ---------------------------------------------------------------------------------------------
# `isautonomous`: certified independence from the trailing (time) argument.
@test isautonomous((u, p, t) -> p[1] * u[1], [1.0], [2.0], 0.0)
@test !isautonomous((u, p, t) -> u[1] * sin(t), [1.0], [2.0], 0.0)
@test !isautonomous((u, p, t) -> u[1] + t, [1.0], [2.0], 0.0)
# `t * 0` is genuinely independent but not certified (no cancellation modeling): conservative.
@test !isautonomous((u, p, t) -> u[1] + t * 0.0, [1.0], [2.0], 0.0)
@test !isautonomous((u, p, t) -> t > 0 ? u[1] : -u[1], [1.0], [2.0], 1.0)

# `issmooth`: certified composition of real-analytic primitives on the domain interior.
@test issmooth((u, p, t) -> exp(u[1]) * sin(t), [1.0], nothing, 0.0)
@test issmooth(x -> sqrt(x + 1.0) * tanh(x), 1.0)
@test issmooth(u -> sum(abs2, u), [1.0, 2.0])
@test !issmooth(u -> abs(u[1]), [1.0])
@test !issmooth(u -> max.(u, 0.0), [1.0])                 # kink through broadcast machinery
@test !issmooth(x -> mod(x, 2.0), 1.0)
@test !issmooth(x -> x > 0 ? x : zero(x), 1.0)            # branch blocks the certificate

# `hasrandomness`: any statically reachable Random-stdlib call, however nested.
rng_leaf(v) = v .+ randn(length(v))
rng_wrap(v) = rng_leaf(v) .* 2
@test hasrandomness(u -> u .+ rand(), [1.0])
@test hasrandomness(rng_wrap, [1.0])
@test hasrandomness(u -> Random.shuffle(u), [1.0, 2.0])
@test !hasrandomness(u -> 2 .* u .+ 1, [1.0])
@test !hasrandomness((u, p, t) -> p[1] * u[1], [1.0], [2.0], 0.0)

# `hasmutation`: per-argument write certification for in-place right-hand sides.
mut_rhs!(du, u, p, t) = (du .= p[1] .* u; nothing)
sneaky_rhs!(du, u, p, t) = (u[1] = 0.0; du .= u; nothing)
@test hasmutation(mut_rhs!, zeros(2), [1.0, 2.0], [3.0], 0.0; arg = 1)
@test !hasmutation(mut_rhs!, zeros(2), [1.0, 2.0], [3.0], 0.0; arg = 2)
@test !hasmutation(mut_rhs!, zeros(2), [1.0, 2.0], [3.0], 0.0; arg = 3)
@test hasmutation(sneaky_rhs!, zeros(2), [1.0, 2.0], [3.0], 0.0; arg = 2)
@test !hasmutation((u, p) -> u .+ p, [1.0], [2.0])                 # out-of-place: nothing written
# Aliasing through another argument withholds the certificate.
alias_u = [1.0]
@test hasmutation((a, b) -> (b[1] = 5.0; nothing), alias_u, alias_u; arg = 1)
# Non-isbits element arrays cannot be certified (interior mutation is invisible to the probe).
@test hasmutation(u -> (u[1][1] = 0.0; nothing), [[1.0]]; arg = 1)
# Witness oracle: a certified-unmutated argument must be exactly unchanged by a real call.
let du = zeros(2), u = [1.0, 2.0], u0 = copy(u)
    @test !hasmutation(mut_rhs!, du, u, [3.0], 0.0; arg = 2)
    mut_rhs!(du, u, [3.0], 0.0)
    @test u == u0
end

# `ispure` / `isinferable`: thin certificates over compiler analyses.
if FunctionProperties._effects_capable()
    @test ispure(x -> 2x + 1, 1.0)
    @test !ispure(x -> x + rand(), 1.0)
    const IMPURE_ACC = Ref(0.0)
    @test !ispure(x -> (IMPURE_ACC[] += x; x), 1.0)
end
@test isinferable(x -> 2x, 1.0)
@test isinferable((u, p, t) -> p[1] .* u, [1.0], [2.0], 0.0)
@test !isinferable(x -> x > 0 ? 1 : 2.0, 1.0)

# ---------------------------------------------------------------------------------------------
# Property-suite hardening: each case below reproduces a demonstrated defect (false certificate
# or missed detection) or an adversarial probe from the hardening battery.

# Text rendering is a value channel: `string(t)` completed with a fixed type-printed string,
# laundering the traced value into untraced data and earning false certificates.
@test !isautonomous((u, p, t) -> u .+ Float64(codeunit(string(t), 1)), [1.0], nothing, 0.5)
@test !issmooth(x -> x + Float64(codeunit(string(x), 1)), 0.5)
@test !isautonomous((u, p, t) -> u .* length("$t"), [1.0], nothing, 0.5)

# Function values in argument position never appear as visible calls: `rand` is Base-owned
# (Random only extends it), so it is matched by name; user singletons are walked by method;
# closures with captures are conservatively random.
@test hasrandomness(v -> v .+ rand.(), [1.0])
@test hasrandomness(v -> map(x -> x + rand(), v), [1.0])
@test hasrandomness(v -> v .+ rand(Random.Xoshiro(1)), [1.0])
@test !hasrandomness(v -> map(x -> x + 1.0, v), [1.0])
@test !hasrandomness(v -> sum(abs2, v), [1.0])

# Aliasing beyond `===`: NamedTuple/struct fields and memory-sharing views must all withhold
# the non-mutation certificate, while a legitimate `nothing` argument must not.
let u = [1.0, 2.0]
    @test hasmutation((a, p) -> (p.cache[1] = 9.0; nothing), u, (cache = u,); arg = 1)
    @test hasmutation((a, v) -> (v[1] = 7.0; nothing), u, view(u, :); arg = 1)
    @test !hasmutation((du, uu, p, t) -> (du .= uu; nothing), zeros(2), u, nothing, 0.0; arg = 2)
end
@test hasmutation((a) -> (fill!(a, 0.0); nothing), [1.0, 2.0]; arg = 1)
@test hasmutation((a, b) -> (copyto!(a, b); nothing), [1.0, 2.0], [3.0, 4.0]; arg = 1)
@test hasmutation((a) -> (empty!(a); nothing), [1.0, 2.0]; arg = 1)         # unsupported: conservative
@test hasmutation((a) -> (view(a, 1:1)[1] = 0.0; nothing), [1.0, 2.0]; arg = 1)
@test hasmutation((a) -> (a[1] += 1.0; a[1] -= 1.0; nothing), [1.0, 2.0]; arg = 1)
@test !hasmutation((x, y) -> x + y, 1.0, 2.0; arg = 1)                       # immutables trivially clean
@test isautonomous((u, p, t) -> u .+ zero(t), [1.0], nothing, 0.5)           # zero(t) is t-independent
@test !isautonomous((u, p, t) -> u .+ Ref(t)[], [1.0], nothing, 0.5)
@test issmooth(x -> (x + 2.0)^2.5, 1.0)
@test !issmooth(x -> sign(x) * x^2, 1.0)

# Fuzz with runtime witness oracles: certified autonomy must mean exact agreement across
# different `t`, and a certified-unmutated argument must be exactly unchanged by a real call.
let rng = Random.Xoshiro(0x0707)
    for k in 1:25
        uses_t = rand(rng, Bool)
        op = rand(rng, 1:3)
        fname = Symbol(:fzham_a, k)
        body = uses_t ?
            (op == 1 ? :(u .+ t) : op == 2 ? :(u .* sin(t)) : :(u .+ t^2)) :
            (op == 1 ? :(u .+ p[1]) : op == 2 ? :(u .* cos(p[1])) : :(u .+ p[1]^2))
        @eval $fname(u, p, t) = $body
        f = @eval $fname
        cert = isautonomous(f, [1.0], [2.0], 0.3)
        uses_t && @test !cert
        cert && @test f([1.0], [2.0], 0.3) == f([1.0], [2.0], 1.7)
    end
    for k in 1:25
        writes_u = rand(rng, Bool)
        fname = Symbol(:fzham_m, k)
        body = writes_u ? :(
                begin
                    du .= p[1] .* uu; uu[1] = 0.0; nothing
                end
            ) :
            :(
                begin
                    du .= p[1] .* uu; nothing
                end
            )
        @eval $fname(du, uu, p, t) = $body
        f = @eval $fname
        cert_mut = hasmutation(f, zeros(2), [1.0, 2.0], [3.0], 0.0; arg = 2)
        writes_u && @test cert_mut
        if !cert_mut
            uu = [1.0, 2.0]; uu0 = copy(uu)
            f(zeros(2), uu, [3.0], 0.0)
            @test uu == uu0
        end
    end
end
