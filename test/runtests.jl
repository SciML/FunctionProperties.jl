using Test
using FunctionProperties
using ComponentArrays, Random

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "QA"
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.instantiate()
    include(joinpath(@__DIR__, "qa", "qa.jl"))
end

if GROUP in ("All", "Core")

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

end
