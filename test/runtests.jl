using FunctionProperties
using Test

@testset "Explicit Imports" begin
    include("explicit_imports.jl")
end

@test hasbranching(1, 2) do x, y
    (x < 0 ? -x : x) + exp(y)
end

@test !hasbranching(1, 2) do x, y
    ifelse(x < 0, -x, x) + exp(y)
end

# Test overloading

f_branch() = true ? 1 : 0
@test FunctionProperties.hasbranching(f_branch)
function FunctionProperties.Cassette.overdub(
        ::FunctionProperties.HasBranchingCtx, ::typeof(f_branch), x...)
    f_branch(x...)
end
@test !FunctionProperties.hasbranching(f_branch)

# Test simple mutating functions
function f(dx, x)
    @inbounds dx[1] = x[1]
end
x = zeros(1)
dx = zeros(1)
@test !FunctionProperties.hasbranching(f, dx, x)

# Test broadcast
function f(x)
    cos.(x .+ x .* x)
end
x = [1.0]
@test !FunctionProperties.hasbranching(f, x)

# Neural networks
using Lux, ComponentArrays, Random
rng = Random.default_rng()
ann = Dense(1, 1, identity)
ps, st = Lux.setup(rng, ann)
p = ComponentArray(ps)
x0 = [-4.0f0, 0.0f0]
t = [0.0]

function f(x, ps, st)
    ps.weight * x
end
@test !FunctionProperties.hasbranching(f, t, p, st)

function f(x, ps, st)
    x .+ x
end
@test !FunctionProperties.hasbranching(f, t, p, st)

function f2(x, ps, st)
    Lux.apply_activation(identity, ps.weight * x .+ vec(ps.bias)), st
end
@test !FunctionProperties.hasbranching(f2, t, p, st)
@test !FunctionProperties.hasbranching(ann, t, p, st)

rng = Random.default_rng()
tspan = (0.0f0, 8.0f0)
ann = Chain(Dense(2, 32, tanh), Dense(32, 32, tanh), Dense(32, 1))
ps, st = Lux.setup(rng, ann)
p = ComponentArray(ps)
θ, ax = getdata(p), getaxes(p)

function dxdt_(dx, x, p, t)
    x1, x2 = x
    dx[1] = x[2] + first(ann(x, p, st))[1]
    dx[2] = first(ann([t, t], p, st))[1]
end
x0 = [-4.0f0, 0.0f0]
ts = Float32.(collect(0.0:0.01:tspan[2]))
@test !FunctionProperties.hasbranching(dxdt_, copy(x0), x0, p, tspan[1])
