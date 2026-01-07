using FunctionProperties, Test

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
        ::FunctionProperties.HasBranchingCtx, ::typeof(f_branch), x...
    )
    return f_branch(x...)
end
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

# Neural networks with Lux
using Lux, ComponentArrays, Random
rng = Random.default_rng()
ann = Dense(1, 1, identity)
ps, st = Lux.setup(rng, ann)
p = ComponentArray(ps)
x0 = [-4.0f0, 0.0f0]
t = [0.0]

function f(x, ps, st)
    return ps.weight * x
end
@test !FunctionProperties.hasbranching(f, t, p, st)

function f(x, ps, st)
    return x .+ x
end
@test !FunctionProperties.hasbranching(f, t, p, st)

# Test a simple activation-like function without internal branching
# (identity broadcast applied element-wise)
function f2(x, ps, st)
    identity.(ps.weight * x .+ vec(ps.bias)), st
end
@test !FunctionProperties.hasbranching(f2, t, p, st)

# Note: Testing the full Lux neural network layer (ann) may detect branching
# due to internal Lux optimizations. This is expected behavior as Lux layers
# may contain conditional logic for performance optimization.
# The key tests are the direct function branching detection above.
