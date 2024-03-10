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
FunctionProperties.Cassette.overdub(::FunctionProperties.HasBranchingCtx, ::typeof(f_branch), x...) = f_branch(x...)
@test !FunctionProperties.hasbranching(f_branch)

# Test simple mutating functions
function f(dx, x)
    @inbounds dx[1] = x[1]
end
x = zeros(1)
dx = zeros(1)
@test !FunctionProperties.hasbranching(f,dx,x)