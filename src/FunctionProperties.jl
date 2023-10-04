module FunctionProperties

using Cassette, DiffRules
using Core: CodeInfo, SlotNumber, SSAValue, ReturnNode, GotoIfNot

const printbranch = false

Cassette.@context HasBranchingCtx

function Cassette.overdub(ctx::HasBranchingCtx, f, args...)
    if Cassette.canrecurse(ctx, f, args...)
        return Cassette.recurse(ctx, f, args...)
    else
        return Cassette.fallback(ctx, f, args...)
    end
end

for (mod, f, n) in DiffRules.diffrules(; filter_modules=nothing)
    if !(isdefined(@__MODULE__, mod) && isdefined(getfield(@__MODULE__, mod), f))
        continue  # Skip rules for methods not defined in the current scope
    end
    @eval function Cassette.overdub(::HasBranchingCtx, f::Core.Typeof($mod.$f),
                                    x::Vararg{Any, $n})
        f(x...)
    end
end

function _pass(::Type{<:HasBranchingCtx}, reflection::Cassette.Reflection)
    ir = reflection.code_info

    if any(x -> isa(x, GotoIfNot), ir.code)
        printbranch && println("GotoIfNot detected in $(reflection.method)\nir = $ir\n")
        Cassette.insert_statements!(ir.code, ir.codelocs,
                                    (stmt, i) -> i == 1 ? 3 : nothing,
                                    (stmt, i) -> Any[Expr(:call,
                                                          Expr(:nooverdub,
                                                               GlobalRef(Base, :getfield)),
                                                          Expr(:contextslot),
                                                          QuoteNode(:metadata)),
                                                     Expr(:call,
                                                          Expr(:nooverdub,
                                                               GlobalRef(Base, :setindex!)),
                                                          SSAValue(1), true,
                                                          QuoteNode(:has_branching)),
                                                     stmt])
        Cassette.insert_statements!(ir.code, ir.codelocs,
                                    (stmt, i) -> i > 2 && isa(stmt, Expr) ? 1 : nothing,
                                    (stmt, i) -> begin
                                        callstmt = Meta.isexpr(stmt, :(=)) ? stmt.args[2] :
                                                   stmt
                                        Meta.isexpr(stmt, :call) ||
                                            Meta.isexpr(stmt, :invoke) || return Any[stmt]
                                        callstmt = Expr(callstmt.head,
                                                        Expr(:nooverdub, callstmt.args[1]),
                                                        callstmt.args[2:end]...)
                                        return Any[Meta.isexpr(stmt, :(=)) ?
                                                   Expr(:(=), stmt.args[1], callstmt) :
                                                   callstmt]
                                    end)
    end
    return ir
end

const pass = Cassette.@pass _pass

"""
    hasbranching(f, x...)

Checks whether the function `f` has branches (if statements) that are dependent on the value x 
that would be taken in a tracing system, such as during AD tracing by a package like ReverseDiff.jl.

## Arguments:

    * `f`: the function to inspect
    * `x`: test arguments for the inspection. These values do not need to be the values that
      would be used in the actual calls to the function but instead prototype values which
      match the types that would be used in the actual function call. This is used to trace to
      the correct internal dispatches.

## Outputs:

    Boolean for whether the function has branches.

## Customizing and Removing Dispatches from the Checks

Some internal functions of a package may cause false positives because a branch may be known to
resolve at compile time. If this is known, then you can add a dispatch to opt that function out
of the analysis via:

```julia
function FunctionProperties.Cassette.overdub(::FunctionProperties.HasBranchingCtx, ::typeof(f), x...) 
    f(x...)
end
```
"""
function hasbranching(f, x...)
    metadata = Dict(:has_branching => false)
    Cassette.overdub(Cassette.disablehooks(HasBranchingCtx(; pass, metadata)), f, x...)
    return metadata[:has_branching]
end

Cassette.overdub(::HasBranchingCtx, ::typeof(+), x...) = +(x...)
Cassette.overdub(::HasBranchingCtx, ::typeof(*), x...) = *(x...)
function Cassette.overdub(::HasBranchingCtx, ::typeof(Base.materialize), x...)
    Base.materialize(x...)
end
function Cassette.overdub(::HasBranchingCtx, ::typeof(Base.literal_pow), x...)
    Base.literal_pow(x...)
end
Cassette.overdub(::HasBranchingCtx, ::typeof(Base.getindex), x...) = Base.getindex(x...)
Cassette.overdub(::HasBranchingCtx, ::typeof(Core.Typeof), x...) = Core.Typeof(x...)
function Cassette.overdub(::HasBranchingCtx, ::Type{Base.OneTo{T}},
                          stop) where {T <: Integer}
    Base.OneTo{T}(stop)
end

export hasbranching

end