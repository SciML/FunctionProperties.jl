module FunctionProperties

using Core: GotoIfNot

"""
    is_leaf(f, args...) -> Bool

Override this to exempt a function from `hasbranching` analysis.
Return `true` to treat `f` as branch-free regardless of its implementation.

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

Boolean for whether the function's immediate IR contains a conditional branch (`GotoIfNot`).

## Customizing and Removing Functions from the Checks

Some functions may produce false positives because their internal branches are compile-time
constants. Override `FunctionProperties.is_leaf` to opt them out:

```julia
FunctionProperties.is_leaf(::typeof(my_fn)) = true
```
"""
function hasbranching(f, x...)
    is_leaf(f, x...) && return false
    argtypes = Tuple{Core.Typeof.(x)...}
    results = try
        code_typed(f, argtypes; optimize = false)
    catch
        return false
    end
    isempty(results) && return false
    ci = first(results)[1]
    return any(isa(s, GotoIfNot) for s in ci.code)
end

export hasbranching, is_leaf

end
