# FunctionProperties

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/FunctionProperties/stable/)

[![codecov](https://codecov.io/gh/SciML/FunctionProperties.jl/branch/master/graph/badge.svg?token=FwXaKBNW67)](https://codecov.io/gh/SciML/FunctionProperties.jl)
[![Build Status](https://github.com/SciML/FunctionProperties.jl/workflows/CI/badge.svg)](https://github.com/SciML/FunctionProperties.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

FunctionProperties.jl is a package which contains utilities for testing the
properties of Julia functions. For example, do you need to know if `f` has
internal branches (if statements) in order to know if a given AD optimization
or symbolic pass is valid? This package's functions allows you to perform
such analyses on functions from a user's code by doing a compiler-based
code inspection.

## Example

```julia
@test hasbranching(1, 2) do x, y
    (x < 0 ? -x : x) + exp(y)
end # true

@test hasbranching(1, 2) do x, y
    ifelse(x < 0, -x, x) + exp(y)
end # false
```
