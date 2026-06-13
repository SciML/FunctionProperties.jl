# FunctionProperties.jl: Compiler-Based Proofs of Function Properties

FunctionProperties.jl is a package which contains utilities for testing the
properties of Julia functions. For example, do you need to know if `f` has
internal branches (if statements) in order to know if a given AD optimization
or symbolic pass is valid? This package's functions allows you to perform
such analyses on functions from a user's code by doing a compiler-based
code inspection.

## Installation

To install FunctionProperties.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("FunctionProperties")
```

## Example

```julia
@test hasbranching(1, 2) do x, y
    (x < 0 ? -x : x) + exp(y)
end # true

@test hasbranching(1, 2) do x, y
    ifelse(x < 0, -x, x) + exp(y)
end # false
```

## How does this package work internally?

`hasbranching` inspects the type-inferred but unoptimized IR of the function
(via `code_typed(f, argtypes; optimize = false)`) and checks for `GotoIfNot`
nodes, which represent value-dependent conditional branches. Constructs that do
not introduce data-dependent control flow, such as `ifelse`, do not produce
`GotoIfNot` nodes and are therefore reported as branch-free.

If a function produces a false positive because its branches are known to
resolve at compile time, opt it out with [`is_leaf`](@ref).

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:
    
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
