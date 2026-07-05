module FunctionProperties

using Core: GotoIfNot

export hasbranching, is_leaf, islinear, isquadratic, isautonomous, issmooth,
    hasrandomness, hasmutation, ispure, isinferable

# Conservative detection of value-dependent branching by scanning type-inferred IR, with
# constant-argument refutation of branches that literals decide.
include("hasbranching.jl")

# Certification of polynomial degree (`islinear`, `isquadratic`) by abstract interpretation
# with a degree-tracking tracer number type.
include("polydegree.jl")

# Certification of smoothness by abstract interpretation with a smoothness tracer.
include("smoothprobe.jl")

# Conservative detection of reachable Random-stdlib calls.
include("randomness.jl")

# Certification of argument non-mutation by write-recording probe arrays.
include("writeprobe.jl")

# Capability-gated certificates over the compiler's effects and inference results.
include("effects.jl")

end
