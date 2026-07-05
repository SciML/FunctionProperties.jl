module FunctionProperties

using Core: GotoIfNot

export hasbranching, is_leaf, islinear, isquadratic

# Conservative detection of value-dependent branching by scanning type-inferred IR, with
# constant-argument refutation of branches that literals decide.
include("hasbranching.jl")

# Certification of polynomial degree (`islinear`, `isquadratic`) by abstract interpretation
# with a degree-tracking tracer number type.
include("polydegree.jl")

end
