using SciMLTesting, FunctionProperties, JET, Test

# `hasbranching` is a compiler-introspection utility: it `code_typed`s `f` and scans the
# resulting IR for `Core.GotoIfNot` nodes, and builds the dispatch signature with
# `Core.Typeof`. Both names are internal to `Core` with no public equivalent
# (`typeof` differs from `Core.Typeof` on type-valued arguments, and `Base.typesof`
# is itself non-public), so these two accesses are ignored in the public-API checks.
run_qa(
    FunctionProperties;
    explicit_imports = true,
    ei_kwargs = (;
        all_explicit_imports_are_public = (; ignore = (:GotoIfNot,)),
        all_qualified_accesses_are_public = (; ignore = (:Typeof,)),
    )
)
