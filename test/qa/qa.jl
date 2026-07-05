using SciMLTesting, FunctionProperties, JET, Test

# `hasbranching` is a compiler-introspection utility: it `code_typed`s `f` and scans the
# resulting typed IR for value-dependent branches, so it necessarily reaches into the
# `Core`/`Base` IR and inference internals, none of which have a public equivalent:
#   - `GotoIfNot` (explicit import via `using Core: GotoIfNot`) is the conditional-branch IR node.
#   - `CodeInfo`/`SSAValue`/`SlotNumber`/`Argument` are typed-IR node types scanned in the body.
#   - `Const`/`PartialStruct` are inference lattice element types read off the IR.
#   - `MethodInstance` is the resolved-call type used to recurse through static calls.
#   - `Typeof` builds the dispatch signature (`typeof` differs from `Core.Typeof` on
#     type-valued arguments, and `Base.typesof` is itself non-public).
#   - `code_typed_by_type` is the non-public typed-IR entry point (`Base.code_typed_by_type`).
# All of these are Core/Base compiler-introspection internals with no public API, so they are
# ignored in the public-API checks.
run_qa(
    FunctionProperties;
    explicit_imports = true,
    ei_kwargs = (;
        all_explicit_imports_are_public = (; ignore = (:GotoIfNot,)),
        all_qualified_accesses_are_public = (;
            ignore = (
                :Typeof, :Argument, :CodeInfo, :Const, :MethodInstance,
                :PartialStruct, :SSAValue, :SlotNumber, :code_typed_by_type,
            ),
        ),
    )
)
