using SciMLTesting, FunctionProperties, JET, Test

# `hasbranching` is a compiler-introspection utility by nature: it scans type-inferred IR for
# `Core.GotoIfNot` nodes and, to fold branches that constant arguments decide, re-runs inference
# with `Core.Const` argument lattices preserved. None of the names this requires have public
# equivalents, so they are allow-listed here rather than papered over:
#
#   - IR/lattice node types (`CodeInfo`, `SSAValue`, `Argument`, `SlotNumber`, `GotoNode`,
#     `NewvarNode`, `ReturnNode`, `Const`, `PartialStruct`, `MethodInstance`, `svec`): the IR
#     being scanned is made of these; there is no public IR representation.
#   - Reflection entry points (`Core.Typeof`, `code_typed_by_type`, `specialize_method`,
#     `get_world_counter`): `typeof` differs from `Core.Typeof` on type-valued arguments, and the
#     signature-based reflection has no public counterpart.
#   - The abstract interpreter (`Compiler`, `NativeInterpreter`, `InferenceResult`,
#     `InferenceState`, `typeinf`, `retrieve_code_info`): there is no public API for "infer this
#     method with constant argument types". This dependency is deliberately confined behind a
#     functional capability probe (`_const_prop_capable`) so the package degrades to the plain
#     type scan wherever these internals change shape.
run_qa(
    FunctionProperties;
    explicit_imports = true,
    ei_kwargs = (;
        all_explicit_imports_are_public = (; ignore = (:GotoIfNot,)),
        all_qualified_accesses_are_public = (;
            ignore = (
                :Typeof, :Argument, :CodeInfo, :Compiler, :Const, :GotoNode,
                :InferenceResult, :InferenceState, :MethodInstance, :NativeInterpreter,
                :NewvarNode, :PartialStruct, :ReturnNode, :SSAValue, :SlotNumber,
                :code_typed_by_type, :get_world_counter, :retrieve_code_info,
                :specialize_method, :svec, :typeinf,
            ),
        ),
    )
)
