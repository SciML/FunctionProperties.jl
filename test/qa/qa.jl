using SciMLTesting, FunctionProperties, JET, Test

# `hasbranching` is a compiler-introspection utility: it `code_typed`s `f` and scans the
# resulting typed IR for value-dependent branches, and -- to fold branches that constant
# arguments decide -- re-runs inference with `Core.Const` argument lattices preserved. It
# therefore necessarily reaches into the `Core`/`Base` IR and inference internals, none of
# which have a public equivalent:
#   - `GotoIfNot` (explicit import via `using Core: GotoIfNot`) is the conditional-branch IR node.
#   - `CodeInfo`/`SSAValue`/`SlotNumber`/`Argument`/`GotoNode`/`NewvarNode`/`ReturnNode` are
#     typed-IR node types scanned in the body.
#   - `Const`/`PartialStruct` are inference lattice element types read off the IR.
#   - `MethodInstance` is the resolved-call type used to recurse through static calls, and
#     `svec` builds the empty sparam vector for `specialize_method`.
#   - `Typeof` builds the dispatch signature (`typeof` differs from `Core.Typeof` on
#     type-valued arguments, and `Base.typesof` is itself non-public).
#   - `code_typed_by_type` is the non-public typed-IR entry point (`Base.code_typed_by_type`),
#     and `specialize_method`/`get_world_counter` are the reflection pieces needed to build a
#     method instance for constant re-inference.
#   - `infer_effects` and the `is_consistent`/`is_effect_free` queries are the effects system
#     wrapped by `ispure`.
#   - `Compiler`/`NativeInterpreter`/`InferenceResult`/`InferenceState`/`typeinf`/
#     `retrieve_code_info` are the abstract interpreter: there is no public API for "infer this
#     method with constant argument types". This dependency is deliberately confined behind a
#     functional capability probe (`_const_prop_capable`) so the package degrades to the plain
#     type scan wherever these internals change shape.
#   - `TwicePrecision` appears only in a constructor disambiguation that Aqua requires: Base
#     defines `(::Type{T<:Number})(::Base.TwicePrecision)`, which is ambiguous against the
#     degree tracer's generic constructor.
# All of these are Core/Base compiler-introspection internals with no public API, so they are
# ignored in the public-API checks.
run_qa(
    FunctionProperties;
    explicit_imports = true,
    ei_kwargs = (;
        all_explicit_imports_are_public = (; ignore = (:GotoIfNot,)),
        all_qualified_accesses_are_public = (;
            ignore = (
                :Typeof, :Argument, :CodeInfo, :Compiler, :Const, :GotoNode,
                :InferenceResult, :InferenceState, :MethodInstance, :NativeInterpreter,
                :NewvarNode, :PartialStruct, :ReturnNode, :SSAValue, :SlotNumber, :TwicePrecision,
                :infer_effects, :is_consistent, :is_effect_free,
                :code_typed_by_type, :get_world_counter, :retrieve_code_info,
                :specialize_method, :svec, :typeinf,
            ),
        ),
    )
)
