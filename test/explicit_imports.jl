using ExplicitImports
using FunctionProperties
using Test

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(FunctionProperties) === nothing
    @test check_no_stale_explicit_imports(FunctionProperties) === nothing
end
