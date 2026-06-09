using FunctionProperties, Aqua, JET, Test

@testset "Aqua" begin
    Aqua.test_all(FunctionProperties)
end

@testset "JET" begin
    JET.test_package(FunctionProperties; target_defined_modules = true)
end
