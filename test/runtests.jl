using Test
using TreeAMR
using TreeWave

@testset "TreeWave.jl" begin
    include("sinewave_tests.jl")
    include("supergaussian_tests.jl")
end
