using Test
using TreeAMR
using TreeWave

@testset "TreeWave.jl" begin
    include("refinement_tests.jl")
    include("sinewave_tests.jl")
    include("supergaussian_tests.jl")
    include("blast_tests.jl")
end
