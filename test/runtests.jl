using Test
using TreeAMR
using TreeWave
using SpecialFunctions: besselj0

# The suite is expected to pass, with identical numbers, at any thread
# count; CI runs it at one and at four.
@info "Running the tests on $(Threads.nthreads()) thread(s)"

@testset "TreeWave.jl" begin
    include("refinement_tests.jl")
    include("threading_tests.jl")
    include("sinewave_tests.jl")
    include("supergaussian_tests.jl")
    include("blast_tests.jl")
    # The same three studies cell-centred, so that the numbers measured
    # before vertex centring became the default stay under test rather
    # than becoming a paragraph in `CODE.md`. See "Centerings" there.
    include("sinewave_cell_tests.jl")
    include("supergaussian_cell_tests.jl")
    include("blast_cell_tests.jl")
    include("type_tests.jl")
    include("device_tests.jl")
end
