# The travelling super-Gaussian pulse: a localized feature, so what it
# measures is whether a *moving* refined region follows it.
#
# The mesh is steered by the per-cell criterion in `refinement.jl`; the
# claims about the indicator itself are in `refinement_tests.jl`.

@testset "A moving refined region tracks a propagating pulse" begin
    # The measure of "without artifacts" is that the adaptive run matches
    # the *uniformly finest* mesh: if the moving coarse-fine interface
    # were reflecting or smearing the pulse, the error would exceed it.
    σ = 0.08
    coarse = uniform_pulse(Val(1); roots=8, N=8, σ=σ)     # level-0 equivalent
    fine = uniform_pulse(Val(1); roots=8, N=32, σ=σ)      # level-2 equivalent
    amr = track_pulse(Val(1); roots=8, N=8, σ=σ, chunk=0.02)

    # Refinement is worth having at all: the coarse mesh is far worse.
    @test coarse.err > 10 * fine.err

    # The refined region never lets the pulse peak escape onto a coarse
    # block over the whole run.
    @test amr.tracking == 1.0
    @test amr.maxlevel == 2

    # And the adaptive run is as accurate as the uniform fine one ...
    @test amr.worst ≈ fine.err rtol = 0.1
    @test amr.worst < coarse.err / 5
    # ... for fewer cells, which is the point of doing this at all.
    @test amr.nblocks * 8 < fine.cells
end
