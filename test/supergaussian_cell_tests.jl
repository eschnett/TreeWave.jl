# The travelling super-Gaussian pulse, **cell-centred** -- the study as it
# stood before vertex centring, kept verbatim beside
# `supergaussian_tests.jl` so that its numbers stay under test.
#
# This is the case where the two layouts should agree, and that is what
# makes it worth running twice: a travelling Gaussian is smooth and
# aligned with nothing, so if vertex and cell centring ever disagree
# here, the disagreement is the port and not the physics. Measured, they
# do not: L∞ 0.09609 cell against 0.09605 vertex, on the same 16 blocks
# at level 2.

@testset "A cell-centred moving refined region tracks a pulse" begin
    # The measure of "without artifacts" is that the adaptive run matches
    # the *uniformly finest* mesh: if the moving coarse-fine interface
    # were reflecting or smearing the pulse, the error would exceed it.
    σ = 0.08
    C = cellcentered(1)
    coarse = uniform_pulse(Val(1); roots=8, N=8, σ=σ, centering=C)
    fine = uniform_pulse(Val(1); roots=8, N=32, σ=σ, centering=C)
    amr = track_pulse(Val(1); roots=8, N=8, σ=σ, chunk=0.02, centering=C)

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
