# The radial blast wave, **cell-centred** -- the study as it stood before
# vertex centring, kept verbatim beside `blast_tests.jl` so that its
# numbers stay under test.
#
# This is the case where the centring is most visible, which is why it is
# the one worth running twice. The peak sits at the centre of the box:
# that is a grid *point* on a vertex mesh, sampled at exactly 1.0, and a
# block corner on a cell-centred one, so the two layouts see different
# curvature at the sharpest part of the initial data. Measured, they
# nevertheless reach the *same mesh* -- 820 blocks, growth 6.0294..., the
# same digits -- with L∞ 0.02971 cell against 0.02910 vertex.
#
# The claims about the indicator itself are in `refinement_tests.jl`.

# The uniform sweep is shared by two testsets below -- the convergence
# claim needs all three meshes, and the adaptive claim needs the coarsest
# and the finest as its two reference points. Running it once keeps the
# suite's 1.5 s N = 32 run to a single occurrence.
const BLASTCELL = cellcentered(2)
blast_uniform_cell = [uniform_blast(Val(2); roots=8, N=N,
                                    centering=BLASTCELL)
                      for N in (8, 16, 32)]

# The "exact ring reproduces its own initial data" testset is *not*
# repeated here. It evaluates `blast_exact` and `blast_initial` at
# arbitrary coordinates and never builds a field set at all, so there is
# no centering in it to vary -- running it twice would cost 0.7 s to
# assert the same thing about the same quadrature.

@testset "A uniform cell-centred mesh converges at 2nd order" begin
    # The ring is not an artifact of the quadrature: if `blast_exact` were
    # wrong by anything that does not itself scale like h², the measured
    # rate would not be 2. Measured 1.98 (L2) and 1.95 (L∞); see CODE.md.
    hs = [r.h for r in blast_uniform_cell]
    @test convergence_rate(hs, [r.l2 for r in blast_uniform_cell]) > 1.7
    @test convergence_rate(hs, [r.err for r in blast_uniform_cell]) > 1.7
end

@testset "Cell-centred refinement follows the expanding ring" begin
    # The mesh has to chase a front that moves outward *and* fades. The
    # depth is an output, not a setting: at σ = 0.08 the indicator's τ
    # falls below refine_tol at level 2 on its own, so `maxlevel_cap`
    # never binds and a hierarchy deeper than 2 would mean the
    # calibration had drifted.
    coarse, fine = blast_uniform_cell[1], blast_uniform_cell[3]
    amr = track_blast(Val(2); centering=BLASTCELL)

    @test amr.maxlevel == 2

    # Refinement is worth having: the coarse mesh is an order worse.
    @test coarse.err > 10 * fine.err
    @test amr.worst < coarse.err / 5

    # The refined region grows with the ring rather than staying put, as
    # it does for the travelling pulse. Measured 136 -> 820 blocks.
    @test amr.growth > 3

    # Most of the ring rides on the finest level. Not *all* of it: where τ
    # has fallen below refine_tol the criterion has judged the block
    # adequately resolved and left it coarse, which is the criterion
    # working rather than failing. Measured 0.911.
    @test amr.covered > 0.85

    # The cost of that judgement, and the reason this does NOT assert the
    # pulse test's `rtol = 0.1` match against the fine mesh: the adaptive
    # run is measurably worse than uniform-fine, at four fifths of its
    # cells. Measured ratio 1.45.
    @test amr.worst < 1.6 * fine.err
    @test amr.nblocks * 8^2 < fine.cells
end

@testset "A frozen cell-centred amplitude scale refines everything" begin
    # `track_pulse` measures `field_scales` once, which is sound for a
    # problem that conserves its amplitude. This one does not, and the
    # failure is not subtle. Two separate things break: the peak falls by
    # 9x over the run, so a floor referred to the initial amplitude is out
    # of proportion to the data; and `∂ₜu ≡ 0` initially, so the
    # variable-2 scale starts at *exactly* zero and the indicator becomes
    # scale-free -- the pathology `lohner` exists to avoid.
    frozen = track_blast(Val(2); t_end=0.1, refresh_scales=false,
                         centering=BLASTCELL)
    tracked = track_blast(Val(2); t_end=0.1, refresh_scales=true,
                          centering=BLASTCELL)

    # 8^2 roots refined twice is 1024 blocks: the entire domain, at the
    # cap, everywhere. Note `covered` is then trivially 1.0 -- everything
    # is at the finest level because everything was refined -- which is
    # why block count and not coverage is the measure here.
    @test frozen.nblocks == 1024
    @test tracked.nblocks < frozen.nblocks / 3
end
