# The radial blast wave, **vertex-centred**: a feature that *loses
# amplitude* as it spreads and whose refined region *grows*. Neither of
# the other two initial conditions does either, and the second of those
# two properties is what puts the refinement criterion's noise floor
# under real strain.
#
# It is also the case where the centring is most visible -- the peak sits
# on a grid point here and on a block corner under cell centring -- so
# the cell-centred run of the identical study is kept in
# `blast_cell_tests.jl` rather than being described.
#
# The claims about the indicator itself are in `refinement_tests.jl`.

# The uniform sweep is shared by two testsets below -- the convergence
# claim needs all three meshes, and the adaptive claim needs the coarsest
# and the finest as its two reference points. Running it once keeps the
# suite's 1.5 s N = 32 run to a single occurrence.
blast_uniform = [uniform_blast(Val(2); roots=8, N=N) for N in (8, 16, 32)]

@testset "The exact ring reproduces its own initial data" begin
    # `blast_exact` is a Hankel-transform quadrature, not a formula, so
    # everything downstream rests on it being right. At t = 0 it has to
    # return the initial data, which catches a wrong transform
    # normalization, a wrong quadrature weight, and a missing periodic
    # image all at once -- the corner of the box is where the nearest
    # image is exactly as close as the source, so an unsummed image shows
    # up there and nowhere else.
    L, σ = 1.0, 0.08
    x₀ = (L / 2, L / 2)
    exact = blast_exact(L, x₀, σ, 0.0)
    initial = blast_initial(2, L, x₀, σ)

    xs = range(0.0, L; length=41)
    worst = maximum(abs(exact((x, y), 1) - initial((x, y), 1))
                    for x in xs, y in xs)
    # Four orders below the discretization error measured against it.
    @test worst < 1e-4
    @test maximum(abs(exact((x, y), 2)) for x in xs, y in xs) < 1e-4

    # The order restriction is enforced, not merely documented: a
    # super-Gaussian has no elementary Hankel transform.
    @test_throws ArgumentError blast_exact(L, x₀, σ, 0.0; n=2)
end

@testset "A uniform mesh converges at 2nd order to the exact ring" begin
    # The ring is not an artifact of the quadrature: if `blast_exact` were
    # wrong by anything that does not itself scale like h², the measured
    # rate would not be 2. Measured 1.98 (L2) and 1.96 (L∞); see CODE.md.
    hs = [r.h for r in blast_uniform]
    @test convergence_rate(hs, [r.l2 for r in blast_uniform]) > 1.7
    @test convergence_rate(hs, [r.err for r in blast_uniform]) > 1.7
end

@testset "Refinement follows the expanding ring" begin
    # The mesh has to chase a front that moves outward *and* fades. The
    # depth is an output, not a setting: at σ = 0.08 the indicator's τ
    # falls below refine_tol at level 2 on its own, so `maxlevel_cap`
    # never binds and a hierarchy deeper than 2 would mean the
    # calibration had drifted.
    coarse, fine = blast_uniform[1], blast_uniform[3]
    amr = track_blast(Val(2))

    @test amr.maxlevel == 2

    # Refinement is worth having: the coarse mesh is an order worse.
    @test coarse.err > 10 * fine.err
    @test amr.worst < coarse.err / 5

    # The refined region grows with the ring rather than staying put, as
    # it does for the travelling pulse. Measured 136 -> 820 blocks,
    # the same mesh the cell-centred run reaches.
    @test amr.growth > 3

    # Most of the ring rides on the finest level. Not *all* of it: where τ
    # has fallen below refine_tol the criterion has judged the block
    # adequately resolved and left it coarse, which is the criterion
    # working rather than failing. Measured 0.9105.
    @test amr.covered > 0.85

    # The cost of that judgement, and the reason this does NOT assert the
    # pulse test's `rtol = 0.1` match against the fine mesh: the adaptive
    # run is measurably worse than uniform-fine, at four fifths of its
    # cells. Measured ratio 1.41.
    @test amr.worst < 1.6 * fine.err
    @test amr.nblocks * 8^2 < fine.cells
end

@testset "A frozen amplitude scale refines the whole domain" begin
    # `track_pulse` measures `field_scales` once, which is sound for a
    # problem that conserves its amplitude. This one does not, and the
    # failure is not subtle. Two separate things break: the peak falls by
    # 9x over the run, so a floor referred to the initial amplitude is out
    # of proportion to the data; and `∂ₜu ≡ 0` initially, so the
    # variable-2 scale starts at *exactly* zero and the indicator becomes
    # scale-free -- the pathology `lohner` exists to avoid.
    frozen = track_blast(Val(2); t_end=0.1, refresh_scales=false)
    tracked = track_blast(Val(2); t_end=0.1, refresh_scales=true)

    # 8^2 roots refined twice is 1024 blocks: the entire domain, at the
    # cap, everywhere. Note `covered` is then trivially 1.0 -- everything
    # is at the finest level because everything was refined -- which is
    # why block count and not coverage is the measure here.
    @test frozen.nblocks == 1024
    @test tracked.nblocks < frozen.nblocks / 3   # measured 220 against 1024
end
