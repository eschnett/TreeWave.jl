# The refinement criterion: a per-cell Löhner indicator reduced to a
# per-block verdict.
#
# These testsets are claims about the *indicator*, not about a run. The
# claim that the criterion actually steers a moving mesh is in
# `supergaussian_tests.jl`, where it is measured against a uniform mesh of
# the same finest spacing.
#
# They run vertex-centred, like everything else, and are not duplicated
# for cell centring. The indicator is arithmetic on stored values: it
# reads a block's `N` owned points and one neighbour either side, which
# is the same walk for either layout, and the one thing the centring
# changes -- the stored index an owned point maps to -- is what
# `test/device_tests.jl` compares the host and device forms on, for both
# layouts. The end-to-end cell-centred claims are in the `*_cell_tests`
# files.

const REFOPS = Operators(prolongation=4, restriction=4)

"""A uniform periodic 1D forest with the pulse in it, ghosts filled."""
function pulse_field(N; roots=8, L=1.0, σ=0.08, x0=0.25, G=1,
                     centering=vertexcentered(1))
    forest = Forest((roots,); N=N, periodic=(true,), extents=((0.0, L),))
    fs = FieldSet(forest, 2; G=G, centering=centering)
    fill_by_coordinates!(pulse_exact(1, L, x0, σ, 0.0), fs)
    fill_ghosts!(fs, GhostSchedule(fs, REFOPS))
    return forest, fs
end

@testset "The indicator vanishes on linear data" begin
    # τ is built from the second difference, which is identically zero for
    # data linear in the index -- so a smooth ramp, however steep, is never
    # a reason to refine. Exact, not approximate.
    for a in (-3.0, 0.0, 1.0, 7.5), b in (-2.0, 0.5, 4.0)
        @test lohner(a - b, a, a + b, 1.0) == 0.0
    end
    @test lohner(5.0, 5.0, 5.0, 1.0) == 0.0        # flat
    @test lohner(0.0, 0.0, 0.0, 1.0) == 0.0        # and not 0/0

    # Bounded above by 1: the numerator is at most the sum of the two
    # first differences, which the denominator already contains.
    for (um, u0, up) in ((0.0, 1.0, 0.0), (-1.0, 1.0, -1.0), (1e3, -1e3, 1e3))
        @test 0.0 <= lohner(um, u0, up, 1.0) <= 1.0
    end
end

@testset "The noise floor is referred to a global scale" begin
    # Regression test for a real bug. Löhner's classic form floors the
    # denominator with the *local* ε(|up| + 2|u0| + |um|), which is
    # scale-free: a ripple ten orders of magnitude below the solution
    # scores as high as the feature itself, and the criterion refines the
    # whole domain. Referring the floor to a global amplitude fixes it.
    dust = (3.9e-16, 3.6e-17, 3.6e-17)             # measured, from the pulse tail
    @test lohner(dust..., 1.0) < 1e-12             # against a scale-1 solution
    @test lohner(dust..., 3.6e-17) > 0.5           # against its own scale

    # The same shape at O(1) is a genuine feature either way.
    @test lohner(0.0, 1.0, 0.0, 1.0) > 0.5
end

@testset "The indicator falls as the mesh refines" begin
    # This is what makes refinement terminate: for smooth data τ shrinks
    # with h, so a fixed threshold is eventually satisfied everywhere. A
    # criterion without this property refines until it hits a cap.
    τs = Float64[]
    for N in (8, 16, 32, 64)
        _, fs = pulse_field(N)
        scales = field_scales(fs)
        push!(τs, maximum(b -> cell_indicator(fs, b, Inf; scales=scales)[1],
                          1:nblocks(fs)))
    end
    @test all(τs[i] > τs[i + 1] for i in 1:(length(τs) - 1))
end

@testset "Well-resolved data is not refined" begin
    # The other side of the same coin: once the mesh resolves the pulse, no
    # block asks for more.
    _, fs = pulse_field(64)
    flags = refine_flags(fs; refine_tol=0.30, coarsen_tol=0.075, maxlevel_cap=4)
    @test !any(m -> (m isa Tuple ? m[1] : m) === Refine, flags)
end

@testset "The indicator sees every variable" begin
    # A system can be under-resolved in one variable while another looks
    # smooth -- for the travelling pulse ∂ₜu carries an order of magnitude
    # more amplitude than u. Reading variable 1 alone, as the criterion
    # this replaced did, misses it entirely.
    forest = Forest((4,); N=64, periodic=(true,), extents=((0.0, 1.0),))
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(1))
    # Variable 1: one wave over the domain, richly resolved. Variable 2:
    # twelve waves, at barely two cells each.
    fill_by_coordinates!((x, v) -> sinpi(2 * (v == 1 ? 1 : 12) * x[1]), fs)
    fill_ghosts!(fs, GhostSchedule(fs, REFOPS))

    both = maximum(b -> cell_indicator(fs, b, Inf;
                                       scales=field_scales(fs))[1], 1:nblocks(fs))
    ualone = maximum(b -> cell_indicator(fs, b, Inf; vars=1:1,
                                         scales=field_scales(fs; vars=1:1))[1],
                     1:nblocks(fs))
    @test both > ualone

    tol = (refine_tol=0.30, coarsen_tol=0.075, maxlevel_cap=2)
    fired(flags) = any(m -> (m isa Tuple ? m[1] : m) === Refine, flags)
    @test fired(refine_flags(fs; tol...))
    @test !fired(refine_flags(fs; vars=1:1, tol...))
end

@testset "Refinement terminates without the cap binding" begin
    # The point of a resolution criterion: the depth is an output. With the
    # cap set far above what the data needs, the indicator alone must be
    # what stops the cascade -- otherwise `maxlevel_cap` is just the old
    # hardcoded target wearing a different name.
    forest = Forest((8,); N=8, periodic=(true,), extents=((0.0, 1.0),))
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(1))
    initial = pulse_exact(1, 1.0, 0.25, 0.08, 0.0)
    fill_by_coordinates!(initial, fs)
    scales = field_scales(fs)
    flag(b, k) = refine_mark(fs, b, k; scales=scales, refine_tol=0.30,
                             coarsen_tol=0.075, maxlevel_cap=6)
    _, _, converged = adapt_to_initial_data!(fs, REFOPS; initial=initial,
                                             flag=flag, buffer=2, maxpasses=12)
    @test converged
    @test maxlevel(forest) == 2                    # not 6
    @test nleaves(forest) * 8 < 8 * 8 * 4          # localized, not uniform level 2
end

@testset "A converged hierarchy is stable" begin
    # The direct test of the hysteresis dead band. With one threshold
    # instead of two, blocks sitting at the boundary refine and coarsen on
    # alternate regrids and this never settles.
    forest = Forest((8,); N=8, periodic=(true,), extents=((0.0, 1.0),))
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(1))
    initial = pulse_exact(1, 1.0, 0.25, 0.08, 0.0)
    fill_by_coordinates!(initial, fs)
    scales = field_scales(fs)
    flag(b, k) = refine_mark(fs, b, k; scales=scales, refine_tol=0.30,
                             coarsen_tol=0.075, maxlevel_cap=2)
    schedule, _, _ = adapt_to_initial_data!(fs, REFOPS; initial=initial,
                                            flag=flag, buffer=2, maxpasses=12)

    fill_ghosts!(fs, schedule)
    flags = flag_blocks(flag, forest)
    @test !any(m -> (m isa Tuple ? m[1] : m) === Refine, flags)
    @test complete_marks(forest, flags; buffer=2) == forest.leaves
end

@testset "A block at the cap holds an equal-level margin" begin
    # The (Keep, box) case. A block refined to the cap has by construction
    # stopped being under-resolved, so it no longer asks to Refine -- but it
    # still holds the feature, and must still report a box, or the margin
    # that travels with the feature would not exist. Keying the buffer on
    # Refine alone cannot express this.
    forest = Forest((8,); N=8, periodic=(true,), extents=((0.0, 1.0),))
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(1))
    initial = pulse_exact(1, 1.0, 0.25, 0.08, 0.0)
    fill_by_coordinates!(initial, fs)
    scales = field_scales(fs)
    flag(b, k) = refine_mark(fs, b, k; scales=scales, refine_tol=0.30,
                             coarsen_tol=0.075, maxlevel_cap=2)
    schedule, _, _ = adapt_to_initial_data!(fs, REFOPS; initial=initial,
                                            flag=flag, buffer=2, maxpasses=12)
    fill_ghosts!(fs, schedule)
    flags = flag_blocks(flag, forest)

    # Some block at the cap reports (Keep, box) rather than a bare Keep.
    atcap = [b for b in 1:nleaves(forest) if level(forest.leaves[b]) == 2]
    @test !isempty(atcap)
    @test any(b -> flags[b] isa Tuple && flags[b][1] === Keep, atcap)

    # And the margin does something: next to the feature, blocks that asked
    # to coarsen are held back.
    bare = [m isa Tuple ? m[1] : m for m in flags]
    @test count(==(Coarsen), buffered_flags(forest, flags, 4)) <
          count(==(Coarsen), bare)
end

@testset "The buffer width covers the feature's motion" begin
    # TreeAMR's measured guidance is that the margin must exceed the travel
    # per regrid interval, so the derivation rounds up and adds a cell.
    forest = Forest((8,); N=8, periodic=(true,), extents=((0.0, 1.0),))
    for chunk in (0.005, 0.01, 0.02)
        cells = refinement_buffer(forest, 2, chunk)
        @test cells > chunk / spacing(forest, 2)
        @test cells <= forest.N
    end

    # Recruitment reaches one ring, so a feature may not cross a whole
    # finest-level block between regrids. That is rejected by name, not
    # silently truncated.
    @test_throws ArgumentError refinement_buffer(forest, 2, 0.05)
    @test_throws ArgumentError refinement_buffer(forest, 6, 0.02)
end

@testset "The thresholds must leave a dead band" begin
    _, fs = pulse_field(8)
    k = blockkey(fs, 1)
    scales = field_scales(fs)
    @test_throws ArgumentError refine_mark(fs, 1, k; scales=scales,
                                           refine_tol=0.1, coarsen_tol=0.1,
                                           maxlevel_cap=2)
    @test_throws ArgumentError refine_mark(fs, 1, k; scales=scales,
                                           refine_tol=0.1, coarsen_tol=0.2,
                                           maxlevel_cap=2)
end
