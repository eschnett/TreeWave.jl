# The wave equation on the seven-patch ball (TreeAMR's M20, step 5): the
# ghosts across every glue interpolated, every glued face point evolved by
# each patch holding it, and Kreiss–Oliger dissipation for the corners
# where three patches meet; and on the ball reduced by a symmetry
# (TreeAMR's M21, step 6). See "The seven-patch ball" in `CODE.md`.

using TreeMultiPatch: SevenPatchBall, layout_forest
using TreeAMR: chart_map

@testset "The scheme's quadrature gives the ball's volume at second order" begin
    # Each patch's trapezoidal rule in its logical coordinates, a shared
    # face point split between its patches and the outer sphere's plane
    # included. A point counted twice, or a face weighted as an interior
    # point, leaves an error that does not fall with `h`.
    errs = map((8, 16)) do N
        p, _ = ball_problem(; N=N, boundary=_ -> TreeWave.zero_hook)
        patch_volume(p) / (4π * 27 / 3) - 1
    end
    # Measured: -6.0e-3 and -1.5e-3, rate 2.00 (and 2.00 again to N = 64).
    @test -7e-3 < errs[1] < -5e-3
    @test 3.9 < errs[1] / errs[2] < 4.1
end

@testset "The plane wave converges at second order on the seven-patch ball" begin
    # The wave crosses every glue, cube to shell and shell to shell, and
    # its patch corners. Copied ghosts would not converge at all across
    # the kinks; an interpolation of too low an order, a ghost located in
    # the wrong patch or an appendix plane left unevolved breaks the rate,
    # in L2 and near the glues. Without dissipation the scheme's own
    # order: measured L2 rates 2.02 and 2.01, L∞ near the glues 2.21 and
    # 2.16.
    rs = [ball_errors(; N=N, dissipation=0) for N in (8, 16, 32)]
    hs = [r.h for r in rs]
    @test 1.9 < convergence_rate(hs, [r.l2 for r in rs]) < 2.2
    @test 1.9 < convergence_rate(hs, [r.linf_near for r in rs]) < 2.4
    # With the corners' dissipation, `ε = 1/2`, whose own error is third
    # order and dominates at these sizes: measured L2 5.2 and L∞ near the
    # glues 6.4 times smaller from `N = 8` to 16 (6.0 and 6.4 from 16 to
    # 32, 5.3 and 4.5 from 32 to 64).
    r8, r16 = ball_errors(; N=8), ball_errors(; N=16)
    @test r8.l2 / r16.l2 > 4
    @test r8.linf_near / r16.linf_near > 4
end

@testset "Data and its image under inversion evolve into each other's images" begin
    # Point inversion maps the seven-patch ball onto itself, patch to
    # patch and node to node, and closed ownership evolves every copy of a
    # face point by its own patch, so the run must commute with it to
    # rounding. A ghost interpolated from a stencil placed asymmetrically,
    # or a face point evolved by one patch only, breaks the symmetry at
    # the scheme's error.
    r = ball_asymmetry(; N=8, symmetry=:inversion, tend=2)
    @test r.difference < 1e-14 * r.scale
end

@testset "Data and its mirror image evolve into each other's images" begin
    # The mirror `x₁ → -x₁` maps the ball onto itself as inversion does,
    # but turns the shells' quarter-turn glues the other way. Under closed
    # ownership each copy of a point is still evolved by its own patch, so
    # the measured difference is rounding (TreeAMR's prototype found the
    # same, `HISTORY.md`, M20); half-open ownership would leave the
    # scheme's error here.
    r = ball_asymmetry(; N=8, symmetry=:mirror, tend=2)
    @test r.difference < 1e-14 * r.scale
end

@testset "Data and its quarter-turned image evolve into each other's images" begin
    # The quarter turn about z maps the ball onto itself too, but a point on
    # |x| = |y| between two shells, where the shells' diagonal corner ghosts
    # lie, is interpolated from the shell TreeMultiPatch's `ties` names, and
    # the two shells' copies of their face differ by the scheme's error. So
    # the run commutes with the turn to rounding under `ties = :turn`, and
    # not under the default `:mirror` (measured 1e-15 and 3e-2, of 1.1),
    # which is what made the quarter-turned ball drift from the full one.
    r = ball_asymmetry(; N=8, symmetry=:turn, tend=2, ties=:turn)
    @test r.difference < 1e-14 * r.scale
    r = ball_asymmetry(; N=8, symmetry=:turn, tend=2, ties=:mirror)
    @test r.difference > 1e-3 * r.scale
end

@testset "Each reduced ball's run is the full ball's, restricted: $sym" for
        sym in (:bitant, :quadrant, :octant, :quarter, :half, :quarter_bitant, :half_bitant)
    # TreeAMR's M21 on the wave equation: the plane wave summed over the
    # symmetry's images, run on the reduced ball (mirror faces, turned
    # seams, ghosts folded across both beside the kinked glues) and on its
    # full ball, which takes the tie rule the symmetry keeps. A mirrored or
    # turned copy with the wrong sign or axes, a fold to the wrong image, or
    # a seam plane evolved with another stencil than the full ball's
    # interior one, shows as the scheme's error and grows; measured 5e-15
    # to 6e-15 of the scale over t = 2.
    r = ball_symmetric_difference(; N=8, symmetry=sym, tend=2)
    @test r.difference < 2e-14 * r.scale
end

@testset "The closed ball's energy stays bounded at the corners' dissipation" begin
    # Random data, `u = v = 0` beyond the outer sphere, many crossing
    # times. Where three patches meet the scheme has a spatial
    # instability that the dissipation `ε = 1/2` damps (TreeAMR's
    # prototype); an appendix plane evolved with the wrong stencil or a
    # dissipation of the wrong sign grows without bound.
    ts, Es = ball_energies(; N=8, tend=40, every=10)
    @test all(<=(Es[1]), Es)
    @test all(i -> Es[i + 1] <= Es[i], 2:length(Es) - 1)
end

@testset "A reduced ball's energy stays bounded too: $sym" for sym in (:octant, :quarter_bitant)
    # As on the full ball, random data on the closed reduced ball, now with
    # mirror faces and a turned seam beside the cube's corners (TreeAMR's
    # M21). A seam plane evolved with the wrong stencil, or a mirrored or
    # turned ghost of the wrong sign, feeds the corners' instability. The
    # two smallest reductions, one with mirrors only and one with the
    # turn; measured, every reduction's energy falls at every sample, to
    # about 1e-4 of its start by t = 40.
    ts, Es = ball_energies(; N=8, tend=40, every=10, roots=(2, 1), symmetry=sym)
    @test all(<=(Es[1]), Es)
    @test all(i -> Es[i + 1] <= Es[i], 2:length(Es) - 1)
end

@testset "Dissipation without the ghosts it reads is refused" begin
    # The fourth differences read two points out.
    layout = SevenPatchBall(1.0, 3.0, (1, 1))
    forest = layout_forest(layout; N=8)
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(3), glue=:interpolate)
    @test_throws "needs G ≥ 2" PatchWaveProblem(fs, nothing, layout; dissipation=0.5)
end
