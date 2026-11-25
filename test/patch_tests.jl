# The wave equation on a forest of patches (TreeAMR's M20, step 3): the
# Laplacian through TreeMultiPatch's metric, and the rotating Bessel mode
# on the four-patch annulus. See "Patches" in `CODE.md`.

using TreeMultiPatch: PolarAnnulus, SevenPatchBall, layout_forest, patch_frame,
                      volume_element
using TreeAMR: chart_map

# The largest difference at the owned points of every block between
# variable 2 of `du` and `exact(ξ, p)`.
function owned_error(fs::FieldSet{T,D}, du, exact) where {T,D}
    a = statearray(du, fs)
    worst = zero(T)
    for b in 1:nblocks(fs)
        p = patch(fs.forest, blockkey(fs, b))
        for I in CartesianIndices(ntuple(_ -> fs.forest.N, D))
            ξ = coordinates(fs, b, ntuple(d -> I[d] + fs.G[d], D))
            worst = max(worst, abs(a[I, 2, b] - exact(ξ, p)))
        end
    end
    return worst
end

@testset "The metric set holds the layout's √g and A^{ab} at every stored point" begin
    # The stencil reads the metric at the ghosts as well as at the owned
    # points, and the metric is filled there by a kernel of its own rather
    # than by TreeAMR's `fill_by_coordinates!`. A position off by a plane
    # or a ghost given the wrong patch would put every coefficient
    # somewhere else; compare them with the layout's own at TreeAMR's
    # `coordinates`, which index the stored array.
    layout = PolarAnnulus(0.5, 2.0, (2, 1))
    forest = annulus_forest(layout; N=4)
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(2),
                  parity=TreeWave.ANNULUS_PARITY)
    p = PatchWaveProblem(fs, nothing, layout)
    m = p.metric.work
    worst = 0.0
    for b in 1:nblocks(fs), I in CartesianIndices(size(m)[1:2])
        ξ = coordinates(fs, b, Tuple(I))
        q = patch(forest, blockkey(fs, b))
        J, S = patch_frame(layout, ξ, q)
        A = (sum(S[1] .* S[1]) / J, sum(S[1] .* S[2]) / J, sum(S[2] .* S[2]) / J)
        worst = max(worst, abs(m[I, 1, b] - J),
                    maximum(abs.(collect(m[I, 2:4, b]) .- A)))
    end
    @test worst < 1e-14
    # The largest logical speed is `1/(r θ′)` at the inner wall's ghost,
    # `r = r0 - r′/N`, and the time step divides by it at the owned points.
    @test maximum(m[:, :, 5, :]) ≈ 1 / ((0.5 - 1.5 / 4) * π / 4) rtol = 1e-14
    @test patch_timestep(p; cfl=1) ≈ (1 / 4) * 0.5 * π / 4 rtol = 1e-14
end

@testset "The patch Laplacian converges at second order, its cross terms included" begin
    # A smooth function filled at every stored point from its physical
    # position, so that no ghost exchange is involved: what is left is the
    # stencil and the metric. On the annulus the cross terms vanish; on the
    # seven-patch ball's shells they do not, and a cross term with the
    # wrong sign, index or factor leaves an error that does not fall with
    # `h`.
    kx = (0.7, -0.4, 0.5)
    for (name, layout, D) in (("annulus", PolarAnnulus(0.5, 2.0, (2, 1)), 2),
                              ("seven-patch ball", SevenPatchBall(0.4, 1.0, (1, 1)), 3))
        k² = sum(abs2, kx[1:D])
        f(ξ, q) = sin(sum(kx[d] * chart_map(layout, ξ, q)[d] for d in 1:D))
        errs = map((4, 8, 16)) do N
            forest = D == 2 ? annulus_forest(layout; N=N) : layout_forest(layout; N=N)
            fs = FieldSet(forest, 2; G=1, centering=vertexcentered(D),
                          parity=D == 2 ? TreeWave.ANNULUS_PARITY : nothing)
            p = PatchWaveProblem(fs, nothing, layout)
            fill_stored!((ξ, q) -> (f(ξ, q), 0.0), fs, block_patches(forest))
            du = statevector(fs)
            TreeWave.patch_rhs!(du, p)
            owned_error(fs, du, (ξ, q) -> -k² * f(ξ, q))
        end
        # Measured: 1.98 and 2.00 on the annulus, 1.82 and 1.83 on the ball,
        # whose rate reaches 1.98 from `N = 16` to 32.
        rates = log2.(errs[1:2] ./ errs[2:3])
        @test all(r -> 1.7 < r < 2.3, rates)
    end
end

@testset "The rotating Bessel mode converges at second order on the annulus" begin
    # The mode turns a quarter of the way round, so every point of it
    # crosses a glued face, and it has `∂_r u = 0` on both walls. A glue,
    # a metric or a wall that is wrong to `O(1)` breaks the rate; the
    # bounds are the measured rates (CODE.md, "Measured results") with
    # room either side.
    for (m, n, r0, roots) in ((1, 1, 1, (2, 1)), (3, 2, 0.5, (3, 2)))
        rs = [annulus_errors(; N=N, m=m, n=n, r0=r0, roots=roots) for N in (8, 16, 32)]
        @test 1.9 < convergence_rate([r.h for r in rs], [r.l2 for r in rs]) < 2.2
        @test 1.8 < convergence_rate([r.h for r in rs], [r.linf for r in rs]) < 2.2
    end
end

@testset "The wave equation on patches refuses a set that is not vertex-centred" begin
    # A glued face is a vertex plane, and the metric is placed at
    # vertices; a cell-centred set would read it half a cell off.
    layout = PolarAnnulus(1.0, 2.0, (2, 1))
    forest = annulus_forest(layout; N=4)
    fs = FieldSet(forest, 2; G=1, centering=cellcentered(2),
                  parity=TreeWave.ANNULUS_PARITY)
    @test_throws "vertex-centred only" PatchWaveProblem(fs, nothing, layout)
end
