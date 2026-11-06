# The 2D viewer's record and figure: what `visualize2d.jl` keeps of a run,
# and how it is drawn. Included by that script after its `using TreeAMR,
# TreeWave`; nothing in here changes how a run is computed.
#
# One figure, three parts:
#
#   1. a filmstrip of `u` at four times, one heatmap per block with the
#      block boundaries drawn and coloured by refinement level. The ring
#      expands, the refined annulus travels with it, and the middle
#      coarsens back once the ring has left -- this is the only case in
#      the package where coarsening does any work, so it is the panel
#      that has to show it;
#   2. `u` against radius for every cell in the final frame. This is the
#      standard Sedov diagnostic and it is sharper than it looks: the
#      exact solution is a function of `r` alone, so every cell of a
#      perfect solution lands on one curve. Vertical spread is the error
#      the Cartesian mesh introduces by not being radial, and an artifact
#      at a coarse-fine interface shows up as a band rather than a curve.
#      Read it only to the left of the `L/2` rule: past that the ring is
#      close enough to its own periodic images that the *exact* solution
#      stops being radial too, so the fan out there is the torus and not
#      the mesh;
#   3. the block count and the worst indicator τ against time, with the
#      refine and coarsen thresholds drawn. The mesh growing while τ sits
#      *below* the refine rule is the statement that the criterion chose
#      level 2 and stopped there on its own.

using CairoMakie
# `Relative` is also TreeAMR's boundary hook (since 0.2.2); here it is
# Makie's layout size.
using CairoMakie: Relative
using KernelAbstractions: get_backend
using Printf
using SixelTerm

# `type="png"` is not just a default worth keeping. CairoMakie's fast
# image path -- the one that pads each heatmap's edges so that abutting
# blocks leave no hairline seam -- is disabled for vector output, where
# every heatmap instead degrades into one polygon per cell. A 772-block
# mesh has 49408 of those.
CairoMakie.activate!(; type="png", px_per_unit=2)

const LEVELCOLORS = Makie.wong_colors()

levelcolor(lvl) = LEVELCOLORS[mod1(lvl + 1, length(LEVELCOLORS))]

"""
    snapshot(fs, t, exact; scales, coarsen_tol)

One frame of a run: every block's `u`, its extent and its level, plus the
scalars the time series and the title need -- the worst indicator τ, the
L∞ error against the exact solution `exact` at time `t`, and how much of
the ring sits at the finest level ([`blast_coverage`](@ref)). The run
calls it with `fs` holding the state at `t`, its ghosts filled, and before
any regrid -- after one, `fs` would describe another mesh.

`scales` are the ones the run is using, so the τ recorded here is the
value the criterion is about to flag on rather than an approximation of
it. That is why they are an argument and not a `field_scales` call inside
this function: under `--frozen-scales` recomputing them here would
quietly draw the τ the run *would* have seen, which is the one thing that
figure must not do.

The error and the coverage are measured first, on the run's own storage:
both are reductions over the whole mesh, which run where the data is, and
`exact` closes over arrays on the run's backend.

This is also where the run's own floating-point type stops. A run may be
`Float32`; everything below this point is a figure, and Makie is happiest
given `Float64`, so the conversion happens once here rather than at every
plot call.

And where its *storage* stops: a run may be on a device, and both
`interiorview` and `cell_indicator` read single cells. One `hostcopy`
here and the figure code below does not have to know.
"""
function snapshot(fs, t, exact; scales, coarsen_tol)
    linf = state_error(fs, exact)
    covered = blast_coverage(fs)

    fs = hostcopy(fs)
    forest = fs.forest
    blocks = map(1:nblocks(fs)) do b
        k = blockkey(fs, b)
        ext = block_extent(forest, k)
        # Where the samples actually are, asked of the field set rather
        # than reconstructed from the extent: the first owned point sits
        # half a spacing inside the block in a cell-centred dimension and
        # exactly on its low edge in a vertex-like one. Both panels below
        # derive their geometry from this pair and from nothing else, so
        # there is one place the centring is read and not two.
        firstpoint = coordinates(fs, b, ntuple(d -> fs.G[d] + 1, 2))
        # Not transposed: `interiorview` gives `z[i,j]` at `(x_i, y_j)`,
        # which is already `heatmap!`'s convention. Transposing it looks
        # almost right on a radially symmetric field, which is what makes
        # it worth saying.
        (ext=map(e -> (Float64(e[1]), Float64(e[2])), ext),
         first=map(Float64, firstpoint),
         h=Float64(spacing(forest, k)),
         u=Float64.(interiorview(fs, b, 1)), lvl=level(k))
    end
    τ = maximum(b -> cell_indicator(fs, b, coarsen_tol; scales=scales)[1],
                1:nblocks(fs))
    return (t=Float64(t), blocks=blocks, nblocks=nblocks(fs), τ=Float64(τ),
            linf=Float64(linf), covered=covered)
end

"""
The L∞ error of the state `fs` holds against `exact`, over the whole
state (`u` and `∂ₜu`) and on the run's own layout: the difference is
between two state vectors, so the exact solution has to be sampled where
the run is.
"""
function state_error(fs, exact)
    exactfs = FieldSet(fs.forest, fs.nvars; G=fs.G, centering=fs.centering,
                       backend=get_backend(fs.work))
    fill_by_coordinates!(exact, exactfs)
    u, ue = statevector(fs), statevector(exactfs)
    gather!(u, fs)
    gather!(ue, exactfs)
    return volume_weighted_norm(fs, u .- ue; p=Inf)
end

"""
One filmstrip panel: the field as one heatmap per block, then the block
boundaries on top.

The x and y arguments are **cell edges**, `N + 1` of them, and that is
load-bearing. A bare `(lo, hi)` tuple is read by `heatmap!` as the centres
of the first and last cell, which inflates every block by `w/2(N-1)` per
side -- 7% at `N = 8` -- so the blocks silently overlap and the mesh looks
subtly wrong rather than broken.

The edges are half a spacing either side of the *samples*, not the block's
outer edges, and the two are the same thing only when the block is
cell-centred. A vertex-centred block's `N` owned samples run from its low
edge to `(N-1)h` past it, so the cells drawn around them are shifted down
by `h/2`: they still tile exactly between same-level neighbours -- one
block's last half-cell meets the next one's first -- but they sit half a
spacing off the block outlines, which is what half-open ownership looks
like when it is drawn. Deriving the edges from `block_extent` instead
would put every vertex-centred sample half a cell from where it is.
"""
function fieldpanel!(ax, snap; colorrange, colorscale)
    local hm
    for b in snap.blocks
        N = size(b.u, 1)
        edges(d) = range(b.first[d] - b.h / 2, b.first[d] + (N - 0.5) * b.h;
                         length=N + 1)
        hm = heatmap!(ax, edges(1), edges(2), b.u;
                      colormap=:balance, colorrange=colorrange,
                      colorscale=colorscale)
    end
    blockoutlines!(ax, snap.blocks)
    return hm
end

"""
Block boundaries, one `lines!` per refinement level rather than one per
block: a `NaN` point breaks the path, so a level's 700-odd rectangles
become a single plot object. Makie spends about a millisecond building
each plot it is given, which at five panels is the difference between a
six-second figure and a twenty-second one.
"""
function blockoutlines!(ax, blocks)
    for lvl in sort(unique(b.lvl for b in blocks))
        pts = Point2f[]
        for b in blocks
            b.lvl == lvl || continue
            (x1, x2), (y1, y2) = b.ext
            append!(pts, (Point2f(x1, y1), Point2f(x2, y1), Point2f(x2, y2),
                          Point2f(x1, y2), Point2f(x1, y1),
                          Point2f(NaN, NaN)))
        end
        lines!(ax, pts; color=(levelcolor(lvl), 0.8), linewidth=0.6)
    end
    return nothing
end

"""
Every interior cell of one frame as `(radius, u, colour)`, for the radial
panel. One `scatter!` of 50000 points costs a third of a second; 772 of
them would cost five times that for the same picture.

The radius is the **minimum-image** distance to the centre. The box is
periodic, and the plain distance would put the four corner regions on a
branch of their own out at `r = 0.71` -- an artifact of the coordinates,
drawn as if it were an artifact of the solution.
"""
function radialpoints(snap, L, x₀)
    rs, us, cs = Float64[], Float64[], RGBAf[]
    for b in snap.blocks
        N = size(b.u, 1)
        c = levelcolor(b.lvl)
        h = b.h
        # From the block's first owned point, which `snapshot` asked the
        # field set for. The old spelling, `ext[1] + (i - 1/2)h`, is the
        # cell-centred case written out, and on a vertex-centred run it
        # would put every point half a spacing too far out -- which in
        # this panel reads as a fan, that is, as exactly the error the
        # panel exists to show.
        for j in 1:N, i in 1:N
            x = b.first[1] + (i - 1) * h
            y = b.first[2] + (j - 1) * h
            dx = mod(x - x₀[1] + L / 2, L) - L / 2
            dy = mod(y - x₀[2] + L / 2, L) - L / 2
            push!(rs, hypot(dx, dy))
            push!(us, b.u[i, j])
            push!(cs, RGBAf(c, 0.35))
        end
    end
    return rs, us, cs
end

"""
Draw the filmstrip, the radial profile and the time series into one
figure. `profile` is the exact *free-space* radial solution at the last
frame's time, as `(r, u)` vectors.
"""
function makefigure(snaps, profile, L, x₀, title; tols, tracking)
    last = snaps[end]
    levels = sort(unique(b.lvl for b in last.blocks))

    # Four frames spanning the run, the first and the last included.
    frames = snaps[round.(Int, range(1, length(snaps); length=4))]

    # One symmetric range and one colorbar for all four frames, so that
    # the strip shows the ring *fading* and not merely moving -- a
    # per-frame range would renormalize the decay away. A symmetric range
    # puts u = 0 at the colormap's white centre; the symlog scale is what
    # keeps the late frames, ten times smaller than the first, from
    # reading as blank.
    amp = maximum(maximum(abs, b.u) for s in frames for b in s.blocks)
    crange = (-amp, amp)
    cscale = Makie.Symlog10(amp / 100)

    fig = Figure(; size=(1100, 1250))
    Label(fig[0, 1], title; fontsize=18, font=:bold, tellwidth=false)

    # A nested layout, so the strip's own gaps and its colorbar stay its
    # business and do not shift the two wide panels below.
    strip = GridLayout(fig[1, 1])
    local hm
    for (k, snap) in enumerate(frames)
        ax = Axis(strip[1, k]; aspect=DataAspect(), titlesize=11,
                  title=@sprintf("t = %.2f   %d blocks", snap.t,
                                 snap.nblocks))
        hm = fieldpanel!(ax, snap; colorrange=crange, colorscale=cscale)
        hidedecorations!(ax)
    end
    Colorbar(strip[1, 5], hm; label="u", width=12, height=Relative(0.92),
             ticklabelsize=9, labelsize=11)
    colgap!(strip, 6)

    ax_prof = Axis(fig[2, 1]; xlabel="r", ylabel="u",
                   title="u against radius, every cell of the final frame \
                          — one curve means the mesh kept the solution radial")
    rs, us, cs = radialpoints(last, L, x₀)
    CairoMakie.scatter!(ax_prof, rs, us; color=cs, markersize=1.5)

    # The exact curve is the free-space radial solution, and it is drawn
    # only out to L/2 -- on purpose. Past that the box's own periodicity
    # takes over: a cell at minimum-image radius r > L/2 is near a face or
    # a corner, where it feels the neighbouring images as well as the
    # source, and the solution there is genuinely not a function of r
    # alone. Spread to the right of the rule is the torus, not the mesh.
    keep = profile.rs .<= L / 2
    lines!(ax_prof, profile.rs[keep], profile.us[keep];
           color=(:black, 0.75), linewidth=1.2, linestyle=:dash)
    vlines!(ax_prof, [L / 2]; color=(:gray, 0.6), linewidth=1,
            linestyle=:dashdot)
    text!(ax_prof, L / 2, 0.0; text="  past here the ring meets its own \
                                      periodic images",
          align=(:left, :bottom), color=:gray, fontsize=10)

    ax_ts = Axis(fig[3, 1]; xlabel="t", ylabel="blocks",
                 title=tracking ?
                       "the refined region grows with the ring, while τ \
                        stays below the refine threshold" :
                       "with the scale frozen the mesh saturates at the \
                        whole domain within three regrids")
    ax_τ = Axis(fig[3, 1]; ylabel="τ", yaxisposition=:right)
    hidespines!(ax_τ)
    hidexdecorations!(ax_τ)
    linkxaxes!(ax_ts, ax_τ)
    ts = [s.t for s in snaps]
    nb = lines!(ax_ts, ts, [Float64(s.nblocks) for s in snaps]; linewidth=2)
    τl = lines!(ax_τ, ts, [s.τ for s in snaps]; linewidth=2,
                linestyle=:dash, color=:firebrick)
    ylims!(ax_τ, 0, 1)
    hlines!(ax_τ, [tols.refine_tol]; color=(:firebrick, 0.9), linewidth=1.2)
    hlines!(ax_τ, [tols.coarsen_tol]; color=(:steelblue, 0.9), linewidth=1.2,
            linestyle=:dot)
    # Labelled, because these two rules live on the right-hand axis and
    # would otherwise read as block counts on the left-hand one.
    text!(ax_τ, snaps[1].t, tols.refine_tol; text=" refine",
          align=(:left, :bottom), color=:firebrick, fontsize=11)
    text!(ax_τ, snaps[1].t, tols.coarsen_tol; text=" coarsen",
          align=(:left, :bottom), color=:steelblue, fontsize=11)

    Legend(fig[2, 2],
           [LineElement(; color=levelcolor(l), linewidth=2) for l in levels],
           ["level $l" for l in levels], "refinement";
           framevisible=false, tellheight=false, valign=:top)
    Legend(fig[3, 2], [nb, τl], ["blocks", "max τ"]; framevisible=false,
           tellheight=false, valign=:top)

    rowsize!(fig.layout, 1, Aspect(1, 0.25))
    rowgap!(fig.layout, 10)
    return fig
end

"""
The blast wave's figure, from what `blast` returned and the options it ran
under. The worst error and the coverage are taken over every frame, the
block count and the depth from the last one, and the growth is the last
frame's block count over the first's.

The exact curve in the radial panel is the free-space radial solution at
the final time, straight from the run's Hankel table rather than sampled
along a ray through the box: a ray would leave the domain past r = L/2
and the periodic image sum would answer with a *neighbouring* ring, which
looks like a second peak.
"""
function blast_figure(r, o)
    first, last = r.snaps[1], r.snaps[end]
    what = o.refresh_scales ? "refinement tracks it" :
           "amplitude scale frozen — the criterion refines everything"
    title = @sprintf("Radial blast wave, σ = %.3g, %s, %s — %s\nworst L∞ = %.3g, \
                      %d blocks at maxlevel %d (×%.1f), %.0f%% of the ring \
                      at the finest level",
                     r.σ, o.T, centeringname(o.centering), what,
                     maximum(s.linf for s in r.snaps), last.nblocks,
                     maximum(b.lvl for b in last.blocks),
                     last.nblocks / first.nblocks,
                     100 * minimum(s.covered for s in r.snaps))
    tab = blast_radial_table(r.reference, last.t)
    return makefigure(r.snaps, (rs=collect(r.reference.rs), us=tab.us),
                      Float64(r.L), map(Float64, r.x₀), title;
                      tols=(refine_tol=r.refine_tol,
                            coarsen_tol=r.coarsen_tol),
                      tracking=o.refresh_scales)
end
