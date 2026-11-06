# The 1D viewer's record and figure: what `visualize.jl` keeps of a run,
# and how it is drawn. Included by that script after its `using TreeAMR,
# TreeWave`; nothing in here changes how a run is computed.
#
# One figure per case, four stacked panels:
#
#   1. the solution, drawn one line per block and colored by refinement
#      level, with the exact solution overlaid and block boundaries marked
#      -- seeing *where* the coarse-fine interfaces are is the point;
#   2. the pointwise error in `u` at the same points, same colouring;
#   3. the refinement indicator τ per cell, with the refine and coarsen
#      thresholds drawn, so the mesh's own decision reads off the figure;
#   4. the volume-weighted L2 and L∞ error norms against time -- taken over
#      the *whole* state, `u` and `∂ₜu` together, so that they are the same
#      quantity the tests assert on. They therefore sit well above panel 2:
#      for the sine mode `∂ₜu` has amplitude ω, and its error dominates.
#
# The first three share an x axis; the fourth is against time.

using CairoMakie
using Printf
using SixelTerm

CairoMakie.activate!(; type="png", px_per_unit=2)

const LEVELCOLORS = Makie.wong_colors()

levelcolor(lvl) = LEVELCOLORS[mod1(lvl + 1, length(LEVELCOLORS))]

"""
The refinement indicator per owned point of one block, in 1D -- the
per-point quantity `cell_indicator` reduces to a single verdict. Recomputed
here rather than returned by the criterion, because the criterion only ever
needs the maximum and the bounding box.
"""
function block_taus(fs, b, scales; vars=1:fs.nvars,
                    ε=oftype(float(first(scales)), 1//100))
    forest = fs.forest
    G, N = fs.G[1], forest.N
    ws = [blockview(fs, b, v) for v in vars]
    return [maximum(lohner(w[i - 1], w[i], w[i + 1], s; ε=ε)
                    for (w, s) in zip(ws, scales))
            for i in (G + 1):(G + N)]
end

"""
    snapshot(fs, t, exact; scales)

One frame of a run: the per-block solution in 1D, the indicator τ, and the
global error norms against the exact solution `exact` at time `t`. The
run calls it with `fs` holding the state at `t`, its ghosts filled, and
before any regrid -- after one, `fs` would describe another mesh.

`scales` are the run's own amplitude scales, so the τ drawn is the value
the criterion acts on and not an approximation of it.

This is also where the run's own floating-point type stops: a run may be
`Float32`, and Makie is happiest given `Float64`, so every array that
reaches the figure is converted here rather than at a dozen plot calls. The
*evaluation* above it stays in the run's type -- `exact` is handed `x` as
the mesh produced it.

And it is where the run's *storage* stops, for the same reason: a run may
be on a device, every line below reads single cells, and Makie is host
code by nature. One `hostcopy` here buys that and nothing else in the
figure code has to know.
"""
function snapshot(fs, t, exact; scales)
    fs = hostcopy(fs)
    forest = fs.forest
    G, N = fs.G[1], forest.N

    blocks = map(1:nblocks(fs)) do b
        k = blockkey(fs, b)
        # `coordinates` indexes the *stored* array, so owned point i is at
        # index i + G. It takes the field set rather than the forest
        # because where a point sits depends on the ghost width and the
        # centering, both of which live there: a cell centre is half a
        # spacing inside the block, a vertex is on its low edge.
        x = [coordinates(fs, b, (i + G,))[1] for i in 1:N]
        num = collect(interiorview(fs, b, 1))
        ex = [exact((xi,), 1) for xi in x]
        ext = block_extent(forest, k)[1]
        (x=Float64.(x), u=Float64.(num), exact=Float64.(ex),
         err=Float64.(num .- ex), lvl=level(k),
         tau=Float64.(block_taus(fs, b, scales)),
         ext=(Float64(ext[1]), Float64(ext[2])))
    end

    # Norms are taken over the whole state vector -- both u and ∂ₜu -- so
    # they are the same quantity the tests assert on.
    u = statevector(fs)
    gather!(u, fs)
    exactfs = FieldSet(forest, 2; G=fs.G, centering=fs.centering)
    fill_by_coordinates!(exact, exactfs)
    ue = statevector(exactfs)
    gather!(ue, exactfs)
    d = u .- ue

    return (t=Float64(t), blocks=blocks,
            l2=Float64(volume_weighted_norm(fs, d)),
            linf=Float64(volume_weighted_norm(fs, d; p=Inf)))
end

"""
Draw the last frame's solution and error, and the norm history, into one
figure. `exact` is the exact solution at the last frame's time, drawn
densely so it reads as the continuum answer rather than as another mesh.
"""
function makefigure(snaps, exact, L, title; tols, steering)
    last = snaps[end]
    levels = sort(unique(b.lvl for b in last.blocks))

    fig = Figure(; size=(1000, 1150))
    Label(fig[0, 1], title; fontsize=18, font=:bold, tellwidth=false)

    ax1 = Axis(fig[1, 1]; ylabel="u",
               title=@sprintf("solution at t = %.4f  (%d blocks, %d cells)",
                              last.t, length(last.blocks),
                              sum(length(b.x) for b in last.blocks)))
    ax2 = Axis(fig[2, 1]; ylabel="u − u_exact",
               title="pointwise error in u at the same time")
    ax3 = Axis(fig[4, 1]; xlabel="t", ylabel="‖error‖",
               yscale=log10,
               title="volume-weighted error norms over the full state (u, ∂ₜu)")
    ax4 = Axis(fig[3, 1]; ylabel="τ",
               title=steering ?
                     "refinement indicator τ — these thresholds steered this mesh" :
                     "refinement indicator τ — thresholds shown for scale (static mesh)")

    # Block boundaries: one vertical rule per interface.
    edges = unique(vcat([b.ext[1] for b in last.blocks],
                        [b.ext[2] for b in last.blocks]))
    for ax in (ax1, ax2, ax4)
        vlines!(ax, edges; color=(:gray, 0.3), linewidth=0.5)
    end

    xs = range(zero(L), L; length=4000)
    lines!(ax1, Float64.(xs), [Float64(exact((x,), 1)) for x in xs];
           color=(:black, 0.55), linewidth=1, linestyle=:dash)

    # Small enough to see individual cells?  Then show them.
    ncells = sum(length(b.x) for b in last.blocks)
    for b in last.blocks
        c = levelcolor(b.lvl)
        lines!(ax1, b.x, b.u; color=c, linewidth=1.6)
        lines!(ax2, b.x, b.err; color=c, linewidth=1.6)
        lines!(ax4, b.x, b.tau; color=c, linewidth=1.6)
        if ncells <= 400
            # Qualified: TreeAMR exports a `scatter!` of its own, which
            # moves a state vector into the working array.
            CairoMakie.scatter!(ax1, b.x, b.u; color=c, markersize=3.5)
            CairoMakie.scatter!(ax2, b.x, b.err; color=c, markersize=3.5)
        end
    end
    hlines!(ax2, [0.0]; color=(:black, 0.4), linewidth=0.5)

    # Where τ crosses the upper rule is where the mesh went finer; a block
    # every one of whose cells is under the lower rule may coarsen.
    hlines!(ax4, [tols.refine_tol]; color=(:firebrick, 0.9), linewidth=1.2)
    hlines!(ax4, [tols.coarsen_tol]; color=(:steelblue, 0.9), linewidth=1.2,
            linestyle=:dash)
    text!(ax4, 0.004, tols.refine_tol; text=" refine", align=(:left, :bottom),
          color=:firebrick, fontsize=11)
    text!(ax4, 0.004, tols.coarsen_tol; text=" coarsen", align=(:left, :top),
          color=:steelblue, fontsize=11)

    Legend(fig[1, 2],
           [LineElement(; color=levelcolor(l), linewidth=2) for l in levels],
           ["level $l" for l in levels], "refinement";
           framevisible=false, tellheight=false, valign=:top)

    # An exact initial condition makes the error identically zero at t = 0,
    # which a log axis cannot draw; NaN breaks the line instead of failing.
    nz(v) = [x > 0 ? x : NaN for x in v]
    ts = [s.t for s in snaps]
    l2 = lines!(ax3, ts, nz([s.l2 for s in snaps]); linewidth=2)
    li = lines!(ax3, ts, nz([s.linf for s in snaps]); linewidth=2,
                linestyle=:dash)
    Legend(fig[4, 2], [l2, li], ["L2", "L∞"]; framevisible=false,
           tellheight=false, valign=:top)

    linkxaxes!(ax1, ax2, ax4)
    hidexdecorations!(ax1; grid=false)
    hidexdecorations!(ax2; grid=false)
    ax4.xlabel = "x"
    return fig
end

"""
The sine mode's figure, from what `sine` returned and the options it ran
under. The mesh is static, so the thresholds on the τ panel are drawn for
scale only.
"""
function sine_figure(r, o)
    last = r.snaps[end]
    title = @sprintf("Standing sine mode, m = %d — two-level mesh, order-%d \
                      operators, %s, %s\nfinal L2 = %.3g, L∞ = %.3g at h = %.3g",
                     r.m, o.ops_order, o.T, centeringname(o.centering),
                     last.l2, last.linf, r.h)
    return makefigure(r.snaps, r.exact, r.L, title;
                      tols=(refine_tol=0.30, coarsen_tol=0.075),
                      steering=false)
end

"""
The pulse's figure, from what `pulse` returned and the options it ran
under. The block count and the depth are the last frame's, the mesh the
figure draws.
"""
function pulse_figure(r, o)
    last = r.snaps[end]
    title = @sprintf("Travelling super-Gaussian pulse, n = %d, σ = %.3g, %s, %s \
                      — refinement tracks it\nworst L∞ = %.3g over the run, \
                      %d blocks at maxlevel %d",
                     o.n, r.σ, o.T, centeringname(o.centering),
                     maximum(s.linf for s in r.snaps), length(last.blocks),
                     maximum(b.lvl for b in last.blocks))
    return makefigure(r.snaps, r.exact, r.L, title;
                      tols=(refine_tol=r.refine_tol,
                            coarsen_tol=r.coarsen_tol),
                      steering=true)
end
