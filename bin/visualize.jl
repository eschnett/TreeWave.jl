#!/usr/bin/env julia
#
# Visualize a TreeWave solution: the solution itself, its error against the
# exact answer, and the error norms over time.
#
#     julia --project=bin bin/visualize.jl
#     julia --project=bin bin/visualize.jl --case=pulse --n=2 --out=/tmp
#     julia --project=bin bin/visualize.jl --case=sine --ops=2
#
# The figures are written as PNGs and, when stdout is a terminal, also
# drawn inline via SixelTerm. Piped or redirected output skips the inline
# draw rather than spraying sixel escapes into a log; `--display` and
# `--no-display` override the detection either way.
#
# One figure per case, three stacked panels:
#
#   1. the solution, drawn one line per block and colored by refinement
#      level, with the exact solution overlaid and block boundaries marked
#      -- seeing *where* the coarse-fine interfaces are is the point;
#   2. the pointwise error in `u` at the same points, same colouring;
#   3. the volume-weighted L2 and L∞ error norms against time -- taken over
#      the *whole* state, `u` and `∂ₜu` together, so that they are the same
#      quantity the tests assert on. They therefore sit well above panel 2:
#      for the sine mode `∂ₜu` has amplitude ω, and its error dominates.
#
# The runs come from `wave_errors` and `track_pulse` via their `observer`
# keyword, so this script contains no time-stepping loop of its own. See
# `CODE.md`.

using CairoMakie
using Printf
using SixelTerm
using TreeAMR
using TreeWave

CairoMakie.activate!(; type="png", px_per_unit=2)

const LEVELCOLORS = Makie.wong_colors()

levelcolor(lvl) = LEVELCOLORS[mod1(lvl + 1, length(LEVELCOLORS))]

"""
One frame of a run: the per-block solution in 1D, and the global error
norms. Built inside the `observer` callback, where `fs` is scattered from
`u` and still describes the mesh `u` was computed on -- after a regrid it
would not.
"""
function snapshot(fs, t, u, exactf)
    forest = fs.forest
    G, N = forest.G, forest.N

    blocks = map(1:nblocks(fs)) do b
        k = blockkey(fs, b)
        # `cell_center` indexes the *stored* array, so interior cell i is
        # at index i + G.
        x = [cell_center(forest, k, (i + G,))[1] for i in 1:N]
        num = collect(interiorview(fs, b, 1))
        exact = [exactf((xi,), 1) for xi in x]
        (x=x, u=num, exact=exact, err=num .- exact, lvl=level(k),
         ext=block_extent(forest, k)[1])
    end

    # Norms are taken over the whole state vector -- both u and ∂ₜu -- so
    # they are the same quantity the tests assert on.
    exactfs = FieldSet(forest, 2)
    fill_by_coordinates!(exactf, exactfs)
    ue = statevector(exactfs)
    gather!(ue, exactfs)
    d = u .- ue

    return (t=t, blocks=blocks,
            l2=volume_weighted_norm(fs, d),
            linf=volume_weighted_norm(fs, d; p=Inf))
end

"""
Draw the last frame's solution and error, and the norm history, into one
figure. `exactf` is the exact solution at the last frame's time, drawn
densely so it reads as the continuum answer rather than as another mesh.
"""
function makefigure(snaps, exactf, L, title)
    last = snaps[end]
    levels = sort(unique(b.lvl for b in last.blocks))

    fig = Figure(; size=(1000, 900))
    Label(fig[0, 1], title; fontsize=18, font=:bold, tellwidth=false)

    ax1 = Axis(fig[1, 1]; ylabel="u",
               title=@sprintf("solution at t = %.4f  (%d blocks, %d cells)",
                              last.t, length(last.blocks),
                              sum(length(b.x) for b in last.blocks)))
    ax2 = Axis(fig[2, 1]; ylabel="u − u_exact",
               title="pointwise error in u at the same time")
    ax3 = Axis(fig[3, 1]; xlabel="t", ylabel="‖error‖",
               yscale=log10,
               title="volume-weighted error norms over the full state (u, ∂ₜu)")

    # Block boundaries: one vertical rule per interface.
    edges = unique(vcat([b.ext[1] for b in last.blocks],
                        [b.ext[2] for b in last.blocks]))
    for ax in (ax1, ax2)
        vlines!(ax, edges; color=(:gray, 0.3), linewidth=0.5)
    end

    xs = range(0, L; length=4000)
    lines!(ax1, xs, [exactf((x,), 1) for x in xs];
           color=(:black, 0.55), linewidth=1, linestyle=:dash)

    # Small enough to see individual cells?  Then show them.
    ncells = sum(length(b.x) for b in last.blocks)
    for b in last.blocks
        c = levelcolor(b.lvl)
        lines!(ax1, b.x, b.u; color=c, linewidth=1.6)
        lines!(ax2, b.x, b.err; color=c, linewidth=1.6)
        if ncells <= 400
            # Qualified: TreeAMR exports a `scatter!` of its own, which
            # moves a state vector into the working array.
            CairoMakie.scatter!(ax1, b.x, b.u; color=c, markersize=3.5)
            CairoMakie.scatter!(ax2, b.x, b.err; color=c, markersize=3.5)
        end
    end
    hlines!(ax2, [0.0]; color=(:black, 0.4), linewidth=0.5)

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
    Legend(fig[3, 2], [l2, li], ["L2", "L∞"]; framevisible=false,
           tellheight=false, valign=:top)

    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1; grid=false)
    return fig
end

"""
The standing sine mode on a static two-level mesh. The solution should lie
on top of the exact one, and the norms should oscillate about a level rather
than grow -- a standing mode neither grows nor decays.

What the error panel shows depends on `ops_order`, and that is the point of
having it as a knob. At order 4 the error is a smooth phase error that runs
*continuously* across both coarse-fine interfaces: the interfaces are
invisible, which is exactly what TreeAMR's interface-order rule buys. At
order 2 the same run develops a visible kink at each interface, because the
prolongated ghosts are then only 2nd-order accurate and the Laplacian
divides that error by h². Run both to see it:

    julia --project=bin bin/visualize.jl --case=sine --ops=2

`periods` is chosen to avoid two special phases. At a whole number of
periods the numerical and exact `u` agree to `O(h⁴)` rather than `O(h²)`:
the error is almost purely a frequency shift `ω̃ = ω(1 + O(h²))`, and
`cos(ω̃T)` differs from `cos(2π)` only at *second* order in that shift, so
sampling there would flatter the scheme. At a quarter or three-quarter
period `cos(ωT) = 0`, so `u` itself vanishes and the solution panel would
show nothing but the error again.
"""
function sinecase(; D=1, N=16, L=1.0, m=1, roots=4, periods=0.9,
                  ops_order=4)
    # G is set by the operator order, not chosen independently: TreeAMR
    # requires G >= prolongation/2 for point-value operators.
    G = ops_order ÷ 2
    snaps = []
    observer = (fs, t, u) -> push!(snaps, snapshot(fs, t, u,
                                                  wave_exact(D, L, m, t)))
    r = wave_errors(Val(D); N=N, G=G, roots=roots, L=L, m=m, periods=periods,
                    ops=Operators(prolongation=ops_order,
                                  restriction=ops_order),
                    observer=observer, nsnapshots=97)
    title = @sprintf("Standing sine mode, m = %d — two-level mesh, order-%d \
                      operators\nfinal L2 = %.3g, L∞ = %.3g at h = %.3g",
                     m, ops_order, r.l2, r.linf, r.h)
    return snaps, wave_exact(D, L, m, snaps[end].t), L, title
end

"""
The travelling pulse with a refined region that follows it. The refined
blocks should sit under the pulse, and the norms should show what each
regrid costs.
"""
function pulsecase(; D=1, N=8, L=1.0, roots=8, σ=0.08, x0=0.25, n=1,
                   t_end=0.5, chunk=0.02, threshold=1e-3, ops_order=4)
    G = ops_order ÷ 2
    snaps = []
    observer = (fs, t, u) -> push!(snaps, snapshot(fs, t, u,
                                    pulse_exact(D, L, x0, σ, t; n=n)))
    r = track_pulse(Val(D); N=N, G=G, roots=roots, L=L, σ=σ, x0=x0, n=n,
                    ops=Operators(prolongation=ops_order,
                                  restriction=ops_order),
                    t_end=t_end, chunk=chunk, threshold=threshold,
                    observer=observer)
    title = @sprintf("Travelling super-Gaussian pulse, n = %d, σ = %.3g — \
                      refinement tracks it\nworst L∞ = %.3g over the run, \
                      %d blocks at maxlevel %d",
                     n, σ, r.worst, r.nblocks, r.maxlevel)
    return snaps, pulse_exact(D, L, x0, σ, snaps[end].t; n=n), L, title
end

function main(args)
    case = "both"
    outdir = joinpath(@__DIR__, "output")
    n = 1
    dim = 1
    ops_order = 4
    # Sixel is for a human looking at a terminal; a pipe gets the paths.
    inline = stdout isa Base.TTY
    for a in args
        if startswith(a, "--case=")
            case = a[8:end]
        elseif startswith(a, "--out=")
            outdir = a[7:end]
        elseif startswith(a, "--n=")
            n = parse(Int, a[5:end])
        elseif startswith(a, "--ops=")
            ops_order = parse(Int, a[7:end])
        elseif startswith(a, "--dim=")
            dim = parse(Int, a[7:end])
        elseif a == "--display"
            inline = true
        elseif a == "--no-display"
            inline = false
        else
            error("unknown argument $a; expected --case=, --out=, --n=, \
                   --ops=, --dim=, --display, --no-display")
        end
    end
    case in ("both", "sine", "pulse") ||
        error("--case must be sine, pulse, or both; got $case")
    # The panels draw one line per block against x, which only means
    # anything in 1D. Higher D would need a different figure, not a
    # different argument.
    dim == 1 || error("only --dim=1 is implemented; got $dim")

    mkpath(outdir)
    written = String[]
    for (name, build) in (("sine", () -> sinecase(; D=dim,
                                                  ops_order=ops_order)),
                          ("pulse", () -> pulsecase(; D=dim, n=n,
                                                    ops_order=ops_order)))
        (case == "both" || case == name) || continue
        @info "running the $name case"
        snaps, exactf, L, title = build()
        fig = makefigure(snaps, exactf, L, title)
        path = joinpath(outdir, "$(name)_$(dim)d.png")
        save(path, fig)
        inline && display(fig)
        push!(written, path)
    end
    for p in written
        println("wrote $p")
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
