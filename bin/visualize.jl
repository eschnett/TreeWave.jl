#!/usr/bin/env julia
#
# Visualize a TreeWave solution: the solution itself, its error against the
# exact answer, and the error norms over time.
#
#     julia --project=bin bin/visualize.jl
#     julia --project=bin bin/visualize.jl --case=pulse --n=2 --out=/tmp
#     julia --project=bin bin/visualize.jl --case=sine --ops=2
#     julia --project=bin bin/visualize.jl --type=f32
#     julia --project=bin bin/visualize.jl --centering=cell
#     julia --project=bin bin/visualize.jl --backend=metal --type=f32
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
#   3. the refinement indicator τ per cell, with the refine and coarsen
#      thresholds drawn, so the mesh's own decision reads off the figure;
#   4. the volume-weighted L2 and L∞ error norms against time -- taken over
#      the *whole* state, `u` and `∂ₜu` together, so that they are the same
#      quantity the tests assert on. They therefore sit well above panel 2:
#      for the sine mode `∂ₜu` has amplitude ω, and its error dominates.
#
# The first three share an x axis; the fourth is against time.
#
# The runs come from `wave_errors` and `track_pulse` via their `observer`
# keyword, so this script contains no time-stepping loop of its own. See
# `CODE.md`.

using CairoMakie
using Printf
using SixelTerm
using TreeAMR
using TreeWave

include(joinpath(@__DIR__, "backend.jl"))

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
One frame of a run: the per-block solution in 1D, and the global error
norms. Built inside the `observer` callback, where `fs` is scattered from
`u` and still describes the mesh `u` was computed on -- after a regrid it
would not.

This is also where the run's own floating-point type stops: a run may be
`Float32`, and Makie is happiest given `Float64`, so every array that
reaches the figure is converted here rather than at a dozen plot calls. The
*evaluation* above it stays in the run's type -- `exactf` is handed `x` as
the mesh produced it.

And it is where the run's *storage* stops, for the same reason: a run may
be on a device, every line below reads single cells, and Makie is host
code by nature. One `hostcopy` here buys that and nothing else in the
figure code has to know. `u` comes down with it, because the norms below
are taken against the host field set.
"""
function snapshot(fs, t, u, exactf)
    fs = hostcopy(fs)
    u = Array(u)
    forest = fs.forest
    G, N = fs.G[1], forest.N

    scales = field_scales(fs)
    blocks = map(1:nblocks(fs)) do b
        k = blockkey(fs, b)
        # `coordinates` indexes the *stored* array, so owned point i is at
        # index i + G. It takes the field set rather than the forest
        # because where a point sits depends on the ghost width and the
        # centering, both of which live there: a cell centre is half a
        # spacing inside the block, a vertex is on its low edge.
        x = [coordinates(fs, b, (i + G,))[1] for i in 1:N]
        num = collect(interiorview(fs, b, 1))
        exact = [exactf((xi,), 1) for xi in x]
        ext = block_extent(forest, k)[1]
        (x=Float64.(x), u=Float64.(num), exact=Float64.(exact),
         err=Float64.(num .- exact), lvl=level(k),
         tau=Float64.(block_taus(fs, b, scales)),
         ext=(Float64(ext[1]), Float64(ext[2])))
    end

    # Norms are taken over the whole state vector -- both u and ∂ₜu -- so
    # they are the same quantity the tests assert on.
    exactfs = FieldSet(forest, 2; G=fs.G, centering=fs.centering)
    fill_by_coordinates!(exactf, exactfs)
    ue = statevector(exactfs)
    gather!(ue, exactfs)
    d = u .- ue

    return (t=Float64(t), blocks=blocks,
            l2=Float64(volume_weighted_norm(fs, d)),
            linf=Float64(volume_weighted_norm(fs, d; p=Inf)))
end

"""
Draw the last frame's solution and error, and the norm history, into one
figure. `exactf` is the exact solution at the last frame's time, drawn
densely so it reads as the continuum answer rather than as another mesh.
"""
function makefigure(snaps, exactf, L, title; tols, steering)
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
    lines!(ax1, Float64.(xs), [Float64(exactf((x,), 1)) for x in xs];
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
function sinecase(::Type{T}=Float64; D=1, N=16, L=one(T), m=1, roots=4,
                 periods=T(9//10), ops_order=4,
                 centering=vertexcentered(1), backend=CPU()) where {T}
    G = viewer_ghosts(ops_order, centering)
    snaps = []
    observer = (fs, t, u) -> push!(snaps, snapshot(fs, t, u,
                                                  wave_exact(D, L, m, t)))
    r = wave_errors(T, Val(D); N=N, G=G, roots=roots, L=L, m=m, periods=periods,
                    ops=Operators(prolongation=ops_order,
                                  restriction=ops_order),
                    centering=centering,
                    backend=backend, observer=observer, nsnapshots=97)
    title = @sprintf("Standing sine mode, m = %d — two-level mesh, order-%d \
                      operators, %s, %s\nfinal L2 = %.3g, L∞ = %.3g at h = %.3g",
                     m, ops_order, T, centeringname(centering),
                     r.l2, r.linf, r.h)
    # The overlay is evaluated at the last *snapshot* time, which `snapshot`
    # has already converted to Float64 -- so hand the exact solution the
    # run's own type back, or it would be built in Float64.
    return (snaps=snaps, exactf=wave_exact(D, L, m, T(snaps[end].t)), L=L,
            title=title, tols=(refine_tol=0.30, coarsen_tol=0.075),
            steering=false)
end

"""
The travelling pulse with a refined region that follows it. The refined
blocks should sit under the pulse, and the norms should show what each
regrid costs.
"""
function pulsecase(::Type{T}=Float64; D=1, N=8, L=one(T), roots=8, σ=T(2//25),
                  x0=T(1//4), n=1, t_end=T(1//2), chunk=T(1//50), ops_order=4,
                  refine_tol=T(3//10), coarsen_tol=T(3//40),
                  centering=vertexcentered(1), backend=CPU()) where {T}
    G = viewer_ghosts(ops_order, centering)
    snaps = []
    observer = (fs, t, u) -> push!(snaps, snapshot(fs, t, u,
                                    pulse_exact(D, L, x0, σ, t; n=n)))
    r = track_pulse(T, Val(D); N=N, G=G, roots=roots, L=L, σ=σ, x0=x0, n=n,
                    ops=Operators(prolongation=ops_order,
                                  restriction=ops_order),
                    t_end=t_end, chunk=chunk, refine_tol=refine_tol,
                    coarsen_tol=coarsen_tol, centering=centering,
                    backend=backend, observer=observer)
    title = @sprintf("Travelling super-Gaussian pulse, n = %d, σ = %.3g, %s, %s \
                      — refinement tracks it\nworst L∞ = %.3g over the run, \
                      %d blocks at maxlevel %d",
                     n, σ, T, centeringname(centering),
                     r.worst, r.nblocks, r.maxlevel)
    return (snaps=snaps,
            exactf=pulse_exact(D, L, x0, σ, T(snaps[end].t); n=n),
            L=L, title=title,
            tols=(refine_tol=refine_tol, coarsen_tol=coarsen_tol),
            steering=true)
end

# `--type=` selects the floating-point type the run is computed in. Only the
# two hardware types are offered: a MultiFloat is a fine thing to *run* (see
# "Precision" in CODE.md) but Makie cannot plot one, so a viewer flag for it
# would be an invitation to a confusing failure.
const FLOATTYPES = Dict("f32" => Float32, "f64" => Float64)

# `--centering=` selects where the values sit. Vertex is the default, as
# it is throughout the package; `cell` renders the comparison.
const CENTERINGS = Dict("vertex" => vertexcentered, "cell" => cellcentered)

centeringname(centering) = all(==(:vertex), centering) ? "vertex-centred" :
                           "cell-centred"

"""
The ghost width the operator order needs, which is not a free choice and
is not the same on both layouts: order-`p` prolongation reaches `p/2`
planes past a cell-centred interface and `p/2 - 1` planes past a vertex
dimension's shared plane, and the Laplacian needs one either way. See
"Operator order" in `CODE.md`.
"""
viewer_ghosts(ops_order, centering) =
    max(1, ops_order ÷ 2 - (all(==(:vertex), centering) ? 1 : 0))

function main(args)
    case = "both"
    outdir = joinpath(@__DIR__, "output")
    n = 1
    dim = 1
    ops_order = 4
    T = Float64
    typetag = ""
    centeringtag = ""
    makecentering = vertexcentered
    backendname = "cpu"
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
        elseif startswith(a, "--type=")
            tag = a[8:end]
            haskey(FLOATTYPES, tag) ||
                error("--type must be f32 or f64; got $tag")
            T = FLOATTYPES[tag]
            # The default type keeps the plain filename, so a Float32 render
            # never overwrites the figure CI checks.
            typetag = tag == "f64" ? "" : "_$tag"
        elseif startswith(a, "--centering=")
            tag = a[13:end]
            haskey(CENTERINGS, tag) ||
                error("--centering must be vertex or cell; got $tag")
            makecentering = CENTERINGS[tag]
            # As with --type=, the default keeps the plain filename so a
            # cell-centred render never overwrites the figure CI checks.
            centeringtag = tag == "vertex" ? "" : "_$tag"
        elseif startswith(a, "--backend=")
            backendname = a[11:end]
        elseif a == "--display"
            inline = true
        elseif a == "--no-display"
            inline = false
        else
            error("unknown argument $a; expected --case=, --out=, --n=, \
                   --ops=, --dim=, --type=, --centering=, --backend=, \
                   --display, --no-display")
        end
    end
    case in ("both", "sine", "pulse") ||
        error("--case must be sine, pulse, or both; got $case")
    # The panels draw one line per block against x, which only means
    # anything in 1D. Higher D would need a different figure, not a
    # different argument.
    dim == 1 || error("only --dim=1 is implemented; got $dim")

    centering = makecentering(dim)

    mkpath(outdir)
    # Everything that touches the storage runs inside `withbackend`, which
    # is what makes a device package loaded a moment ago visible to it.
    written = withbackend(backendname, T) do backend
        paths = String[]
        for (name, build) in (("sine", () -> sinecase(T; D=dim,
                                                      ops_order=ops_order,
                                                      centering=centering,
                                                      backend=backend)),
                              ("pulse", () -> pulsecase(T; D=dim, n=n,
                                                        ops_order=ops_order,
                                                        centering=centering,
                                                        backend=backend)))
            (case == "both" || case == name) || continue
            @info "running the $name case"
            c = build()
            fig = makefigure(c.snaps, c.exactf, c.L, c.title; tols=c.tols,
                             steering=c.steering)
            path = joinpath(outdir,
                            "$(name)_$(dim)d$(typetag)$(centeringtag).png")
            save(path, fig)
            inline && display(fig)
            push!(paths, path)
        end
        paths
    end
    for p in written
        println("wrote $p")
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
