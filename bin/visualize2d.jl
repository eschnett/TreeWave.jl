#!/usr/bin/env julia
#
# Visualize the 2D radial blast wave: the ring expanding, the refined
# annulus following it, and how well the mesh keeps the solution radial.
#
#     julia --project=bin bin/visualize2d.jl
#     julia --project=bin bin/visualize2d.jl --frozen-scales --out=/tmp
#     julia --project=bin bin/visualize2d.jl --type=f32
#     julia --project=bin bin/visualize2d.jl --centering=cell
#     julia --project=bin bin/visualize2d.jl --backend=metal --type=f32
#
# A separate script from `visualize.jl` and not a `--dim=2` flag on it,
# because nothing transfers: that viewer draws one line per block against
# x, which in 2D is not a worse picture but no picture at all. `CODE.md`
# says as much -- higher D needs a different figure, not a different
# argument.
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
#
# The run comes from `track_blast` via its `observer` keyword, so this
# script contains no time-stepping loop of its own. See `CODE.md`.

using CairoMakie
using Printf
using SixelTerm
using TreeAMR
using TreeWave

include(joinpath(@__DIR__, "backend.jl"))

# `type="png"` is not just a default worth keeping. CairoMakie's fast
# image path -- the one that pads each heatmap's edges so that abutting
# blocks leave no hairline seam -- is disabled for vector output, where
# every heatmap instead degrades into one polygon per cell. A 772-block
# mesh has 49408 of those.
CairoMakie.activate!(; type="png", px_per_unit=2)

const LEVELCOLORS = Makie.wong_colors()

levelcolor(lvl) = LEVELCOLORS[mod1(lvl + 1, length(LEVELCOLORS))]

"""
One frame of a run: every block's `u`, its extent and its level, plus the
two scalars the time series needs. Built inside the `observer` callback,
where `fs` is scattered from `u` and still describes the mesh `u` was
computed on -- after a regrid it would not.

`track_blast` calls the observer after its ghost exchange and after the
scale refresh, so the τ recorded here is the value the criterion is about
to flag on rather than an approximation of it — provided the caller hands
in the same `scales` the run is using, which is why they are an argument
and not a `field_scales` call inside this function. Under
`--frozen-scales` recomputing them here would quietly draw the τ the run
*would* have seen, which is the one thing that figure must not do.

This is also where the run's own floating-point type stops. A run may be
`Float32`; everything below this point is a figure, and Makie is happiest
given `Float64`, so the conversion happens once here rather than at every
plot call.

And where its *storage* stops: a run may be on a device, and both
`interiorview` and `cell_indicator` read single cells. One `hostcopy`
here and the figure code below does not have to know. Note that `scales`
is measured by the caller from the field set as it really is -- that
reduction runs on a device -- so the τ drawn is still the run's own.
"""
function snapshot(fs, t, u; coarsen_tol, scales)
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
    return (t=Float64(t), blocks=blocks, nblocks=nblocks(fs), τ=Float64(τ))
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
The blast wave with a refined annulus that follows it.

`refresh_scales=false` renders the failure mode instead, and it is worth
rendering: the criterion's noise floor is referred to an amplitude
measured once from the initial data, this problem's amplitude falls by an
order of magnitude, and its `∂ₜu` starts at exactly zero. Frozen, the
indicator refines the entire domain by the second frame.
"""
function blastcase(::Type{T}=Float64; N=8, L=one(T), roots=8, σ=T(2//25),
                  t_end=T(2//5), chunk=T(1//50), n=1, ops_order=4,
                  refine_tol=T(3//10), coarsen_tol=T(3//40),
                  refresh_scales=true, centering=vertexcentered(2),
                  backend=CPU()) where {T}
    G = viewer_ghosts(ops_order, centering)
    x₀ = (L / 2, L / 2)
    snaps = []
    # Mirror the driver's scale policy rather than re-deriving it: frozen
    # means frozen at the first frame. (The driver measures its own frozen
    # scales just before the initial adaptation and this is just after, so
    # the two differ slightly on the peak; the figure's point is the block
    # count, which does not.)
    frozen = Ref{Any}(nothing)
    function observer(fs, t, u)
        scales = field_scales(fs)
        if !refresh_scales
            frozen[] === nothing && (frozen[] = scales)
            scales = frozen[]
        end
        return push!(snaps, snapshot(fs, t, u; coarsen_tol=coarsen_tol,
                                     scales=scales))
    end
    r = track_blast(T, Val(2); N=N, G=G, roots=roots, L=L, σ=σ, n=n, x₀=x₀,
                    ops=Operators(prolongation=ops_order,
                                  restriction=ops_order),
                    t_end=t_end, chunk=chunk, refine_tol=refine_tol,
                    coarsen_tol=coarsen_tol, centering=centering,
                    refresh_scales=refresh_scales, backend=backend,
                    observer=observer)
    what = refresh_scales ? "refinement tracks it" :
           "amplitude scale frozen — the criterion refines everything"
    title = @sprintf("Radial blast wave, σ = %.3g, %s, %s — %s\nworst L∞ = %.3g, \
                      %d blocks at maxlevel %d (×%.1f), %.0f%% of the ring \
                      at the finest level",
                     σ, T, centeringname(centering), what, r.worst, r.nblocks,
                     r.maxlevel, r.growth, 100 * r.covered)
    # The free-space radial solution at the final time, straight from the
    # Hankel table rather than sampled along a ray through the box: a ray
    # would leave the domain past r = L/2 and the periodic image sum would
    # answer with a *neighbouring* ring, which looks like a second peak.
    ref = blast_reference(L, x₀, σ; rmax=t_end + 6σ)
    tab = blast_radial_table(ref, snaps[end].t)
    return (snaps=snaps, profile=(rs=collect(ref.rs), us=tab.us),
            L=Float64(L), x₀=map(Float64, x₀),
            title=title, tracking=refresh_scales,
            tols=(refine_tol=refine_tol, coarsen_tol=coarsen_tol))
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
    outdir = joinpath(@__DIR__, "output")
    n = 1
    ops_order = 4
    refresh_scales = true
    T = Float64
    typetag = ""
    centeringtag = ""
    centering = vertexcentered(2)
    backendname = "cpu"
    # Sixel is for a human looking at a terminal; a pipe gets the paths.
    inline = stdout isa Base.TTY
    for a in args
        if startswith(a, "--out=")
            outdir = a[7:end]
        elseif startswith(a, "--n=")
            n = parse(Int, a[5:end])
        elseif startswith(a, "--ops=")
            ops_order = parse(Int, a[7:end])
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
            centering = CENTERINGS[tag](2)
            # As with --type=, the default keeps the plain filename so a
            # cell-centred render never overwrites the figure CI checks.
            centeringtag = tag == "vertex" ? "" : "_$tag"
        elseif startswith(a, "--backend=")
            backendname = a[11:end]
        elseif a == "--frozen-scales"
            refresh_scales = false
        elseif a == "--display"
            inline = true
        elseif a == "--no-display"
            inline = false
        else
            error("unknown argument $a; expected --out=, --n=, --ops=, \
                   --type=, --centering=, --backend=, --frozen-scales, \
                   --display, --no-display")
        end
    end

    mkpath(outdir)
    @info "running the blast case"
    # The run is what touches the storage; `withbackend` is what makes a
    # device package loaded a moment ago visible to it. Everything below
    # is a figure, drawn from the host copies `snapshot` already took.
    c = withbackend(backendname, T) do backend
        blastcase(T; n=n, ops_order=ops_order, refresh_scales=refresh_scales,
                  centering=centering, backend=backend)
    end
    fig = makefigure(c.snaps, c.profile, c.L, c.x₀, c.title; tols=c.tols,
                     tracking=c.tracking)
    path = joinpath(outdir,
                    (refresh_scales ? "blast_2d" : "blast_2d_frozen") *
                    typetag * centeringtag * ".png")
    save(path, fig)
    inline && display(fig)
    println("wrote $path")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
