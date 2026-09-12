# The refinement criterion: a per-cell error indicator, reduced to a
# per-block verdict.
#
# The split is TreeAMR's: the mesh takes a flag per block, and "per-cell
# criteria are reduced to a block verdict inside the application's flag
# function". So the indicator lives here, and so does the reduction --
# but the buffering that follows from it is the mesh's job, and this file
# only has to report *where* the criterion fired so TreeAMR can dilate it.
#
# What refinement is for is resolution, not amplitude. The criterion this
# replaced refined where |u| was large, which only ever worked because the
# pulse sat on a zero background; on the sine mode it would have refined
# the whole domain. A resolution criterion instead asks how well the mesh
# represents the data it holds, so the depth of the hierarchy becomes
# something a run measures rather than something the caller declares.

"""
    lohner(um, u0, up, scale; ε=0.01)

The Löhner error indicator for three consecutive cell values along one
dimension: the second difference normalized by the first differences,

    τ = |up - 2u0 + um| / (|up - u0| + |u0 - um| + 4ε·scale)

Undivided differences, so the spacing enters implicitly: `τ` measures how
well the *mesh* resolves the data, not the data's curvature. That is the
point — for smooth data `τ` falls as `h` shrinks, so a fixed threshold
makes refinement terminate on its own. `τ ∈ [0, 1]`, since the numerator
is bounded by the two first differences.

`scale` is a **global** reference amplitude for the variable, and it is
what makes the indicator usable. Löhner's classic form floors the
denominator with the local `ε(|up| + 2|u0| + |um|)` instead, which is
scale-free and therefore cannot tell a real feature from a ripple in the
numerical dust: on this package's Gaussian pulse, three consecutive tail
values of `3.9e-16, 3.6e-17, 3.6e-17` — ten orders of magnitude below the
peak — score `τ = 0.986`, because the floor shrinks along with them. The
measured consequence was a criterion that refined the entire domain at
every threshold tried. Referring the floor to a global amplitude fixes
it: negligible regions get a negligible numerator against a fixed floor,
and score ~0. See [`field_scales`](@ref).

!!! note "Not the canonical threshold"
    Löhner's usual `τ > 0.8` is a shock detector. Smooth data never comes
    close, so thresholds for a problem like this one are much smaller and
    must be calibrated against the mesh — see `CODE.md`.

`τ` vanishes wherever the second difference does, so a Gaussian's
inflection points score zero even though the feature is right there. That
is why the reduction to a block verdict below is a maximum over cells
rather than a vote, and why the reported box is a bounding box.
"""
function lohner(um, u0, up, scale; ε=0.01)
    num = abs(up - 2 * u0 + um)
    den = abs(up - u0) + abs(u0 - um) + 4 * ε * abs(scale)
    return iszero(den) ? zero(num) : num / den
end

"""
    field_scales(fs; vars=1:fs.nvars)

The global reference amplitude of each variable in `vars`: the largest
`|value|` over every block interior. This is the `scale` that
[`lohner`](@ref) refers its noise floor to.

One reduction over the whole field set, so it is computed once per
flagging pass and not once per block.

!!! note "When a fixed scale is enough"
    For a problem whose amplitude is roughly conserved — the wave
    equation being one — the scale can be measured once from the initial
    data and reused for the run, which is what [`track_pulse`](@ref)
    does. A problem whose solution grows or decays by orders of magnitude
    needs it refreshed, or the floor drifts out of proportion to the data.

    [`track_blast`](@ref) is that problem and refreshes at every regrid.
    Geometric spreading takes its peak down by an order of magnitude, and
    its `∂ₜu` starts at *exactly* zero, so a scale frozen at `t = 0`
    leaves variable 2 with no floor at all and the criterion refines the
    whole domain. Measured; see `CODE.md`.
"""
field_scales(fs::FieldSet; vars=1:fs.nvars) =
    [maximum(b -> maximum(abs, interiorview(fs, b, v)), 1:nblocks(fs)) for v in vars]

"""
    cell_indicator(fs, b, box_tol; scales, vars=1:fs.nvars, ε=0.01)

The worst [`lohner`](@ref) indicator over every interior cell, dimension,
and variable of block `b`, together with the bounding box of the cells
that exceeded `box_tol`. Returns `(τmax, box)`, with `box` an
`NTuple{D,UnitRange{Int}}` in the block's own **interior** indices `1:N`
as TreeAMR's flag boxes are specified, or `nothing` if no cell exceeded
`box_tol`.

Taking the maximum over variables matters for a system: for the
travelling pulse `∂ₜu` has an order of magnitude more amplitude than `u`,
so a feature can be badly under-resolved in one while the other still
looks smooth.

!!! warning "Ghosts must be filled first"
    The stencil reaches one cell beyond the interior at each face, so the
    caller must `fill_ghosts!` before flagging. `regrid!` fills ghosts
    only *after* flags are computed — for the transfer's prolongation —
    so it cannot do this for you, and stale ghosts here corrupt the
    verdict silently rather than raising anything.
"""
function cell_indicator(fs::FieldSet{T,D}, b::Integer, box_tol;
                        scales, vars=1:fs.nvars, ε=0.01) where {T,D}
    forest = fs.forest
    N, G = forest.N, forest.G
    G >= 1 || throw(ArgumentError(
        "the Löhner stencil reads one cell beyond the interior, so G >= 1 is " *
        "required; got G = $G"))
    length(scales) == length(vars) || throw(DimensionMismatch(
        "got $(length(scales)) scales for $(length(vars)) variables"))

    # Ghost-inclusive views, so the stencil at the first and last interior
    # cell has something to read.
    ws = [blockview(fs, b, v) for v in vars]
    unit = ntuple(d -> CartesianIndex(ntuple(k -> k == d ? 1 : 0, Val(D))), Val(D))

    τmax = zero(float(T))
    fired = false
    lo = ntuple(_ -> N, Val(D))
    hi = ntuple(_ -> 1, Val(D))

    for I in CartesianIndices(ntuple(_ -> (G + 1):(G + N), Val(D)))
        τ = zero(τmax)
        for (w, scale) in zip(ws, scales), d in 1:D
            e = unit[d]
            τ = max(τ, lohner(w[I - e], w[I], w[I + e], scale; ε=ε))
        end
        τmax = max(τmax, τ)
        τ > box_tol || continue
        # Stored indices run G+1:G+N; a flag box is stated over 1:N.
        if fired
            lo = ntuple(d -> min(lo[d], I[d] - G), Val(D))
            hi = ntuple(d -> max(hi[d], I[d] - G), Val(D))
        else
            fired = true
            lo = hi = ntuple(d -> I[d] - G, Val(D))
        end
    end

    return (τmax, fired ? ntuple(d -> lo[d]:hi[d], Val(D)) : nothing)
end

"""
    refine_mark(fs, b, k; scales, refine_tol, coarsen_tol, maxlevel_cap,
                vars=1:fs.nvars, ε=0.01)

The regrid mark for block `b` with key `k`, in the form
[`flag_blocks`](@ref) accepts: either a bare `RegridFlag` or a
`(flag, box)` pair.

The two thresholds mean different things, and the distinction is what
makes the whole scheme work:

- `refine_tol` is **"under-resolved here"** — the mesh is not
  representing what it holds, so go finer.
- `coarsen_tol` is **"there is something here at all"** — above it the
  feature is present even if adequately resolved.

Which gives four cases:

    τ > refine_tol, level < cap  ->  (Refine, box)
    box !== nothing              ->  (Keep, box)   equal-level margin
    box === nothing, level > 0   ->  Coarsen       (bare)
    otherwise                    ->  Keep          (bare)

with `box` the bounding box of the cells above `coarsen_tol` — the
feature's own footprint, which is the thing worth putting a margin
around.

**Why the box threshold is `coarsen_tol` and not `refine_tol`.** A block
that has been refined to the cap has, by construction, stopped being
under-resolved: its `τ` has fallen *below* `refine_tol`, which is exactly
why refinement terminated there. Keying the box on `refine_tol` would
therefore make a feature-holding block at the cap report nothing, and the
`(Keep, box)` margin below would be unreachable in the one case it exists
for.

**`(Keep, box)` at the cap.** Reporting a box is what makes a block a
dilation source, and a source asks for `level+1` when it says `Refine`
but its own `level` when it says `Keep`. A block holding the feature at
the finest level it may reach therefore still asks for its own level
around itself, which is an equal-level margin that travels with the
feature. Keying the buffer on `Refine` alone cannot do this: once the
block is refined it stops asking, and TreeAMR measured that version to be
inert at a steady-state frontier.

**A quiet `Keep` must stay bare.** A box on a block whose criterion did
not fire would recruit that block's neighbours, and since most blocks are
quiet most of the time, coarsening would die everywhere at once.

`maxlevel_cap` is named a cap rather than a `maxlevel` because it is one:
at calibrated tolerances the indicator terminates refinement on its own
and the cap never binds. The name also keeps it from shadowing TreeAMR's
exported `maxlevel` query inside a caller that wants both.

The gap between the thresholds is also the hysteresis dead band: a block
refines above `refine_tol` but coarsens only when every cell has dropped
below `coarsen_tol`, so one sitting near a single threshold cannot flip on
alternate regrids.
"""
function refine_mark(fs::FieldSet{T,D}, b::Integer, k::MortonKey{D};
                     scales, refine_tol, coarsen_tol, maxlevel_cap,
                     vars=1:fs.nvars, ε=0.01) where {T,D}
    coarsen_tol < refine_tol || throw(ArgumentError(
        "coarsen_tol ($coarsen_tol) must lie below refine_tol ($refine_tol): the " *
        "gap between them is the dead band that stops blocks flickering"))

    τ, box = cell_indicator(fs, b, coarsen_tol; scales=scales, vars=vars, ε=ε)
    if τ > refine_tol && level(k) < maxlevel_cap
        return (Refine, box)
    elseif box !== nothing
        return (Keep, box)
    elseif level(k) > 0
        return Coarsen
    else
        return Keep
    end
end

"""
    refine_flags(fs; refine_tol, coarsen_tol, maxlevel_cap, scales, vars, ε=0.01)

[`refine_mark`](@ref) for every leaf, as the flag vector `regrid!` takes.
`scales` defaults to a fresh [`field_scales`](@ref) reduction, computed
once for the whole pass. Ghosts must be filled first; see
[`cell_indicator`](@ref).
"""
function refine_flags(fs::FieldSet; vars=1:fs.nvars,
                      scales=field_scales(fs; vars=vars), kwargs...)
    return flag_blocks(fs.forest) do b, k
        refine_mark(fs, b, k; scales=scales, vars=vars, kwargs...)
    end
end

"""
    refinement_buffer(forest, maxlevel_cap, travel)

The buffer width in cells that covers a feature moving `travel` in
physical units between one regrid and the next, measured at the spacing
of level `maxlevel_cap`.

The width is the application's to choose because it is physics — feature
speed times regrid cadence — which the mesh cannot know. What TreeAMR
measured is that it must *exceed* the motion it covers: a margin narrower
than the travel per interval came out slightly worse than no margin at
all, since the feature leaves the refined region either way and the
narrow buffer only adds cells. Hence the `+ 1` rather than a bare `ceil`.

Uses the spacing at `maxlevel_cap` rather than `minimum_spacing`, which
reports the *current* finest spacing — coarse while the hierarchy is
still being built, and so would derive a uselessly narrow margin on the
first adaptation pass.

Recruitment reaches exactly one ring of neighbours, so TreeAMR caps the
buffer at `N`. That cap is really a statement about cadence: the feature
may not move more than one finest-level *block* per regrid, and this
throws naming that constraint rather than letting the caller discover it
as an opaque rejection from inside `regrid!`.
"""
function refinement_buffer(forest::Forest, maxlevel_cap::Integer, travel::Real)
    h = spacing(forest, maxlevel_cap)
    cells = ceil(Int, travel / h) + 1
    cells <= forest.N || throw(ArgumentError(
        "a feature travelling $travel between regrids needs a $cells-cell margin " *
        "at level $maxlevel_cap (h = $h), which exceeds the block width N = " *
        "$(forest.N). Recruitment reaches one ring of neighbours, so the travel " *
        "must stay under one finest-level block width, $(forest.N * h): regrid " *
        "more often."))
    return cells
end
