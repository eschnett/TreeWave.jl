# The refinement criterion: a per-cell error indicator, reduced to a
# per-block verdict.
#
# The split is TreeAMR's: the mesh takes a flag per block, and "per-cell
# criteria are reduced to a block verdict inside the application's flag
# function". So the indicator lives here, and so does the reduction --
# but the buffering that follows from it is the mesh's job, and this file
# only has to report *where* the criterion fired so TreeAMR can dilate it.
#
# The criterion exists in two forms, because block storage may be on a
# device and a host loop cannot read it. They share the per-cell
# indicator (`cell_tau`) and they reach the identical verdict; what
# differs is who walks the cells -- this file, or TreeAMR's
# `firing_boxes` kernel. See "Running on a device" in `CODE.md`.
#
# What refinement is for is resolution, not amplitude. The criterion this
# replaced refined where |u| was large, which only ever worked because the
# pulse sat on a zero background; on the sine mode it would have refined
# the whole domain. A resolution criterion instead asks how well the mesh
# represents the data it holds, so the depth of the hierarchy becomes
# something a run measures rather than something the caller declares.

"""
    lohner(um, u0, up, scale; ε=1//100)

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

`ε` defaults to `1//100` converted to `scale`'s own type rather than to the
literal `0.01`. The literal is a `Float64` operand and would drag the whole
denominator — and therefore every `τ` — into `Float64` however the field is
stored; the rational is exact and folds away at compile time. See "Precision"
in `CODE.md`.

!!! note "Not the canonical threshold"
    Löhner's usual `τ > 0.8` is a shock detector. Smooth data never comes
    close, so thresholds for a problem like this one are much smaller and
    must be calibrated against the mesh — see `CODE.md`.

`τ` vanishes wherever the second difference does, so a Gaussian's
inflection points score zero even though the feature is right there. That
is why the reduction to a block verdict below is a maximum over cells
rather than a vote, and why the reported box is a bounding box.
"""
function lohner(um, u0, up, scale; ε=oftype(float(scale), 1//100))
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

Threaded over blocks, with one partial maximum per block and the maxima
combined afterwards. `max` is exact and the partials live in a
block-indexed array, so the answer is bit-identical whatever the thread
count — the property TreeAMR's M5 holds itself to, and which stops at the
first application loop that does not.

!!! warning "Not from inside a flag callback"
    TreeAMR calls `flag_blocks`' callback concurrently, so this must be
    evaluated *before* the flagging pass, never inside it. It is:
    [`refine_flags`](@ref) evaluates the `scales` keyword at its own call
    site, and the drivers hoist it further still.

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
function field_scales(fs::FieldSet{T}; vars=1:fs.nvars) where {T}
    R = real(float(T))
    G = fs.forest.G
    # One reduction per variable rather than one loop over both: a
    # device reduction is a kernel launch and a launch takes a single
    # variable range, and two launches once per regrid cost nothing
    # against the sweep that follows.
    return [maximum(TreeAMR.block_partials(w -> maximum(abs, w),
                                           (a, x) -> max(a, abs(x)), zero(R),
                                           fs.work, fs; g=G, vars=v:v))
            for v in vars]
end

"""
    cell_tau(work, idx, b, vars, scales, ε)

The worst [`lohner`](@ref) indicator over every dimension and variable at
one cell: `work` the ghost-inclusive working array, `idx` the cell's
**stored** index, `b` its block.

This is the whole of the criterion's arithmetic, and it is one function
because it is evaluated from two places that must not drift apart --
[`cell_indicator`](@ref)'s host loop and [`firing_flags`](@ref)'s device
kernel. The argument list is not this package's choice: it is exactly
what TreeAMR's `firing_boxes` hands a per-cell predicate, so the device
form can pass its arguments straight through.

`vars` and `scales` are **tuples**, not vectors. A kernel argument has to
be `isbits`, and a captured `Vector` is not; the same rule is why `ε` is
passed rather than defaulted here and why the accumulator starts at
`zero(scales[1])` rather than `zero(T)` — a captured `Type` is the
documented way to fail to compile for a device.
"""
@inline function cell_tau(work, idx::NTuple{D,Int}, b::Integer,
                          vars::NTuple{V,Int}, scales::NTuple{V}, ε) where {D,V}
    τ = zero(scales[1])
    for j in 1:V
        v = vars[j]
        scale = scales[j]
        u0 = work[idx..., v, b]
        for d in 1:D
            up = work[Base.setindex(idx, idx[d] + 1, d)..., v, b]
            um = work[Base.setindex(idx, idx[d] - 1, d)..., v, b]
            τ = max(τ, lohner(um, u0, up, scale; ε=ε))
        end
    end
    return τ
end

"""
    cell_indicator(fs, b, box_tol; scales, vars=1:fs.nvars, ε=T(1//100))

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

The host form of the criterion: it indexes the working array cell by
cell, so it needs the data under its own pointer. The per-cell
arithmetic is [`cell_tau`](@ref), shared with the device form; see
[`firing_flags`](@ref).

!!! warning "Ghosts must be filled first"
    The stencil reaches one cell beyond the interior at each face, so the
    caller must `fill_ghosts!` before flagging. `regrid!` fills ghosts
    only *after* flags are computed — for the transfer's prolongation —
    so it cannot do this for you, and stale ghosts here corrupt the
    verdict silently rather than raising anything.
"""
function cell_indicator(fs::FieldSet{T,D}, b::Integer, box_tol;
                        scales, vars=1:fs.nvars, ε=T(1//100)) where {T,D}
    forest = fs.forest
    N, G = forest.N, forest.G
    G >= 1 || throw(ArgumentError(
        "the Löhner stencil reads one cell beyond the interior, so G >= 1 is " *
        "required; got G = $G"))
    length(scales) == length(vars) || throw(DimensionMismatch(
        "got $(length(scales)) scales for $(length(vars)) variables"))

    # Tuples, built once outside the loop: that is what `cell_tau` takes,
    # because that is what a kernel argument may be.
    vt = ntuple(j -> Int(vars[j]), length(vars))
    st = Tuple(scales)

    τmax = zero(float(T))
    fired = false
    lo = ntuple(_ -> N, Val(D))
    hi = ntuple(_ -> 1, Val(D))

    # Over interior indices `1:N`, with the stored index derived -- the
    # same walk TreeAMR's firing kernel makes, so the two forms of the
    # criterion can be read against each other.
    for c in CartesianIndices(ntuple(_ -> N, Val(D)))
        i = ntuple(d -> Tuple(c)[d], Val(D))
        idx = ntuple(d -> i[d] + G, Val(D))
        τ = cell_tau(fs.work, idx, b, vt, st, ε)
        τmax = max(τmax, τ)
        τ > box_tol || continue
        if fired
            lo = ntuple(d -> min(lo[d], i[d]), Val(D))
            hi = ntuple(d -> max(hi[d], i[d]), Val(D))
        else
            fired = true
            lo = hi = i
        end
    end

    return (τmax, fired ? ntuple(d -> lo[d]:hi[d], Val(D)) : nothing)
end

"""
    refine_mark(fs, b, k; scales, refine_tol, coarsen_tol, maxlevel_cap,
                vars=1:fs.nvars, ε=T(1//100))

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
                     vars=1:fs.nvars, ε=T(1//100)) where {T,D}
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
    firing_flags(fs; scales, refine_tol, coarsen_tol, maxlevel_cap, vars, ε)

[`refine_mark`](@ref)'s verdict for every leaf, reached without a host
loop over cells: the flag vector `regrid!` takes, computed on whatever
backend the field set lives on.

The split is TreeAMR's `firing_boxes`: the mesh walks every interior cell
of every block in one kernel and reduces the cells that fired to a count
and a bounding box, and the application turns that into flags, because
which flag is physics the mesh cannot know. The four cases below are
[`refine_mark`](@ref)'s, line for line, and the box threshold is
`coarsen_tol` for the reason stated there.

**Two sweeps, because there are two thresholds.** A firing count answers
one yes-or-no question per block, and the criterion asks two: is any cell
above `refine_tol` (a count of zero is exactly `τmax <= refine_tol`,
since the count is over cells and `τmax` is their maximum), and where are
the cells above `coarsen_tol`. The second cannot be recovered from the
first, so the cells are walked twice. That is the cost of the form, and
it is paid once per regrid against an evolution of many steps.

Both predicates close over tuples and scalars only. See
[`cell_tau`](@ref) for why that is not a detail.
"""
function firing_flags(fs::FieldSet{T,D}; scales, refine_tol, coarsen_tol,
                      maxlevel_cap, vars=1:fs.nvars, ε=T(1//100)) where {T,D}
    coarsen_tol < refine_tol || throw(ArgumentError(
        "coarsen_tol ($coarsen_tol) must lie below refine_tol ($refine_tol): the " *
        "gap between them is the dead band that stops blocks flickering"))
    length(scales) == length(vars) || throw(DimensionMismatch(
        "got $(length(scales)) scales for $(length(vars)) variables"))

    vt = ntuple(j -> Int(vars[j]), length(vars))
    st = Tuple(scales)

    refires = firing_boxes(fs) do work, idx, b, x
        cell_tau(work, idx, b, vt, st, ε) > refine_tol
    end
    boxfires = firing_boxes(fs) do work, idx, b, x
        cell_tau(work, idx, b, vt, st, ε) > coarsen_tol
    end

    return map(1:nblocks(fs)) do b
        k = blockkey(fs, b)
        nrefine, _ = refires[b]
        nbox, box = boxfires[b]
        if nrefine > 0 && level(k) < maxlevel_cap
            return (Refine, box)
        elseif nbox > 0
            return (Keep, box)
        elseif level(k) > 0
            return Coarsen
        else
            return Keep
        end
    end
end

"""
    refine_flags(fs; refine_tol, coarsen_tol, maxlevel_cap, scales, vars, ε)

[`refine_mark`](@ref) for every leaf, as the flag vector `regrid!` takes.
`scales` defaults to a fresh [`field_scales`](@ref) reduction, computed
once for the whole pass. Ghosts must be filled first; see
[`cell_indicator`](@ref).

Whichever form of the criterion the storage calls for: the host loop on
the CPU, [`firing_flags`](@ref) on a device. The choice lives here so
that no driver has to make it: `track_pulse` and `track_blast` pass a
`backend` to the field set and then call this, and nothing between them
knows which form ran.

The host form is kept rather than retired in favour of the one that runs
everywhere, and not out of caution. It is the readable statement of the
criterion, it is what every number recorded in `CODE.md` was measured
with, and it returns `τmax` — which the 2D viewer draws and
`test/refinement_tests.jl` makes claims about — where a firing count
does not. It is also what the device form is *tested against*: the two
agree flag for flag and box for box on the CPU backend, which is the one
assertion that catches a drift between them.
"""
function refine_flags(fs::FieldSet; vars=1:fs.nvars,
                      scales=field_scales(fs; vars=vars), kwargs...)
    get_backend(fs.work) isa CPU ||
        return firing_flags(fs; scales=scales, vars=vars, kwargs...)
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
    cells = ceilint(travel / h) + 1
    cells <= forest.N || throw(ArgumentError(
        "a feature travelling $travel between regrids needs a $cells-cell margin " *
        "at level $maxlevel_cap (h = $h), which exceeds the block width N = " *
        "$(forest.N). Recruitment reaches one ring of neighbours, so the travel " *
        "must stay under one finest-level block width, $(forest.N * h): regrid " *
        "more often."))
    return cells
end
