# Where the work runs.
#
# TreeAMR's M6 makes the *storage* decide: a backend keyword on `FieldSet`
# and `GhostSchedule` and nowhere else, after which `statevector`
# allocates where the field set lives, `regrid!` reallocates there, and
# every kernel takes its backend from the array it is handed. So an
# application that already goes through `map_blocks!` and
# `fill_by_coordinates!` is most of the way there, and what is left is
# only the places where *this* package keeps something of its own that a
# kernel reads.
#
# There are two such places, and this file is both of them: per-block
# metadata the RHS kernel reads (`WaveProblem`'s spacings, and the blast
# wave's radial profiles), and the reverse direction, bringing a field set
# back to the host so something that can only run there -- a plot -- can
# read it.

"""
    to_backend(backend, a)

`a` on `backend`: itself on the CPU, a fresh device array copied from it
otherwise.

The mesh does this for its own metadata — `block_origins` and
`block_spacings` are uploaded inside `fill_by_coordinates!` and
`firing_boxes` — and an application has to do it for its own. It is
spelled out here rather than reaching for `TreeAMR.todevice` on purpose:
this package stands for a downstream user of the public API, and
`KernelAbstractions.allocate` is the whole of what that requires.
"""
to_backend(::CPU, a::AbstractArray) = a

function to_backend(backend::Backend, a::AbstractArray)
    dev = allocate(backend, eltype(a), size(a))
    copyto!(dev, a)
    return dev
end

"""
    hostcopy(fs)

`fs` with its working array on the host: `fs` itself when it is already
there, a new field set over the same forest otherwise.

This is for the consumers that cannot be moved onto a device rather than
for ones that have not been. The viewers in `bin/` are the case: they read
single cells through [`blockview`](@ref) and [`coordinates`](@ref) and hand
them to CairoMakie, which is host code by nature, so the data has to come
down whatever the run was computed on. One copy at the top of a snapshot
buys that, and every line of figure code below it is unchanged.

It is deliberately *not* how the numeric diagnostics work. `field_scales`
and [`blast_coverage`](@ref) run once per chunk against the whole state,
and a copy of the whole state per chunk is hundreds of megabytes on a run
worth putting on a device at all; those go through per-block reductions
that stay where the data is. See "Running on a device" in `CODE.md`.
"""
function hostcopy(fs::FieldSet{T}) where {T}
    get_backend(fs.work) isa CPU && return fs
    # The whole layout, not merely the forest: a field set carries its own
    # ghost width and centering from TreeAMR's M8 on, and either one left
    # at its default would give the copy a differently shaped working
    # array — which `copyto!` would then reject, or worse, accept.
    host = FieldSet{T}(fs.forest, fs.nvars; G=fs.G, centering=fs.centering)
    copyto!(host.work, fs.work)
    return host
end
