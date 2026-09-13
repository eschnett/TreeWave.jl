# The workload behind the thread-count independence test (M5).
#
# Run as a standalone script —
#
#     julia -t N --project=. test/thread_workload.jl
#
# — it prints a digest of everything a full cycle produces: the state
# vector at every chunk, the error and tracking numbers, the mesh the run
# ended on. Two runs at different thread counts must print the same lines,
# character for character. That is stricter than TreeAMR's M5 acceptance
# ("matches serial to roundoff") and it is the property the code actually
# has, upstream and here: every parallel loop writes its own slot, and
# every combination of partials happens in a fixed order.
#
# It drives the package's own `track_pulse` and `track_blast` rather than
# a workload of its own. `CODE.md` names a fourth time-stepping loop as
# the thing to avoid, and the `observer` keyword exists precisely so a
# watcher does not need one — so the digests come out of the observer.
#
# Nothing here may use anything outside `Base` and the package: the script
# has to run from the root environment as well as from `test/`, which is
# what makes a digest mismatch bisectable. Hence `hash` rather than
# `sha256`, and `repr` rather than `@sprintf` — `repr` round-trips a
# Float64 exactly, so a difference in the last bit shows up as different
# characters.

using TreeAMR
using TreeWave

digest(u::Vector{Float64}) = string(hash(u); base=16, pad=16)

"""
One pulse run and one blast run, reduced to a handful of printed lines.

The pair is chosen to cover everything that threads: the pulse gives the
initial-data adaptation, the flagging pass and a moving regrid, and the
blast adds the pieces that are TreeWave's own parallel loops — the Hankel
table, its per-chunk contraction, the refreshed amplitude scales and the
coverage reduction.
"""
function thread_digests()
    lines = String[]

    watch(tag) = (fs, t, u) -> push!(lines, string(
        tag, " t=", repr(t), " u=", digest(u),
        " l2=", repr(volume_weighted_norm(fs, u)),
        " linf=", repr(volume_weighted_norm(fs, u; p=Inf)),
        " blocks=", nblocks(fs)))

    pulse = track_pulse(Val(1); roots=8, N=8, σ=0.08, t_end=0.04, chunk=0.02,
                        observer=watch("pulse"))
    push!(lines, string("pulse worst=", repr(pulse.worst),
                        " tracking=", repr(pulse.tracking),
                        " blocks=", pulse.nblocks, " maxlevel=", pulse.maxlevel))

    blast = track_blast(Val(2); roots=8, N=8, σ=0.08, t_end=0.04, chunk=0.02,
                        observer=watch("blast"))
    push!(lines, string("blast worst=", repr(blast.worst),
                        " covered=", repr(blast.covered),
                        " blocks=", blast.nblocks, " growth=", repr(blast.growth)))

    return lines
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && foreach(println, thread_digests())
