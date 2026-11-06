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
# Like `visualize.jl`, this is an example of driving TreeWave from a
# script: `blast` below is the whole run, and the script owns its loop.
# The command line is in `options.jl`, the device in `backend.jl`, and
# what is recorded of the run and how it is drawn in `figure2d.jl`. See
# "Driving a run from a script" in `CODE.md`.

using TreeAMR
using TreeAMR: scatter!       # CairoMakie exports a `scatter!` of its own
using TreeWave
using OrdinaryDiffEqLowOrderRK: RK4
using SciMLBase: ODEProblem, solve

include(joinpath(@__DIR__, "options.jl"))
include(joinpath(@__DIR__, "backend.jl"))
include(joinpath(@__DIR__, "figure2d.jl"))

"""
The blast wave with a refined annulus that follows it: a loop of chunks,
the mesh rebuilt between them, as for the pulse in `visualize.jl`.

What differs from the pulse is one line, and that line is the reason this
case exists: the criterion's amplitude scales are re-measured at every
regrid, because geometric spreading takes the ring's peak from 1.0 to
about 0.11 over the run. `refresh_scales=false` renders the failure mode
instead, and it is worth rendering: this initial data's `∂ₜu` starts at
exactly zero, and with the scales frozen at `t = 0` the indicator refines
the entire domain by the second frame ("The noise floor must be global" in
`CODE.md`).
"""
function blast(T; G, ops, centering, backend, refresh_scales=true,
               σ=T(2//25), N=8, roots=8, L=one(T), t_end=T(2//5),
               chunk=T(1//50), cfl=T(1//4), maxlevel_cap=2,
               refine_tol=T(3//10), coarsen_tol=T(3//40), ε=T(1//100))
    D = 2
    x₀ = (L / 2, L / 2)
    # The exact ring is a Hankel quadrature, tabulated once out to the
    # widest the ring gets; each frame's exact solution is a contraction
    # against the table.
    reference = blast_reference(L, x₀, σ; rmax=t_end + 6σ)
    exact(t) = blast_exact(T, reference, t; backend)

    # The mesh, a uniform periodic box, and the initial peak on it.
    forest = Forest{T}((roots, roots); N, periodic=(true, true),
                       extents=((zero(T), L), (zero(T), L)))
    fs = FieldSet(forest, 2; G, centering, backend)
    initial = blast_initial(D, L, x₀, σ)
    fill_by_coordinates!(initial, fs)

    # The refinement criterion, and the mesh adapted to the initial data.
    # The buffer is how far the ring travels in a chunk.
    scales = field_scales(fs)
    flags(f) = refine_flags(f; scales, refine_tol, coarsen_tol, maxlevel_cap,
                            ε)
    buffer = refinement_buffer(forest, maxlevel_cap, chunk)
    schedule, _, _ = adapt_to_initial_data!(fs, ops; initial, flags, buffer,
                                            maxpasses=8)

    # The run, a chunk at a time, recorded at the end of each.
    snaps = [snapshot(fs, zero(T), exact(zero(T)); scales, coarsen_tol)]
    nchunks = TreeWave.chunk_count(t_end, chunk)
    for c in 1:nchunks
        t0, t1 = TreeWave.chunk_bounds(c, nchunks, t_end, chunk)
        u = statevector(fs)
        gather!(u, fs)
        nsteps = max(1, ceil(Int, (t1 - t0) / (cfl * minimum_spacing(forest))))
        sol = solve(ODEProblem(wave_rhs!, u, (t0, t1),
                               WaveProblem(fs, schedule)), RK4();
                    dt=(t1 - t0) / nsteps, adaptive=false,
                    save_everystep=false)
        scatter!(fs, sol.u[end])
        fill_ghosts!(fs, schedule)              # τ reads a 3-point stencil
        refresh_scales && (scales = field_scales(fs))   # the ring fades
        push!(snaps, snapshot(fs, t1, exact(t1); scales, coarsen_tol))

        c < nchunks || break                    # no regrid after the last
        if regrid!(forest, fs => schedule; flags=flags(fs), buffer)
            schedule = GhostSchedule(fs, ops)   # a schedule fits one layout
        end
    end
    return (; snaps, reference, L, x₀, σ, refine_tol, coarsen_tol)
end

# The command line and the figure; the run is above.
function main(args)
    o = viewer_options(args; D=2, frozen_scales=true)
    mkpath(o.out)
    @info "running the blast case"
    # The run is what touches the storage; `withbackend` is what makes a
    # device package loaded a moment ago visible to it. Everything after
    # it is a figure, drawn from the host copies `snapshot` already took.
    r = withbackend(o.backend, o.T) do backend
        blast(o.T; o.G, o.ops, o.centering, backend, o.refresh_scales)
    end
    fig = blast_figure(r, o)
    path = joinpath(o.out, (o.refresh_scales ? "blast_2d" : "blast_2d_frozen") *
                           o.suffix * ".png")
    save(path, fig)
    o.inline && display(fig)
    println("wrote $path")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
