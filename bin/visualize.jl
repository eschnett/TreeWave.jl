#!/usr/bin/env julia
#
# Visualize the 1D cases: the standing sine mode and the travelling pulse,
# each run here and drawn as its solution, its error against the exact
# answer, the refinement indicator, and the error norms over time.
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
# This is also the example of driving TreeWave from a script. The script
# owns the run -- the mesh, the time loop, the regrids -- and calls
# TreeAMR, TreeWave and the ODE solver itself; nothing in the package
# drives. `sine` and `pulse` below are the whole of each run, and are what
# to read. Everything else is kept out of their way: the command line is
# in `options.jl`, the device in `backend.jl`, and what is recorded of a
# run and how it is drawn in `figure1d.jl`. See "Driving a run from a
# script" in `CODE.md`.

using TreeAMR
using TreeAMR: scatter!       # CairoMakie exports a `scatter!` of its own
using TreeWave
using OrdinaryDiffEqLowOrderRK: RK4
using SciMLBase: ODEProblem, solve

include(joinpath(@__DIR__, "options.jl"))
include(joinpath(@__DIR__, "backend.jl"))
include(joinpath(@__DIR__, "figure1d.jl"))

"""
The standing sine mode on a static two-level mesh. The solution should lie
on top of the exact one, and the norms should oscillate about a level rather
than grow -- a standing mode neither grows nor decays.

The mesh never changes, so there is no loop: one `solve`, saved at
`nsnapshots` times, and a frame recorded from each saved state.

What the error panel shows depends on the operator order, and that is the
point of having it as a flag. At order 4 the error is a smooth phase error
that runs *continuously* across both coarse-fine interfaces: the interfaces
are invisible, which is exactly what TreeAMR's interface-order rule buys. At
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
function sine(T; G, ops, centering, backend, m=1, periods=T(9//10), N=16,
              roots=4, L=one(T), cfl=T(1//4), nsnapshots=97)
    D = 1
    exact(t) = wave_exact(D, L, m, t)

    # The mesh, a periodic box with its middle refined once, and the
    # initial state on it.
    forest = wave_forest(T, Val(D), N; roots, L)
    fs = FieldSet(forest, 2; G, centering, backend)
    schedule = GhostSchedule(fs, ops)
    fill_by_coordinates!(exact(zero(T)), fs)
    u = statevector(fs)
    gather!(u, fs)

    # The run: fixed-step RK4, landing exactly on `t_end`.
    t_end = periods * 2 * T(π) / wave_omega(D, L, m)
    nsteps = ceil(Int, t_end / (cfl * minimum_spacing(forest)))
    sol = solve(ODEProblem(wave_rhs!, u, (zero(T), t_end),
                           WaveProblem(fs, schedule)), RK4();
                dt=t_end / nsteps, adaptive=false, save_everystep=false,
                saveat=range(zero(T), t_end; length=nsnapshots))

    # The record: one frame per saved state.
    snaps = map(sol.t, sol.u) do t, u
        scatter!(fs, u)
        fill_ghosts!(fs, schedule)              # τ reads a 3-point stencil
        snapshot(fs, t, exact(t); scales=field_scales(fs))
    end
    return (; snaps, exact=exact(t_end), L, m, h=minimum_spacing(forest))
end

"""
The travelling pulse with a refined region that follows it. The refined
blocks should sit under the pulse, and the norms should show what each
regrid costs.

The mesh changes as the pulse moves, and a regrid changes both the length
and the meaning of the state vector, so the run is a loop of chunks: solve
one, record it, regrid, and start the next from the new mesh ("Regridding:
restart per chunk" in `CODE.md`).
"""
function pulse(T; G, ops, centering, backend, n=1, σ=T(2//25), x0=T(1//4),
               N=8, roots=8, L=one(T), t_end=T(1//2), chunk=T(1//50),
               cfl=T(1//4), maxlevel_cap=2, refine_tol=T(3//10),
               coarsen_tol=T(3//40), ε=T(1//100))
    D = 1
    exact(t) = pulse_exact(D, L, x0, σ, t; n)

    # The mesh, a uniform periodic box, and the initial state on it.
    forest = Forest{T}((roots,); N, periodic=(true,), extents=((zero(T), L),))
    fs = FieldSet(forest, 2; G, centering, backend)
    fill_by_coordinates!(exact(zero(T)), fs)

    # The refinement criterion, and the mesh adapted to the initial data.
    # The noise floor is referred to the initial amplitude, which the wave
    # equation conserves; the buffer is how far the pulse travels in a
    # chunk ("The refinement criterion" in `CODE.md`).
    scales = field_scales(fs)
    flags(f) = refine_flags(f; scales, refine_tol, coarsen_tol, maxlevel_cap,
                            ε)
    buffer = refinement_buffer(forest, maxlevel_cap, chunk)
    schedule, _, _ = adapt_to_initial_data!(fs, ops; initial=exact(zero(T)),
                                            flags, buffer, maxpasses=8)

    # The run, a chunk at a time, recorded at the end of each.
    snaps = [snapshot(fs, zero(T), exact(zero(T)); scales)]
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
        push!(snaps, snapshot(fs, t1, exact(t1); scales))

        c < nchunks || break                    # no regrid after the last
        if regrid!(forest, fs => schedule; flags=flags(fs), buffer)
            schedule = GhostSchedule(fs, ops)   # a schedule fits one layout
        end
    end
    return (; snaps, exact=exact(t_end), L, σ, refine_tol, coarsen_tol)
end

# The command line and the figures; the runs are above.
function main(args)
    o = viewer_options(args; D=1, cases=("sine", "pulse"), order=true)
    mkpath(o.out)
    # Everything that touches the storage runs inside `withbackend`, which
    # is what makes a device package loaded a moment ago visible to it.
    figures = withbackend(o.backend, o.T) do backend
        map(o.cases) do case
            @info "running the $case case"
            if case == "sine"
                r = sine(o.T; o.G, o.ops, o.centering, backend)
                case => sine_figure(r, o)
            else
                r = pulse(o.T; o.G, o.ops, o.centering, backend, o.n)
                case => pulse_figure(r, o)
            end
        end
    end
    for (case, fig) in figures
        path = joinpath(o.out, "$(case)_1d$(o.suffix).png")
        save(path, fig)
        o.inline && display(fig)
        println("wrote $path")
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
