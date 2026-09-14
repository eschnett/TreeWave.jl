# Initial condition 2: a travelling super-Gaussian pulse.
#
# A super-Gaussian of order `n` is a near-flat-top bump with steep
# shoulders; `n = 1` recovers the ordinary Gaussian and larger `n`
# concentrates the variation into a narrower shoulder. That makes it the
# useful shape for exercising refinement: the feature worth refining is
# spatially compact and unambiguous, so a refinement criterion either
# tracks it or visibly does not.
#
# The order is a keyword, defaulting to `n = 1`, because it trades
# localization against resolvability and the mesh has to pay for it.
# Measured shoulder width (where G falls from 0.9 to 0.1) at σ = 0.08, in
# cells of the finest mesh the pulse tests use (h = 1/256):
#
#     | n | shoulder | cells | max|G'| |
#     |---|---|---|---|
#     | 1 | 0.0954   | 24.4  |   10.7  |
#     | 2 | 0.0530   | 13.6  |   19.0  |
#     | 4 | 0.0284   |  7.3  |   37.1  |
#     | 8 | 0.0148   |  3.8  |   73.7  |
#
# Two things get worse together as `n` rises: the shoulder needs more
# cells, and `∂ₜu = -G'` grows like `n/σ`, so the state itself is larger
# and the error scales with it. At `n = 8` the shoulder is under four
# cells wide on that mesh and the pulse is simply not represented; the
# same test at `n = 1` resolves it with 24. Raising `n` therefore means
# raising the resolution of every pulse run to match.

supergaussian(x, σ, n) = exp(-(x/σ)^(2n))
dsupergaussian(x, σ, n) = -(2n/σ) * (x/σ)^(2n-1) * supergaussian(x, σ, n)

"""
A Super-Gaussian pulse travelling in +x at the wave speed, exact for the 1D
wave equation and (with `σ ≪ L`) periodic to roundoff:

    u = G(d),  ∂ₜu = -G'(d),   d = x - x₀ - t  (wrapped)

The minus sign is what makes the pulse travel in +x rather than -x: for
`u = G(x - t)` the chain rule gives `∂ₜu = -G'`. Getting it wrong does not
produce a pulse moving the other way so much as a pulse that splits, since
the initial data then matches neither characteristic.

In more than one dimension it is a plane pulse, uniform in the
transverse directions, so `∇²u = ∂ₓ²u` and it stays exact.

`n` is the super-Gaussian order: it sets how steep the shoulders are, and
therefore how much resolution the pulse demands. It defaults to `1`, the
ordinary Gaussian; see the note at the top of this file before raising it.
"""
function pulse_exact(D, L, x0, σ, t; n=1)
    return function (x, var)
        d = wrap(x[1] - x0 - t + L / 2, L) - L / 2
        sg = supergaussian(d, σ, n)
        dsg = dsupergaussian(d, σ, n)
        return var == 1 ? sg : -dsg
    end
end

"""
Evolve a travelling pulse, regridding every `chunk` of time so the
refined region follows it. Returns the worst error over the run and how
well the refinement tracked the pulse.

Regridding changes both the length and the meaning of the state vector,
so each chunk is a fresh `solve`: stop, rebuild the schedule and the
state vector, restart — the pattern `CODE.md` prescribes for anything
beyond a one-step method.

Refinement is driven by the per-cell criterion in `refinement.jl`: where
the mesh fails to resolve the data, not where the amplitude is large. The
depth is therefore an *output* of the run — at the default tolerances the
indicator stops at level 2 on its own, with `maxlevel_cap` never binding — and
the same [`refine_mark`](@ref) serves both the initial adaptation and the
evolution so the two cannot drift apart.

The buffer width is derived from the pulse's motion by
[`refinement_buffer`](@ref); pass `buffer` explicitly to override it.

Pass `observer` to watch the run rather than only its outcome: it is
called as `observer(fs, t, u)` once per chunk (and once at `t = 0`),
after the data has been scattered into `fs` and before the regrid that
would invalidate it. This is what the viewer in `bin/` uses.

`T` is the floating-point type the run is computed in, defaulting to
`Float64`. It is said once, to the forest, and reaches the field set, the
schedule and the state vector from there. This case needs only `exp` and
integer powers, so unlike the sine mode it runs at a MultiFloats type as
well; see "Precision" in `CODE.md`.
"""
function track_pulse(::Type{T}, ::Val{D}; N=8, G=2, roots=8, L=one(T),
                     σ=T(1//20), x0=T(1//4), n=1,
                     ops=Operators(prolongation=4, restriction=4),
                     t_end=T(1//2), chunk=T(1//20), cfl=T(1//4), maxlevel_cap=2,
                     refine_tol=T(3//10), coarsen_tol=T(3//40), ε=T(1//100),
                     buffer=nothing, observer=nothing) where {T,D}
    forest = Forest{T}(ntuple(_ -> roots, D); N=N, G=G,
                       periodic=ntuple(_ -> true, D),
                       extents=ntuple(_ -> (zero(T), L), D))
    fs = FieldSet(forest, 2)

    # The wave speed is 1, so the pulse travels exactly `chunk` between one
    # regrid and the next. Deriving the margin from that is the
    # application's job -- it is physics the mesh cannot know. An explicit
    # `buffer` overrides the derivation, which exists so that a test can
    # measure what a deliberately too-narrow margin costs.
    buffer = buffer === nothing ? refinement_buffer(forest, maxlevel_cap, chunk) :
             buffer

    fill_by_coordinates!(pulse_exact(D, L, x0, σ, zero(T); n=n), fs)

    # The indicator's noise floor is referred to a global amplitude, and the
    # wave equation conserves that, so measuring it once from the initial
    # data serves the whole run. It also has to be fixed for the
    # adaptation cycle below, whose flag callback has no pass boundary to
    # refresh it at.
    scales = field_scales(fs)

    flag(b, k) = refine_mark(fs, b, k; scales=scales, refine_tol=refine_tol,
                             coarsen_tol=coarsen_tol, maxlevel_cap=maxlevel_cap, ε=ε)

    schedule, _, _ = adapt_to_initial_data!(fs, ops;
                                            initial=pulse_exact(D, L, x0, σ, zero(T);
                                                                n=n),
                                            flag=flag, buffer=buffer, maxpasses=8)

    if observer !== nothing
        u0 = statevector(fs)
        gather!(u0, fs)
        observer(fs, zero(T), u0)
    end

    worst = zero(T)
    refined_fraction = T[]
    # Counted rather than accumulated: the old `while t < t_end - 1e-12`
    # guard compared a time against an absolute slack, and at `Float32`
    # `1e-12` is far below one ulp of `t`, so the guard degenerated into
    # `t < t_end` and the final chunk's fate turned on rounding. The chunk
    # index is exact in every type.
    nchunks = ceilint(t_end / chunk)
    for c in 1:nchunks
        tstart = min((c - 1) * chunk, t_end)
        t = min(c * chunk, t_end)
        t > tstart || break
        problem = WaveProblem(fs, schedule)
        u = statevector(fs)
        gather!(u, fs)
        dt = cfl * minimum_spacing(forest)
        nsteps = max(1, ceilint((t - tstart) / dt))
        sol = solve(ODEProblem(wave_rhs!, u, (tstart, t), problem), RK4();
                    dt=(t - tstart) / nsteps, adaptive=false, save_everystep=false)
        scatter!(fs, sol.u[end])

        observer === nothing || observer(fs, t, sol.u[end])

        # Error against the exact travelling pulse.
        exact = FieldSet(forest, 2)
        fill_by_coordinates!(pulse_exact(D, L, x0, σ, t; n=n), exact)
        ue = statevector(exact)
        gather!(ue, exact)
        worst = max(worst, volume_weighted_norm(fs, sol.u[end] .- ue; p=Inf))

        # How much of the pulse sits in refined blocks -- the measure of
        # whether the refined region is actually following it.
        inside = zero(T)
        total = zero(T)
        for b in 1:nblocks(fs)
            peak = maximum(abs, interiorview(fs, b, 1))
            total = max(total, peak)
            level(blockkey(fs, b)) > 0 && (inside = max(inside, peak))
        end
        push!(refined_fraction, total > 0 ? inside / total : zero(T))

        # The indicator reads a 3-point stencil, so it needs ghosts; and
        # `regrid!` fills them only afterwards, for the transfer.
        fill_ghosts!(fs, schedule)
        flags = flag_blocks(flag, forest)
        if regrid!(forest, fs, schedule; flags=flags, buffer=buffer)
            schedule = GhostSchedule(forest, ops)
        end
    end

    return (worst=worst, tracking=minimum(refined_fraction),
            nblocks=nleaves(forest), maxlevel=maxlevel(forest))
end

track_pulse(valD::Val; kwargs...) = track_pulse(Float64, valD; kwargs...)

"""
The same travelling pulse on a *uniform* mesh, as the reference the
adaptive run is judged against: matching the finest uniform mesh is what
"tracks the pulse without artifacts" has to mean.
"""
function uniform_pulse(::Type{T}, ::Val{D}; roots, N, G=2, L=one(T),
                       σ=T(2//25), x0=T(1//4), n=1,
                       ops=Operators(prolongation=4, restriction=4),
                       t_end=T(1//2), cfl=T(1//4)) where {T,D}
    forest = Forest{T}(ntuple(_ -> roots, D); N=N, G=G,
                       periodic=ntuple(_ -> true, D),
                       extents=ntuple(_ -> (zero(T), L), D))
    fs = FieldSet(forest, 2)
    schedule = GhostSchedule(forest, ops)
    fill_by_coordinates!(pulse_exact(D, L, x0, σ, zero(T); n=n), fs)
    u = statevector(fs)
    gather!(u, fs)
    dt = cfl * minimum_spacing(forest)
    nsteps = ceilint(t_end / dt)
    sol = solve(ODEProblem(wave_rhs!, u, (zero(T), t_end), WaveProblem(fs, schedule)),
                RK4(); dt=t_end / nsteps, adaptive=false, save_everystep=false)
    exact = FieldSet(forest, 2)
    fill_by_coordinates!(pulse_exact(D, L, x0, σ, t_end; n=n), exact)
    ue = statevector(exact)
    gather!(ue, exact)
    return (err=volume_weighted_norm(fs, sol.u[end] .- ue; p=Inf),
            cells=nleaves(forest) * N^D)
end

uniform_pulse(valD::Val; kwargs...) = uniform_pulse(Float64, valD; kwargs...)
