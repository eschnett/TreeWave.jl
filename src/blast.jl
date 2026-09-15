# Initial condition 3: a radial blast wave.
#
# A super-Gaussian peak released at rest in the middle of the box. It
# spreads as a ring at the wave speed, and the two initial conditions
# above cannot substitute for it, because it is the only one whose
# feature *changes amplitude* and whose refined region *grows*:
#
#   - the standing mode has O(1) amplitude everywhere, forever;
#   - the travelling pulse keeps its amplitude and its width, so the
#     refined region that follows it is the same size at every time.
#
# This is the Sedov blast-wave test in the only form this package can run
# today. There is no shock and no Riemann solver -- it is still the wave
# equation -- but the mesh problem is the one a Sedov test poses: a front
# expanding at a known speed, losing amplitude to geometric spreading,
# with the refined region tracking it outward and the interior coarsening
# back behind it. It is the first case here where coarsening does any
# work at all.
#
# Two dimensions and not three. In 3D Huygens' principle holds and the
# radial solution is elementary, `u = [h(r+t) + h(r-t)]/2r` with
# `h(s) = s·f(s)`; in 2D there is no Huygens principle, a wake trails the
# ring and never clears, and the exact solution is the Hankel quadrature
# below rather than a formula. 3D is nonetheless far too expensive for an
# adaptive shell -- see `CODE.md`, which records the measurement.

"""
    blast_initial(D, L, x₀, σ; n=1)

A radial super-Gaussian peak at rest: `u = G(r)`, `∂ₜu = 0`, with `r` the
periodically wrapped distance from `x₀`. The `(x, var) -> value` callback
[`fill_by_coordinates!`](@ref) takes.

Released at rest rather than outgoing, and that is forced rather than
chosen: a purely outgoing radial wave is `P(r-t)/r`, which is singular at
the origin unless `P(0) = 0`, so a peak *at the centre* cannot be purely
outgoing. It splits into an ingoing and an outgoing half, the ingoing half
passes through the origin immediately, and what survives is a single
expanding ring.

`∂ₜu ≡ 0` has a consequence the refinement criterion cares about, and
[`track_blast`](@ref) documents it: [`field_scales`](@ref) measured on
this data returns *exactly zero* for variable 2.

The radius is accumulated in a loop rather than with `sum(d -> …, 1:D)`
because this closure is a kernel argument; adding to a `zero` is exact,
so nothing measured moves. Everything it captures is a scalar or a
tuple of them.
"""
function blast_initial(D, L, x₀, σ; n=1)
    return function (x, var)
        var == 1 || return zero(float(σ))
        r2 = zero(float(σ))
        for d in 1:D
            r2 += (wrap(x[d] - x₀[d] + L / 2, L) - L / 2)^2
        end
        return supergaussian(sqrt(r2), σ, n)
    end
end

"""
    blast_reference(L, x₀, σ; rmax, nr=2001, nk=2000)

Everything the exact 2D blast solution needs that does **not** depend on
time, built once.

In two dimensions the radially symmetric solution is the inverse Hankel
transform of the initial profile, propagated mode by mode:

    u(r,t)  =  ∫₀^∞ f̂(k) J₀(kr) cos(kt) k dk
    ∂ₜu(r,t) = -∫₀^∞ f̂(k) J₀(kr) sin(kt) k² dk

with `f̂` the order-0 Hankel transform of `f`. For the ordinary Gaussian
that transform is elementary, `f̂(k) = (σ²/2)·exp(-k²σ²/4)`, which is the
whole reason an exact solution is available at all — and why it is
available only at `n = 1`.

Only `cos(kt)` and `sin(kt)` depend on `t`, so `J₀(kr)` is tabulated here
on the `(r, k)` grid and every later time is a contraction against it.
That is not an optimization worth hiding: evaluating the transform afresh
costs about 0.65 s, and a run that measures its error once per chunk wants
twenty of them, so the cached form turns thirteen seconds into one. It
costs `nr × nk` floats — 32 MB at the defaults.

The `k` integral is cut at `13/σ`, where the Gaussian envelope is
`e^(-42)`, and taken by the midpoint rule.

**The table is a host-side `Float64` object whatever the run's type is**, and
the arguments are converted to `Float64` here rather than at every call site.
`besselj0` has hardware-float methods only, so there is no choice for a
software type — but it is also the right answer for a `Float32` run on a
device, because a reference table is the last thing that should be uploaded
as fp64. [`blast_exact`](@ref) converts it into the run's own type once, on
the host, so nothing past that conversion sees a `Float64`. The cost is that the
reference is capped at `Float64` accuracy; its ~9e-7 quadrature error is four
orders below the finest discretization error measured against it, so nothing
this package can resolve notices.

The table is filled one `k` column per thread. Four million independent
`besselj0` calls with nothing to combine is the most favourable shape a
parallel loop can have, and serially it is the largest single block of
unbroken serial work in the package — of the order of a second against a
whole adaptive run of a few seconds. Column-major, so each task writes
contiguous pages; measured at 39× on 64 cores, see `CODE.md`.
"""
function blast_reference(L, x₀, σ; rmax, nr=2001, nk=2000)
    L, σ, rmax = tofloat64(L), tofloat64(σ), tofloat64(rmax)
    x₀ = map(tofloat64, x₀)
    dk = (13 / σ) / nk
    ks = [(j - 0.5) * dk for j in 1:nk]
    ws = [(σ^2 / 2) * exp(-(k * σ)^2 / 4) * k * dk for k in ks]
    rs = range(0.0, rmax; length=nr)
    J = Matrix{Float64}(undef, nr, nk)
    Threads.@threads for j in 1:nk
        k = ks[j]
        @inbounds for i in 1:nr
            J[i, j] = besselj0(k * rs[i])
        end
    end
    return (L=L, x₀=x₀, σ=σ, rs=rs, dr=step(rs), rmax=rmax, ks=ks, ws=ws, J=J)
end

"""
    blast_radial_table(ref, t)

The radial profiles of `u` and `∂ₜu` at time `t`, on the reference's own
radius grid. One pass over the cached `J₀(kr)`; both profiles are
accumulated together because they share every entry of it.

Threaded over *radii*, not over modes. The mode loop cannot be split:
every `k` contributes to every `r`, so `us` and `vs` would be shared
accumulators and the sum's order — and therefore its last bits — would
follow the thread count. Giving each task a contiguous range of radii
instead leaves each `us[i]` summed over all modes in the same order it
always was, so the result is bit-identical to the serial one for any
partition at all, and the `@simd` inner loop survives intact.

The one thing this does not fix is placement: `J` is first touched by
column in [`blast_reference`](@ref) and read by row here, so on a NUMA
node no first-touch pattern serves both passes. That is the application's
own instance of the effect TreeAMR measured in M5, and the reason to run
under `numactl --interleave=all`.
"""
function blast_radial_table(ref, t)
    nr, nk = size(ref.J)
    us = zeros(Float64, nr)
    vs = zeros(Float64, nr)
    rows = collect(Iterators.partition(1:nr, cld(nr, Threads.nthreads())))
    Threads.@threads for c in eachindex(rows)
        for j in 1:nk
            w = ref.ws[j]
            k = ref.ks[j]
            wc, wsn = w * cos(k * t), w * sin(k * t) * k
            @inbounds @simd for i in rows[c]
                Jij = ref.J[i, j]
                us[i] += Jij * wc
                vs[i] -= Jij * wsn
            end
        end
    end
    return (us=us, vs=vs)
end

"""
    blast_exact([T], ref, t)
    blast_exact([T], L, x₀, σ, t; n=1, nr=2001, nk=2000)

The exact solution of [`blast_initial`](@ref) at time `t`, as the
`(x, var) -> value` callback a field set is filled by. **Two dimensions
only**, and **`n = 1` only**: for a higher super-Gaussian order the Hankel
transform in [`blast_reference`](@ref) is not elementary and would need a
quadrature of its own. `n` is accepted and rejected rather than absent, so
that a caller who raised the order on [`blast_initial`](@ref) is told
instead of being handed a silently wrong answer.

The four-argument form builds its own [`blast_reference`](@ref) and is for
a single time; pass a reference instead when you want many, as
[`track_blast`](@ref) does.

Two properties make the radial table finite and correct:

**The solution is zero beyond `t + 6σ`.** It has no support ahead of the
front, and a Gaussian's tail at six standard deviations is `e^(-36)`. That
is what bounds `rmax`.

**The box is periodic, so the closure sums over periodic images.** This is
the difference from [`pulse_exact`](@ref), which needs no image sum: a
plane pulse of width `σ ≪ L` is always far from its own images, but a ring
reaches the corners of the box, where the nearest image lies exactly as
far away as the source does. The number of image rings is derived from
`rmax` rather than fixed, so raising `t_end` cannot quietly invalidate it.

Against a uniform mesh this reproduces 2nd-order convergence; `CODE.md`
records the measured rates, and the quadrature's own error is some four
orders below the finest discretization error, so what the comparison
measures is the scheme and not the table.
"""
function blast_exact(::Type{T}, ref, t; backend::Backend=CPU()) where {T}
    # The quadrature is `Float64` (see `blast_reference`); converting its two
    # profiles here is the one and only place that crosses over, so the
    # closure below — which is what a field set is filled from, and therefore
    # what runs on a device — is arithmetic in `T` alone.
    tab = blast_radial_table(ref, tofloat64(t))
    # The two profiles are the only arrays any callback in this package
    # closes over, so they are also the only ones it has to place. Once
    # per error measurement, `nr` values each.
    us = to_backend(backend, map(T, tab.us))
    vs = to_backend(backend, map(T, tab.vs))
    L, rmax, dr = T(ref.L), T(ref.rmax), T(ref.dr)
    x₀ = map(T, ref.x₀)
    nimages = ceil(Int, ref.rmax / ref.L)          # the reference is Float64
    return function (x, var)
        # `zero(x[1])`, not `zero(T)`: a closure that captures a `Type`
        # is the documented way to fail to compile for a device, and the
        # coordinate carries the type anyway.
        u = zero(x[1])
        v = zero(x[1])
        for j1 in (-nimages):nimages, j2 in (-nimages):nimages
            r = hypot(x[1] - x₀[1] - j1 * L, x[2] - x₀[2] - j2 * L)
            r < rmax || continue
            q = r / dr
            i = floorint(q) + 1
            s = q - (i - 1)
            u += (1 - s) * us[i] + s * us[i + 1]
            v += (1 - s) * vs[i] + s * vs[i + 1]
        end
        return var == 1 ? u : v
    end
end

blast_exact(ref, t; kwargs...) = blast_exact(Float64, ref, t; kwargs...)

function blast_exact(::Type{T}, L, x₀, σ, t; n=1, nr=2001, nk=2000,
                     backend::Backend=CPU()) where {T}
    check_blast_order(n)
    ref = blast_reference(L, x₀, σ; rmax=t + 6σ, nr=nr, nk=nk)
    return blast_exact(T, ref, t; backend=backend)
end

blast_exact(L, x₀, σ, t; kwargs...) = blast_exact(Float64, L, x₀, σ, t; kwargs...)

"""
The exact solution exists only for the ordinary Gaussian. Thrown from the
drivers as well as from [`blast_exact`](@ref), because a driver that
measures an error is the place where a raised order actually does damage.
"""
function check_blast_order(n)
    n == 1 || throw(ArgumentError(
        "the exact 2D blast solution is available only for the ordinary " *
        "Gaussian, n = 1; got n = $n. The solution is the inverse Hankel " *
        "transform of the initial profile, and that transform is elementary " *
        "only for a Gaussian -- a super-Gaussian of order $n would need a " *
        "quadrature of its own. Run the case at n = $n if you like, but " *
        "measure it against a uniform reference mesh, not against this."))
    return nothing
end

"""
    blast_coverage(fs)

The fraction of the feature's own cells that sit at the finest level
present: over cells whose `|u|` exceeds half the current peak, how many
belong to a block at `maxlevel`.

This exists because [`track_pulse`](@ref)'s measure — the peak `|u|` lies
inside a refined block — is too weak for a feature that is *extended*. A
ring can be losing its refinement all the way around while the single
highest cell still happens to sit on a fine block, and measured, that is
not hypothetical: the peak measure reports exactly 1.0 for a run whose
mesh has visibly stopped following the ring. This one reports 0.91 for the
calibrated run and falls when the mesh falls behind.

A fraction of *cells*, so it is `Float64` at every precision rather than the
run's own type — see the rule under "Precision" in `CODE.md`.
"""
function blast_coverage(fs::FieldSet{T}) where {T}
    # Two per-block reduction passes, both through the mesh's own
    # `block_mapreduce`, so neither reads a cell from the host and both are
    # bit-identical whatever the thread count -- see `field_scales`. The
    # second cannot start until the first has finished: it needs the peak.
    peaks = block_mapreduce(abs, max, zero(T), fs; vars=1)
    peak = maximum(peaks)
    finest = maximum(b -> level(blockkey(fs, b)), 1:nblocks(fs))

    # Counts per block, summed afterwards: integers, so the total is exact
    # under any order.
    half = peak / 2
    hot = block_mapreduce(x -> abs(x) > half, +, 0, fs; vars=1)
    total = sum(hot)
    total == 0 && return 1.0
    # Which blocks are the fine ones needs the tree, so that part stays
    # here; what it reduces is one integer per block.
    isfine(b) = level(blockkey(fs, b)) == finest
    fine = sum(hot[b] for b in 1:nblocks(fs) if isfine(b); init=0)
    return fine / total
end

"""
Evolve a radial blast wave, regridding every `chunk` of time so the
refined annulus follows the expanding ring. Returns the worst error over
the run, how well the refinement covered the ring, and how much the mesh
grew.

The loop is [`track_pulse`](@ref)'s, and deliberately a separate copy of
it rather than a shared abstraction over initial data — `CODE.md` names
that as a non-goal. What differs is one line, and that line is the whole
reason this case exists:

**`scales` is re-measured at every regrid.** [`track_pulse`](@ref)
measures [`field_scales`](@ref) once from the initial data, which is sound
for a problem that conserves its amplitude. This one does not: geometric
spreading takes the peak from 1.0 to about 0.11 over the run, and a floor
referred to the initial amplitude is then an order of magnitude out of
proportion to the data it is supposed to be flooring. Worse, this initial
data has `∂ₜu ≡ 0`, so the variable-2 scale starts at *exactly* zero — at
`t = 0` that is harmless, because the numerator vanishes with it and
[`lohner`](@ref) returns zero, but one chunk later `∂ₜu` is numerical dust
in the far field measured against a floor of zero, which is precisely the
scale-free pathology [`lohner`](@ref) exists to avoid. Frozen at `t = 0`,
the criterion refines the entire domain.

`refresh_scales=false` is kept so a test can measure that rather than the
docstring merely asserting it.

Pass `observer` to watch the run rather than only its outcome: it is
called as `observer(fs, t, u)` once per chunk (and once at `t = 0`), after
the data has been scattered into `fs` and before the regrid that would
invalidate it. This is what the viewer in `bin/` uses.

`T` is the floating-point type the run is computed in, defaulting to
`Float64`. The initial data needs only `exp`, so this case runs at a
MultiFloats type as well — but the exact ring it is measured against is a
`Float64` quadrature whatever `T` is; see [`blast_reference`](@ref) and
"Precision" in `CODE.md`.

`backend` is where it is computed, defaulting to the host. This is the
case where that asks the most: the quadrature stays a host `Float64`
table, and its two radial profiles are converted and uploaded once per
error measurement by [`blast_exact`](@ref). See "Running on a device" in
`CODE.md`.
"""
function track_blast(::Type{T}, ::Val{D}; N=8, G=2, roots=8, L=one(T),
                     σ=T(2//25), n=1, x₀=nothing,
                     ops=Operators(prolongation=4, restriction=4),
                     t_end=T(2//5), chunk=T(1//50), cfl=T(1//4), maxlevel_cap=2,
                     refine_tol=T(3//10), coarsen_tol=T(3//40), ε=T(1//100),
                     buffer=nothing, refresh_scales=true,
                     backend::Backend=CPU(), observer=nothing) where {T,D}
    forest = Forest{T}(ntuple(_ -> roots, D); N=N, G=G,
                       periodic=ntuple(_ -> true, D),
                       extents=ntuple(_ -> (zero(T), L), D))
    fs = FieldSet(forest, 2; backend=backend)
    x₀ = x₀ === nothing ? ntuple(_ -> L / 2, D) : x₀
    check_blast_order(n)

    # One reference for the whole run: `rmax` has to cover the ring at its
    # widest, and the cached `J₀(kr)` is then what makes measuring the
    # error once per chunk affordable rather than the dominant cost.
    reference = blast_reference(L, x₀, σ; rmax=t_end + 6σ)

    # The ring expands at the wave speed, so it travels `chunk` between one
    # regrid and the next -- the same derivation the travelling pulse uses,
    # for the same reason.
    buffer = buffer === nothing ?
             refinement_buffer(forest, maxlevel_cap, chunk) : buffer

    initial = blast_initial(D, L, x₀, σ; n=n)
    fill_by_coordinates!(initial, fs)

    scales = field_scales(fs)
    # `scales` is reassigned below, so this reads the current value at
    # every call rather than the initial one -- which is the point.
    flags(f) = refine_flags(f; scales=scales, refine_tol=refine_tol,
                            coarsen_tol=coarsen_tol,
                            maxlevel_cap=maxlevel_cap, ε=ε)

    schedule, _, _ = adapt_to_initial_data!(fs, ops; initial=initial,
                                            flags=flags, buffer=buffer,
                                            maxpasses=8)
    nblocks_initial = nleaves(forest)

    if observer !== nothing
        u0 = statevector(fs)
        gather!(u0, fs)
        observer(fs, zero(T), u0)
    end

    worst = zero(T)
    covered = 1.0
    # Counted rather than accumulated; see the same loop in `track_pulse` for
    # why an absolute `1e-12` slack cannot survive a change of precision.
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
                    dt=(t - tstart) / nsteps, adaptive=false,
                    save_everystep=false)
        scatter!(fs, sol.u[end])

        # Error against the exact ring, at every chunk rather than only at
        # the end: the mesh is rebuilt twenty times over the run and the
        # question is whether any one of those rebuilds hurt.
        exact = FieldSet(forest, 2; backend=backend)
        fill_by_coordinates!(blast_exact(T, reference, t; backend=backend),
                             exact)
        ue = statevector(exact)
        gather!(ue, exact)
        worst = max(worst, volume_weighted_norm(fs, sol.u[end] .- ue; p=Inf))

        # The indicator reads a 3-point stencil, so it needs ghosts; and
        # `regrid!` fills them only afterwards, for the transfer.
        fill_ghosts!(fs, schedule)
        refresh_scales && (scales = field_scales(fs))
        covered = min(covered, blast_coverage(fs))

        # The observer runs here, after the ghosts and the scales are the
        # ones the flagging below will actually use, so a viewer drawing τ
        # draws the value the mesh is about to act on -- and still before
        # the regrid that would invalidate `fs`.
        observer === nothing || observer(fs, t, sol.u[end])

        # No regrid after the final chunk: it would only leave the caller
        # holding a mesh that no returned solution was ever computed on,
        # and `nblocks` below is meant to describe the mesh `worst` was
        # measured against.
        c < nchunks || break
        if regrid!(forest, fs, schedule; flags=flags(fs), buffer=buffer)
            # For the field set's backend, not the host: the stencils are
            # read inside the transfer kernel.
            schedule = GhostSchedule(forest, ops; T=T, backend=backend)
        end
    end

    return (worst=worst, covered=covered, nblocks=nleaves(forest),
            maxlevel=maxlevel(forest),
            growth=nleaves(forest) / nblocks_initial)
end

track_blast(valD::Val; kwargs...) = track_blast(Float64, valD; kwargs...)

"""
The same blast wave on a *uniform* mesh, as the reference the adaptive run
is judged against, exactly as [`uniform_pulse`](@ref) is for the
travelling pulse.

Unlike the pulse, the adaptive run here is not expected to *match* the
uniform-fine mesh. The ring is extended, the criterion leaves the part of
it whose `τ` has fallen below `refine_tol` at a coarser level, and the
measured cost of that is recorded in `CODE.md`. Matching would mean the
criterion was refining more than it judged necessary.
"""
function uniform_blast(::Type{T}, ::Val{2}; roots, N, G=2, L=one(T),
                       σ=T(2//25), n=1, x₀=nothing,
                       ops=Operators(prolongation=4, restriction=4),
                       t_end=T(2//5), cfl=T(1//4),
                       backend::Backend=CPU()) where {T}
    forest = Forest{T}((roots, roots); N=N, G=G, periodic=(true, true),
                       extents=ntuple(_ -> (zero(T), L), 2))
    fs = FieldSet(forest, 2; backend=backend)
    schedule = GhostSchedule(forest, ops; T=T, backend=backend)
    x₀ = x₀ === nothing ? (L / 2, L / 2) : x₀
    check_blast_order(n)
    fill_by_coordinates!(blast_initial(2, L, x₀, σ; n=n), fs)
    u = statevector(fs)
    gather!(u, fs)
    dt = cfl * minimum_spacing(forest)
    nsteps = ceilint(t_end / dt)
    sol = solve(ODEProblem(wave_rhs!, u, (zero(T), t_end),
                           WaveProblem(fs, schedule)), RK4();
                dt=t_end / nsteps, adaptive=false, save_everystep=false)
    exact = FieldSet(forest, 2; backend=backend)
    fill_by_coordinates!(blast_exact(T, L, x₀, σ, t_end; backend=backend),
                         exact)
    ue = statevector(exact)
    gather!(ue, exact)
    return (err=volume_weighted_norm(fs, sol.u[end] .- ue; p=Inf),
            l2=volume_weighted_norm(fs, sol.u[end] .- ue),
            h=minimum_spacing(forest), cells=nleaves(forest) * N^2)
end

uniform_blast(val2::Val{2}; kwargs...) = uniform_blast(Float64, val2; kwargs...)
