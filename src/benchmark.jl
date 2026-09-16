# Where an application's time goes, and how much of it threads.
#
# TreeAMR's own `bench/threads.jl` measures the mesh: scatter, ghost fill,
# the RHS launch, the norm. There is no point repeating that here. What it
# cannot see is everything between one right-hand-side evaluation and the
# next -- the integrator's stage arithmetic, the refinement criterion, the
# exact solution an error measurement needs -- and that is what this file
# measures. `wave_rhs!` is kept as the one phase common to both tables, so
# the two can be read against each other.
#
# Nothing here prints: `src/` stays free of I/O, and `bin/benchmark.jl`
# formats what these return.

"""
    best(f, reps; backend=CPU())

The shortest of `reps` timings of `f`, after one warm-up call. The minimum
is what a scaling study wants: noise only ever adds time, so the fastest
run is the one least contaminated by everything else on the node.

Each timing ends with a `synchronize`, which is a no-op on the CPU and is
the difference between a measurement and a fiction on a device: a kernel
launch and a broadcast both return before the work is done, so a phase
timed without it reports how long it took to *ask*. The phases that go
through the mesh synchronize on their own — `map_blocks!` and
`fill_by_coordinates!` do — but the integrator's stage arithmetic is an
ordinary broadcast and does not.
"""
function best(f, reps; backend::Backend=CPU())
    f()
    synchronize(backend)
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed begin
                    f()
                    synchronize(backend)
                end)
    end
    return t
end

"""
    benchmark_phases([T], ::Val{D}; N, roots, G=2, ops, reps=5,
                     centering=vertexcentered(D), backend=CPU(), ...)

Seconds per phase of the path an adaptive run actually pays for, on the
two-level mesh [`wave_forest`](@ref) builds. Returns
`(sizes=..., timings=[name => seconds, ...])` with the phases in the order
a step visits them; the caller formats.

`T`, `centering` and `backend` are what they are everywhere else in the
package: the type the run computes in, where its values sit, and where it
runs — defaulting to `Float64`, vertex, and the host. A device column and
a host column of this table can therefore be read against each other,
which is the point of it, and so can a vertex column and a cell-centred
one. Note that `workbytes` in the returned `sizes` differs between the
two: a vertex-like dimension stores one plane more.

The phases, and why each is here:

- `step` — one RK4 step, as the cost of a `solve` of `steps` of them
  divided by `steps`. This is the number that matters and the one
  TreeAMR's benchmark structurally cannot produce. `steps` has to be
  large enough to amortize what `solve` allocates up front: the
  integrator's caches are five more copies of the state vector, which at
  benchmark sizes is gigabytes and, at `steps = 2`, was measured to be
  half of the reported per-step cost.
- `rhs` — one `wave_rhs!`, the scatter → ghost fill → kernel sequence. Four
  of these make up the bulk of a step. The anchor against TreeAMR's table.
- `stage_broadcast` — one RK4-shaped combination of five state-sized
  vectors. `OrdinaryDiffEq` does its stage arithmetic as ordinary
  broadcasts, which are serial, so the gap between `4·rhs` and `step` is
  this phase plus the integrator's own bookkeeping. It is measured
  separately because a serial term in a threaded step is an Amdahl
  ceiling, and a ceiling should be a measured number rather than an
  inference.
- `initial_data` — `fill_by_coordinates!` with the pulse, i.e. the real
  callback, not a `sin`. Upstream turned this into a kernel in M5, so the
  callback now runs concurrently across blocks.
- `field_scales`, `refine_flags` — the refinement criterion: the global
  amplitude reduction and the per-cell Löhner sweep over every block. Both
  run once per regrid. `refine_flags` times *whichever form the backend
  calls for*, so the two columns under that name are two algorithms —
  a host loop over cells against two `firing_boxes` kernels. That is the
  honest comparison, since it is what a regrid costs either way, but it
  is not the same code and the ratio should not be read as a speedup of
  one thing.
- `blast_reference`, `blast_radial_table` — the Hankel quadrature behind
  the exact ring: a one-off table of 4M Bessel evaluations and the
  per-chunk contraction against it. Mesh-independent, so their sizes come
  from `nr`/`nk` rather than from the forest. Host `Float64` work at every
  backend, by design — `besselj0` has hardware-float methods only and a
  reference table is the last thing worth uploading as fp64 — so these
  two rows are expected not to move with the backend at all.
- `blast_coverage` — the per-chunk diagnostic reduction over the feature's
  own cells.
- `blast_exact` (D = 2 only) — filling a field set from the exact ring,
  the heaviest callback in the package: nine periodic images and two
  interpolations per cell.
"""
function benchmark_phases(::Type{T}, ::Val{D}; N, roots, G=2,
                          ops=Operators(prolongation=4, restriction=4),
                          reps=5, σ=T(2//25), L=one(T), x0=T(1//4),
                          cfl=T(1//4), steps=10,
                          refine_tol=T(3//10), coarsen_tol=T(3//40),
                          maxlevel_cap=2, nr=2001, nk=2000,
                          centering=vertexcentered(D),
                          backend::Backend=CPU()) where {T,D}
    heavy = max(2, reps ÷ 2)                     # for the allocating phases
    bestof(f, n) = best(f, n; backend=backend)

    forest = wave_forest(T, Val(D), N; roots=roots, L=L)
    fs = FieldSet(forest, 2; G=G, centering=centering, backend=backend)
    schedule = GhostSchedule(fs, ops)
    problem = WaveProblem(fs, schedule)

    initial = pulse_exact(D, L, x0, σ, zero(T))
    fill_by_coordinates!(initial, fs)
    u = statevector(fs)
    gather!(u, fs)
    du = similar(u)
    dt = cfl * minimum_spacing(forest)

    t_step = bestof(heavy) do
        solve(ODEProblem(wave_rhs!, u, (zero(T), steps * dt), problem), RK4();
              dt=dt, adaptive=false, save_everystep=false)
    end / steps
    t_rhs = bestof(() -> wave_rhs!(du, u, problem, zero(T)), reps)

    # One RK4 combination, on vectors the integrator's caches are the size
    # of. Five reads and a write per entry, which is the shape of the
    # arithmetic `OrdinaryDiffEq` puts between RHS evaluations.
    k1, k2, k3, k4, tmp = (similar(u) for _ in 1:5)
    t_stage = bestof(reps) do
        @. tmp = u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    end

    t_initial = bestof(() -> fill_by_coordinates!(initial, fs), heavy)

    # The criterion reads a 3-point stencil, so it wants ghosts; and the
    # scales have to exist before the flagging pass can be timed.
    fill_ghosts!(fs, schedule)
    scales = field_scales(fs)
    t_scales = bestof(() -> field_scales(fs), reps)
    t_flags = bestof(heavy) do
        refine_flags(fs; scales=scales, refine_tol=refine_tol,
                     coarsen_tol=coarsen_tol, maxlevel_cap=maxlevel_cap)
    end

    x₀ = ntuple(_ -> L / 2, D)
    rmax = T(2//5) + 6σ
    build() = blast_reference(L, x₀, σ; rmax=rmax, nr=nr, nk=nk)
    t_reference = best(build, 2)                 # host work, any backend
    reference = build()
    t_table = best(() -> blast_radial_table(reference, 0.2), reps)

    timings = ["step" => t_step,
               "rhs" => t_rhs,
               "stage_broadcast" => t_stage,
               "initial_data" => t_initial,
               "field_scales" => t_scales,
               "refine_flags" => t_flags,
               "blast_reference" => t_reference,
               "blast_radial_table" => t_table]

    push!(timings, "blast_coverage" => bestof(() -> blast_coverage(fs), reps))

    if D == 2
        exact = blast_exact(T, reference, T(1//5); backend=backend)
        t_exact = bestof(() -> fill_by_coordinates!(exact, fs), heavy)
        push!(timings, "blast_exact" => t_exact)
    end

    sizes = (D=D, N=N, roots=roots, blocks=nleaves(forest),
             cells=nleaves(forest) * N^D, statelength=length(u),
             workbytes=sizeof(fs.work), nr=nr, nk=nk,
             floattype=T, centering=centering,
             backend=nameof(typeof(backend)))
    return (sizes=sizes, timings=timings)
end

benchmark_phases(valD::Val; kwargs...) =
    benchmark_phases(Float64, valD; kwargs...)

"""
    benchmark_driver([T]; roots, N, σ, chunk, t_end, reps=2, G=2,
                     centering=vertexcentered(2), backend=CPU(), kwargs...)

Wall time for a whole [`track_blast`](@ref) run — evolution, error
measurement, flagging, regridding and all — as the end-to-end number the
phase table above is meant to explain. Returns
`(seconds=..., nblocks=..., growth=...)`.

An adaptive benchmark cannot be made bigger by raising `N` alone: `τ`
falls as `h` shrinks, so at fixed `σ` a finer mesh simply stops refining
and the run measures a smaller hierarchy rather than a larger one. The
size is raised by holding `σ / h₀` fixed instead — raise `roots` and `N`
together and shrink `σ` to match — which is why every one of these is a
keyword with no default.
"""
function benchmark_driver(::Type{T}=Float64; roots, N, σ, chunk, t_end, reps=2,
                          G=2, centering=vertexcentered(2),
                          backend::Backend=CPU(), kwargs...) where {T}
    run() = track_blast(T, Val(2); roots=roots, N=N, σ=T(σ), chunk=T(chunk),
                        t_end=T(t_end), G=G, centering=centering,
                        backend=backend, kwargs...)
    result = run()                               # also the warm-up
    seconds = Inf
    for _ in 1:reps
        seconds = min(seconds, @elapsed run())
    end
    return (seconds=seconds, nblocks=result.nblocks, growth=result.growth,
            worst=result.worst)
end
