# Initial condition 1: the standing sine mode.
#
# On a periodic box of side L the separable standing mode
#
#     u(x,t) = cos(ωt) ∏ sin(2πm x_d / L),   ω = 2πm √D / L
#
# is an exact solution of the wave equation in any number of dimensions,
# which is what the convergence study measures against. Because the exact
# answer is known at every time, this is the initial condition that pins
# down the *order* of the discretization.

"""
Angular frequency of the `m`-th sine mode on a box of side `L`.

The type follows `L`, and every constant is built in it: `2π` is a `Float64`
and `sqrt(D)` with an `Int` `D` is another, so the obvious spelling would
hand a `Float32` box a `Float64` frequency and promote the whole solution
back up with it. See "Precision" in `CODE.md`.
"""
function wave_omega(D, L, m)
    T = typeof(float(L))
    return 2 * T(π) * m * sqrt(T(D)) / L
end

"""
The exact solution, as a `(x, v) -> value` callback for a field set.

`2π` is built once here rather than inside the closure: for a software float
type `T(π)` goes through `BigFloat`, which must not happen per cell.

The closure is a kernel argument, so it captures scalars and nothing
else. That is also why the product over dimensions is an accumulating
loop and not `prod(... for d in 1:D)`, and why the accumulator starts at
`one(twoπ)` rather than `one(T)`: `T` here is a local, so closing over it
would put a `Type` in a kernel argument. Multiplying by one is exact, so
no measured number moves.
"""
function wave_exact(D, L, m, t)
    T = typeof(float(L))
    ω = wave_omega(D, L, m)
    twoπ = 2 * T(π)
    return function (x, var)
        shape = one(twoπ)
        for d in 1:D
            shape *= sin(twoπ * m * x[d] / L)
        end
        return var == 1 ? cos(ω * t) * shape : -ω * sin(ω * t) * shape
    end
end

"""
A two-level hierarchy: a `roots^D` periodic box with the middle sub-box
refined once, held fixed in physical space as `N` varies so that a
convergence study really does just shrink `h`. With `refined=false` the
same box is left uniform, as a control.

The leading `T` is the floating-point type the whole run is computed in; it
reaches the field set, the schedule and the state vector by way of the
forest, which is the only place it has to be said. It defaults to `Float64`.
"""
function wave_forest(::Type{T}, ::Val{D}, N, G; roots=4, L=one(T),
                     refined=true) where {T,D}
    forest = Forest{T}(ntuple(_ -> roots, D); N=N, G=G,
                       periodic=ntuple(_ -> true, D),
                       extents=ntuple(_ -> (zero(T), L), D))
    refined || return forest
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> L / 4 < (ext[d][1] + ext[d][2]) / 2 < 3L / 4, 1:D)
    end
    refine!(forest, targets)
    balance!(forest)
    return forest
end

wave_forest(valD::Val, N, G; kwargs...) = wave_forest(Float64, valD, N, G; kwargs...)

"""
Evolve the sine mode to `t_end` with fixed-step RK4 and return the
volume-weighted L2 and L∞ errors, plus the finest spacing.

Pass `observer` to watch the run rather than only its outcome: it is
called as `observer(fs, t, u)` at `nsnapshots` times spanning the run,
with `fs` already scattered from `u`, which is what the viewer in `bin/`
uses. Leaving it `nothing` keeps the cheap path — no intermediate
solution is stored.

`T` is the floating-point type the run is computed in, defaulting to
`Float64`. Note that this case needs `sin` and `cos`, so it is not
available at a MultiFloats type; see "Precision" in `CODE.md`.

`backend` is where it is computed, defaulting to the host. It is said
twice — to the field set and to the schedule — and everything else
follows the storage; see "Running on a device" in `CODE.md`.
"""
function wave_errors(::Type{T}, ::Val{D}; N, G=1,
                     ops=Operators(prolongation=2, restriction=2),
                     roots=4, L=one(T), m=1,
                     cfl=T(1//4), periods=T(1//4), alg=RK4(), refined=true,
                     backend::Backend=CPU(),
                     observer=nothing, nsnapshots=64) where {T,D}
    forest = wave_forest(T, Val(D), N, G; roots=roots, L=L, refined=refined)
    fs = FieldSet(forest, 2; backend=backend)
    problem = WaveProblem(fs, GhostSchedule(forest, ops; T=T, backend=backend))

    fill_by_coordinates!(wave_exact(D, L, m, zero(T)), fs)
    u0 = statevector(fs)
    gather!(u0, fs)

    h = minimum_spacing(forest)
    t_end = periods * 2 * T(π) / wave_omega(D, L, m)
    dt = cfl * h
    nsteps = ceilint(t_end / dt)
    dt = t_end / nsteps                          # land exactly on t_end

    # `saveat` spans the run inclusive of both ends, so `sol.u[end]` is
    # still the solution at `t_end` and the error below is unaffected. It
    # also keeps `nsnapshots` state vectors alive, which on a device is
    # device memory -- fine at the sizes a viewer asks for, and the
    # reason the observer path is opt-in.
    saveat = observer === nothing ? T[] :
             collect(range(zero(T), t_end; length=nsnapshots))

    # The tspan, not the state, is what fixes the *time* type for SciML, so
    # `dt` and every `t` handed to the RHS are `T` only because of this.
    prob = ODEProblem(wave_rhs!, u0, (zero(T), t_end), problem)
    sol = solve(prob, alg; dt=dt, adaptive=false, save_everystep=false,
                saveat=saveat)

    if observer !== nothing
        for (i, t) in enumerate(sol.t)
            scatter!(fs, sol.u[i])
            observer(fs, t, sol.u[i])
        end
    end

    exact = FieldSet(forest, 2; backend=backend)
    fill_by_coordinates!(wave_exact(D, L, m, t_end), exact)
    uexact = statevector(exact)
    gather!(uexact, exact)

    err = sol.u[end] .- uexact
    return (l2=volume_weighted_norm(fs, err),
            linf=volume_weighted_norm(fs, err; p=Inf),
            h=h, nsteps=nsteps, nblocks=nleaves(forest))
end

wave_errors(valD::Val; kwargs...) = wave_errors(Float64, valD; kwargs...)
