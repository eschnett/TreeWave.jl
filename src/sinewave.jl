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

"""Angular frequency of the `m`-th sine mode on a box of side `L`."""
wave_omega(D, L, m) = 2π * m * sqrt(D) / L

"""The exact solution, as a `(x, v) -> value` callback for a field set."""
function wave_exact(D, L, m, t)
    ω = wave_omega(D, L, m)
    return function (x, var)
        shape = prod(sin(2π * m * x[d] / L) for d in 1:D)
        return var == 1 ? cos(ω * t) * shape : -ω * sin(ω * t) * shape
    end
end

"""
A two-level hierarchy: a `roots^D` periodic box with the middle sub-box
refined once, held fixed in physical space as `N` varies so that a
convergence study really does just shrink `h`. With `refined=false` the
same box is left uniform, as a control.
"""
function wave_forest(::Val{D}, N, G; roots=4, L=1.0, refined=true) where {D}
    forest = Forest(ntuple(_ -> roots, D); N=N, G=G,
                    periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, L), D))
    refined || return forest
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> 0.25L < (ext[d][1] + ext[d][2]) / 2 < 0.75L, 1:D)
    end
    refine!(forest, targets)
    balance!(forest)
    return forest
end

"""
Evolve the sine mode to `t_end` with fixed-step RK4 and return the
volume-weighted L2 and L∞ errors, plus the finest spacing.

Pass `observer` to watch the run rather than only its outcome: it is
called as `observer(fs, t, u)` at `nsnapshots` times spanning the run,
with `fs` already scattered from `u`, which is what the viewer in `bin/`
uses. Leaving it `nothing` keeps the cheap path — no intermediate
solution is stored.
"""
function wave_errors(::Val{D}; N, G=1, ops=Operators(prolongation=2, restriction=2),
                     roots=4, L=1.0, m=1,
                     cfl=0.25, periods=0.25, alg=RK4(), refined=true,
                     observer=nothing, nsnapshots=64) where {D}
    forest = wave_forest(Val(D), N, G; roots=roots, L=L, refined=refined)
    fs = FieldSet(forest, 2)
    problem = WaveProblem(fs, GhostSchedule(forest, ops))

    fill_by_coordinates!(wave_exact(D, L, m, 0.0), fs)
    u0 = statevector(fs)
    gather!(u0, fs)

    h = minimum_spacing(forest)
    t_end = periods * 2π / wave_omega(D, L, m)
    dt = cfl * h
    nsteps = ceil(Int, t_end / dt)
    dt = t_end / nsteps                          # land exactly on t_end

    # `saveat` spans the run inclusive of both ends, so `sol.u[end]` is
    # still the solution at `t_end` and the error below is unaffected.
    saveat = observer === nothing ? Float64[] :
             collect(range(0.0, t_end; length=nsnapshots))

    prob = ODEProblem(wave_rhs!, u0, (0.0, t_end), problem)
    sol = solve(prob, alg; dt=dt, adaptive=false, save_everystep=false,
                saveat=saveat)

    if observer !== nothing
        for (i, t) in enumerate(sol.t)
            scatter!(fs, sol.u[i])
            observer(fs, t, sol.u[i])
        end
    end

    exact = FieldSet(forest, 2)
    fill_by_coordinates!(wave_exact(D, L, m, t_end), exact)
    uexact = statevector(exact)
    gather!(uexact, exact)

    err = sol.u[end] .- uexact
    return (l2=volume_weighted_norm(fs, err),
            linf=volume_weighted_norm(fs, err; p=Inf),
            h=h, nsteps=nsteps, nblocks=nleaves(forest))
end
