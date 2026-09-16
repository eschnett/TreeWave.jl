# The standing sine mode, **cell-centred** -- the study as it stood
# before vertex centring, kept verbatim beside `sinewave_tests.jl` so
# that its numbers stay under test rather than becoming a remark in
# `CODE.md`. Every call says `centering = cellcentered(D)` out loud, and
# every ghost width and tolerance is the one that was measured.
#
# The claim the pair makes together is that the interface-order rule has
# two forms: cell-centred needs *both* operator orders raised and
# `G = 2`; vertex-centred needs only the prolongation and `G = 1`. This
# file is the first half. See "Centerings" in `CODE.md`.

using OrdinaryDiffEqLowOrderRK: RK4
using SciMLBase: ODEProblem, solve

@testset "Cell-centred wave equation on a uniform grid: D=$D" for D in (1, 2)
    # Control: with no coarse-fine interfaces the 2nd-order Laplacian and
    # fixed-step RK4 must give a clean 2nd-order rate. Anything the
    # refined runs below lose is then attributable to the interface.
    hs, l2 = Float64[], Float64[]
    for N in (8, 16, 32)
        r = wave_errors(Val(D); N=N, G=1, refined=false,
                        centering=cellcentered(D))
        push!(hs, r.h)
        push!(l2, r.l2)
    end
    @test all(l2[i] > l2[i + 1] for i in 1:(length(l2) - 1))
    @test convergence_rate(hs, l2) ≈ 2.0 atol = 0.15
end

@testset "Cell-centred wave equation on a two-level mesh: D=$D" for D in (1, 2)
    # Interpolation order must exceed the differencing order by two:
    # a prolongated ghost carries an O(h^p) error, and the 2nd-order
    # Laplacian divides it by h², leaving an O(h^(p-2)) truncation error
    # along the coarse-fine interface. With p = 4 that is O(h²) and does
    # not pollute the interior scheme.
    ops = Operators(prolongation=4, restriction=4)
    hs, l2, linf = Float64[], Float64[], Float64[]
    for N in (8, 16, 32)
        r = wave_errors(Val(D); N=N, G=2, ops=ops, centering=cellcentered(D))
        push!(hs, r.h)
        push!(l2, r.l2)
        push!(linf, r.linf)
        @test r.nblocks > 2^D                     # refinement really happened
        @test isfinite(r.l2)
    end

    @test all(l2[i] > l2[i + 1] for i in 1:(length(l2) - 1))
    @test all(linf[i] > linf[i + 1] for i in 1:(length(linf) - 1))
    @test convergence_rate(hs, l2) ≈ 2.0 atol = 0.15
    @test convergence_rate(hs, linf) ≈ 2.0 atol = 0.2
end

@testset "Both operator orders limit the cell-centred rate: D=$D" for D in (1,)
    # Documents the mechanism above, and guards it: with order-2
    # operators the interface error is O(1) and drags the global rate
    # down to first order, even though the interior scheme is 2nd order.
    # Both operators matter -- raising only one leaves the other side of
    # the interface first order.
    rate(ops, G) = begin
        hs, l2 = Float64[], Float64[]
        for N in (8, 16, 32)
            r = wave_errors(Val(D); N=N, G=G, ops=ops,
                            centering=cellcentered(D))
            push!(hs, r.h)
            push!(l2, r.l2)
        end
        convergence_rate(hs, l2)
    end

    @test rate(Operators(prolongation=2, restriction=2), 1) ≈ 1.0 atol = 0.2
    @test rate(Operators(prolongation=4, restriction=2), 2) < 1.5
    @test rate(Operators(prolongation=2, restriction=4), 2) < 1.5
    @test rate(Operators(prolongation=4, restriction=4), 2) ≈ 2.0 atol = 0.15
end

@testset "Cell-centred wave equation in 3D" begin
    # Smoke test only: 3D convergence runs are expensive, so this checks
    # that the same code path works and that the solution stays sane.
    r = wave_errors(Val(3); N=8, G=2, ops=Operators(prolongation=4, restriction=4),
                    centering=cellcentered(3))
    @test r.nblocks > 8
    @test isfinite(r.l2)
    @test r.l2 < 0.05
    @test r.linf < 0.2
end

@testset "Cell-centred energy stays bounded" begin
    # A standing mode neither grows nor decays; a wrong interface
    # treatment usually shows up as slow drift long before it shows up
    # as an outright instability.
    D, L, m = 1, 1.0, 1
    forest = wave_forest(Val(D), 16)
    fs = FieldSet(forest, 2; G=2, centering=cellcentered(D))
    problem = WaveProblem(fs, GhostSchedule(fs, Operators(prolongation=4,
                                                          restriction=4)))
    fill_by_coordinates!(wave_exact(D, L, m, 0.0), fs)
    u0 = statevector(fs)
    gather!(u0, fs)

    t_end = 4 * 2π / wave_omega(D, L, m)          # four full periods
    dt = 0.25 * minimum_spacing(forest)
    nsteps = ceil(Int, t_end / dt)
    sol = solve(ODEProblem(wave_rhs!, u0, (0.0, t_end), problem), RK4();
                dt=t_end / nsteps, adaptive=false, save_everystep=false)

    @test all(isfinite, sol.u[end])
    amplitude(u) = volume_weighted_norm(fs, u; p=Inf)
    @test amplitude(sol.u[end]) ≈ amplitude(u0) rtol = 0.05
end
