# The standing sine mode, **vertex-centred**: an exact solution, so what
# it measures is the convergence *order* of the discretization.
#
# The acceptance criterion is that volume-weighted L2/L∞ errors against
# the exact sine-mode solution converge at 2nd order.
#
# Nothing in the application changed to get here. The Laplacian reads its
# own point and its neighbours a spacing away, which is the same stencil
# wherever those points sit, so the whole staggered study is the
# cell-centred one with one keyword moved. What *does* change is what the
# mesh does at a coarse-fine interface, and this file asserts the two
# consequences: restriction along a stagger is exact injection, so the
# restriction order cannot enter the rate; and prolongation reaches
# `p/2 - 1` planes past the shared plane, so `G = 1` is the whole
# requirement at order 4. The cell-centred study is kept verbatim in
# `sinewave_cell_tests.jl`.

using OrdinaryDiffEqLowOrderRK: RK4
using SciMLBase: ODEProblem, solve

@testset "Wave equation on a uniform grid: D=$D" for D in (1, 2)
    # Control: with no coarse-fine interfaces the 2nd-order Laplacian and
    # fixed-step RK4 must give a clean 2nd-order rate. Anything the
    # refined runs below lose is then attributable to the interface.
    hs, l2 = Float64[], Float64[]
    for N in (8, 16, 32)
        r = wave_errors(Val(D); N=N, G=1, refined=false)
        push!(hs, r.h)
        push!(l2, r.l2)
    end
    @test all(l2[i] > l2[i + 1] for i in 1:(length(l2) - 1))
    @test convergence_rate(hs, l2) ≈ 2.0 atol = 0.15   # measured 2.001
end

@testset "Wave equation on a two-level mesh: D=$D" for D in (1, 2)
    # Interpolation order must exceed the differencing order by two:
    # a prolongated ghost carries an O(h^p) error, and the 2nd-order
    # Laplacian divides it by h², leaving an O(h^(p-2)) truncation error
    # along the coarse-fine interface. With p = 4 that is O(h²) and does
    # not pollute the interior scheme.
    #
    # `G = 1`, not 2: along a stagger the order-4 prolongation stencil
    # reaches one plane past the source's shared plane, and the Laplacian
    # reaches one point past the owned range, so one ghost serves both.
    ops = Operators(prolongation=4, restriction=4)
    hs, l2, linf = Float64[], Float64[], Float64[]
    for N in (8, 16, 32)
        r = wave_errors(Val(D); N=N, G=1, ops=ops)
        push!(hs, r.h)
        push!(l2, r.l2)
        push!(linf, r.linf)
        @test r.nblocks > 2^D                     # refinement really happened
        @test isfinite(r.l2)
    end

    @test all(l2[i] > l2[i + 1] for i in 1:(length(l2) - 1))
    @test all(linf[i] > linf[i + 1] for i in 1:(length(linf) - 1))
    @test convergence_rate(hs, l2) ≈ 2.0 atol = 0.15    # measured 1.991
    @test convergence_rate(hs, linf) ≈ 2.0 atol = 0.2   # measured 2.01 / 2.06
end

@testset "Only prolongation limits the vertex rate: D=$D" for D in (1, 2)
    # The cell-centred study needs *both* orders raised, because each side
    # of the interface gets its ghosts from a different operator. Along a
    # stagger the restriction side is injection -- the coincident fine
    # point copied, exact for any data, with no order to raise -- so the
    # whole rate is the prolongation's: 1 at order 2, 2 at order 4,
    # whatever the restriction order says.
    rate(ops) = begin
        hs, l2 = Float64[], Float64[]
        for N in (8, 16, 32)
            r = wave_errors(Val(D); N=N, G=1, ops=ops)
            push!(hs, r.h)
            push!(l2, r.l2)
        end
        convergence_rate(hs, l2)
    end

    @test rate(Operators(prolongation=2, restriction=2)) ≈ 1.0 atol = 0.2
    @test rate(Operators(prolongation=2, restriction=4)) ≈ 1.0 atol = 0.2
    @test rate(Operators(prolongation=4, restriction=2)) ≈ 2.0 atol = 0.15
    @test rate(Operators(prolongation=4, restriction=4)) ≈ 2.0 atol = 0.15
end

@testset "The restriction order is inert along a stagger: D=$D" for D in (1, 2)
    # Stronger than the rates above, and the reason they come in pairs:
    # the restriction stencil in a vertex-like dimension is width 1 with
    # weight 1 *regardless of the order asked for*, so the two runs are
    # not merely equally accurate, they are the same computation. A
    # change upstream that quietly used `p` along a stagger would break
    # this long before it moved a rate.
    for p in (2, 4)
        a = wave_errors(Val(D); N=16, G=1, ops=Operators(prolongation=p,
                                                         restriction=2))
        b = wave_errors(Val(D); N=16, G=1, ops=Operators(prolongation=p,
                                                         restriction=4))
        @test a.l2 === b.l2
        @test a.linf === b.linf
    end
end

@testset "G = 1 is the whole requirement at order 4: D=$D" for D in (1, 2)
    # The vertex row of TreeAMR's operator table says G >= p/2 - 1, so a
    # second ghost plane buys nothing at order 4 -- not "almost nothing",
    # nothing: the same stencils read the same points. Cell centring needs
    # G >= p/2 and refuses G = 1 outright, which is what makes the
    # relaxation worth a test rather than a remark, and what stops a
    # future caller copying `G = 2` across from the cell-centred file
    # under the impression that it is load-bearing here.
    ops = Operators(prolongation=4, restriction=4)
    one_ghost = wave_errors(Val(D); N=16, G=1, ops=ops)
    two_ghosts = wave_errors(Val(D); N=16, G=2, ops=ops)
    @test one_ghost.l2 === two_ghosts.l2
    @test one_ghost.linf === two_ghosts.linf

    @test_throws "G >= 2" wave_errors(Val(D); N=16, G=1, ops=ops,
                                      centering=cellcentered(D))
end

@testset "The state vector holds N^D per block at either centring" begin
    # A vertex-centred block stores one plane more per dimension -- the
    # boundary plane it shares with its high-side neighbour -- but it does
    # not *own* it, so the ODE state is the same length and the same shape
    # as it was. Everything downstream of `statevector` rests on that:
    # `scatter!`/`gather!`, the RHS launch, and every norm.
    forest = wave_forest(Val(2), 8; roots=2, refined=false)
    vertex = FieldSet(forest, 2; G=1, centering=vertexcentered(2))
    cell = FieldSet(forest, 2; G=2, centering=cellcentered(2))

    @test length(statevector(vertex)) == forest.N^2 * 2 * nblocks(vertex)
    @test length(statevector(vertex)) == length(statevector(cell))

    # The stored array is where they differ: N + 2G + 1 against N + 2G.
    @test size(vertex.work)[1:2] == (8 + 2 * 1 + 1, 8 + 2 * 1 + 1)
    @test size(cell.work)[1:2] == (8 + 2 * 2, 8 + 2 * 2)

    # And the first owned point sits on the block's own origin when the
    # dimension is vertex-like, half a cell inside it when it is not.
    h = spacing(forest, blockkey(vertex, 1))
    origin = block_origin(forest, blockkey(vertex, 1))
    @test coordinates(vertex, 1, (2, 2)) == origin
    @test all(coordinates(cell, 1, (3, 3)) .≈ origin .+ h / 2)
end

@testset "Wave equation in 3D" begin
    # Smoke test only: 3D convergence runs are expensive, so this checks
    # that the same code path works and that the solution stays sane.
    r = wave_errors(Val(3); N=8, G=1, ops=Operators(prolongation=4, restriction=4))
    @test r.nblocks > 8
    @test isfinite(r.l2)
    @test r.l2 < 0.05                             # measured 0.00400
    @test r.linf < 0.2                            # measured 0.0220
end

@testset "Energy stays bounded" begin
    # A standing mode neither grows nor decays; a wrong interface
    # treatment usually shows up as slow drift long before it shows up
    # as an outright instability.
    D, L, m = 1, 1.0, 1
    forest = wave_forest(Val(D), 16)
    fs = FieldSet(forest, 2; G=1, centering=vertexcentered(D))
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
