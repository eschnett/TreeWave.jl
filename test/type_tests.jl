# Element-type genericity: the drivers must run in a caller-chosen float
# type, with no Float64 left in the arithmetic.
#
# The failure mode these guard is code that is generic in name only --
# computing in Float64 and converting at the end. That is invisible in a
# Float64 run and fatal on a device with no hardware fp64, which is the
# case Float32 exists here for.
#
# Each non-default type catches a different fault:
#
#   Float32    is the *leak detector*. A stray Float64 operand widens the
#              result, so a returned Float64 names the leak.
#   Float32x2  is the *off the beaten path* detector: a software type built
#              from two Float32 limbs, which no Float64 fast path can
#              serve. It cannot detect leaks -- MultiFloats promotes
#              Float64 *downward*, so a leak is absorbed silently -- it
#              tests instead that nothing depends on a hardware float at
#              all. Float64x2 works the same way and is what a run wanting
#              more precision than Float64 would use.
#
# The convergence *orders* are asserted only at Float64, in
# `sinewave_tests.jl` and `blast_tests.jl`, and deliberately not repeated
# here: roundoff in the Laplacian is eps/h^2, which at Float32 and
# h = 1/256 is ~8e-3 -- comparable to the discretization error the finest
# mesh of those sweeps measures. A Float32 sweep cannot show second order,
# and moving the meshes until it did would be fitting the test to the
# answer. What survives a change of precision is asserted instead: the
# types, boundedness, and the mesh the criterion chooses.
#
# The sine mode is absent from the MultiFloat runs on purpose. It needs
# `sin` and `cos`, which MultiFloats does not implement -- they `error`,
# pointing at `MultiFloats.use_bigfloat_transcendentals()`, which evals
# BigFloat-backed methods into Base. TreeWave does not call that, as
# TreeAMR does not; see "Precision" in CODE.md.

using MultiFloats: Float32x2

const TYPEOPS = Operators(prolongation=4, restriction=4)
const FLOATTYPES = (Float64, Float32, Float32x2)

# Short enough that the MultiFloat run is affordable, long enough that the
# mesh is rebuilt several times.
shortpulse(T) = (; roots=8, N=8, G=2, σ=T(2//25), t_end=T(1//10), ops=TYPEOPS)

@testset "Base's gaps at a software float are bridged: T=$T" for T in FLOATTYPES
    # `mod`, `ceil(Int, ·)` and `Float64(·)` are all MethodErrors at a
    # MultiFloat, and every one of them sits on a path the drivers take.
    # Without these three the package is Float32-and-Float64-only, and the
    # symptom is a MethodError from inside a run rather than anything the
    # type system warns about.
    @test TreeWave.wrap(T(9//4), one(T)) ≈ T(1//4)
    @test TreeWave.wrap(-T(1//4), one(T)) ≈ T(3//4)
    @test TreeWave.wrap(T(9//4), one(T)) isa T
    @test TreeWave.ceilint(T(5//2)) === 3
    @test TreeWave.floorint(T(5//2)) === 2
    @test TreeWave.ceilint(T(2)) === 2                 # already integral
    @test TreeWave.tofloat64(T(1//2)) === 0.5

    # And `wrap` really is `mod` where `mod` exists, which is what lets the
    # Float64 numbers in CODE.md stay put.
    T <: Base.IEEEFloat && @test TreeWave.wrap(T(9//4), one(T)) === mod(T(9//4), one(T))
end

@testset "The pulse computes in the type it is given: T=$T" for T in FLOATTYPES
    # Every number a driver returns has to carry the caller's type. At
    # Float32 a Float64 here would name a promotion inside the run; at
    # Float32x2 it would be an outright MethodError long before this.
    kw = shortpulse(T)
    u = uniform_pulse(T, Val(1); kw...)
    @test u.err isa T
    @test isfinite(u.err)

    a = track_pulse(T, Val(1); kw..., chunk=T(1//50))
    @test a.worst isa T
    @test a.tracking isa T
    @test isfinite(a.worst)
    @test a.tracking == 1                              # the peak never escapes
    @test a.maxlevel == 2

    # The indicator too: it is where the ε noise floor lives, and a bare
    # `0.01` there would drag every τ into Float64 whatever the field holds.
    forest = Forest{T}((8,); N=8, G=2, periodic=(true,),
                       extents=((zero(T), one(T)),))
    fs = FieldSet(forest, 2)
    fill_by_coordinates!(pulse_exact(1, one(T), T(1//4), T(2//25), zero(T)), fs)
    fill_ghosts!(fs, GhostSchedule(forest, TYPEOPS))
    scales = field_scales(fs)
    @test scales isa Vector{T}
    @test cell_indicator(fs, 1, Inf; scales=scales)[1] isa T
    @test lohner(zero(T), one(T), zero(T), one(T)) isa T
end

@testset "The sine mode computes in the type it is given: T=$T" for T in (Float64,
                                                                         Float32)
    # Not for a MultiFloat: this case is built on `sin` and `cos`, which
    # MultiFloats does not implement. See the header.
    r = wave_errors(T, Val(1); N=8, G=2, ops=TYPEOPS)
    @test r.l2 isa T
    @test r.linf isa T
    @test r.h isa T
    @test isfinite(r.l2)
    @test r.l2 < 0.01
    @test wave_omega(1, one(T), 1) isa T
end

@testset "Float32 reaches the same mesh as Float64" begin
    # The claim that matters for a reduced-precision test run: the *mesh*
    # the criterion chooses is precision-insensitive even where the error
    # is not. If it were not, a Float32 run would stop being a rehearsal
    # for the Float64 one and become a different experiment.
    kw = (roots=8, N=8, G=2, ops=TYPEOPS)
    a64 = track_pulse(Float64, Val(1); kw..., σ=0.08, chunk=0.02)
    a32 = track_pulse(Float32, Val(1); kw..., σ=0.08f0, chunk=0.02f0)

    @test a32.nblocks == a64.nblocks                   # measured 16
    @test a32.maxlevel == a64.maxlevel                 # measured 2
    @test a32.tracking == 1

    # And the error is the discretization's, not the arithmetic's: at
    # h = 1/256 over t = 0.5 the Float32 roundoff is well under the
    # truncation error. Measured 0.09609 against 0.09687.
    @test a32.worst ≈ a64.worst rtol = 0.05
end

@testset "The blast wave survives reduced precision" begin
    # The 2D case, and the one that could have failed on its own: its
    # exact solution is a Hankel quadrature over `besselj0`, which exists
    # for hardware floats only. `blast_reference` keeps that table in
    # Float64 on the host and `blast_exact` converts it into the run's type
    # once, so the ring the run is measured against is the same ring at
    # every precision.
    b64 = track_blast(Float64, Val(2); t_end=0.1)
    b32 = track_blast(Float32, Val(2); t_end=0.1f0)

    @test b32.worst isa Float32
    @test b32.nblocks == b64.nblocks                   # measured 208
    @test b32.maxlevel == b64.maxlevel
    @test b32.covered == b64.covered                   # identical decisions
    @test b32.worst ≈ b64.worst rtol = 0.05
end

@testset "The blast reference is a Float64 table whatever the run is" begin
    # Deliberate, not an oversight: the quadrature's own error is ~9e-7,
    # four orders below any discretization error measured against it, and
    # a reference table is the last thing that should be uploaded to a
    # device as fp64. The conversion happens once, on the host, outside the
    # per-cell closure.
    L, σ = 1.0f0, 0.08f0
    x₀ = (L / 2, L / 2)
    ref = blast_reference(L, x₀, σ; rmax=6σ, nr=501, nk=500)
    @test eltype(ref.J) === Float64
    @test ref.L isa Float64

    exact = blast_exact(Float32, ref, 0.0f0)
    initial = blast_initial(2, L, x₀, σ)
    @test exact(x₀, 1) isa Float32
    @test exact(x₀, 2) isa Float32

    xs = range(0.0f0, L; length=21)
    worst = maximum(abs(exact((x, y), 1) - initial((x, y), 1)) for x in xs, y in xs)
    @test worst < 1.0f-4
end
