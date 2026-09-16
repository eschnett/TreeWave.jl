# Running on a device: the claims that hold whether or not there is one.
#
# The mesh decides where the work runs from where the storage is, so
# every driver here takes a `backend` alongside its `T`. What can go
# wrong divides in two, and so does this file:
#
#   - Things a *host* run can check, and which therefore run in CI. The
#     refinement criterion has two forms -- a host loop over cells and a
#     kernel over the same cells through TreeAMR's `firing_boxes` -- and
#     the second is the one a device uses. Both are available on the CPU
#     backend, so they can be compared there, which is the assertion that
#     catches a drift between them. Nothing else in the suite would: a
#     device run that flags differently still runs, and just builds a
#     different mesh.
#
#   - Things only a device can check: that the storage really lands on
#     it, that a whole run reproduces the host run, and that `Float64` is
#     refused where there is no hardware fp64. Those need a device
#     package, which is deliberately not a dependency of this package or
#     of TreeAMR -- CUDA and Metal are large, and the cluster should not
#     build one to run the tests. Add one by hand and name it:
#
#         julia --project=test -e 'using Pkg; Pkg.add("Metal")'
#         TREEWAVE_TEST_BACKEND=metal julia --project=test test/runtests.jl
#
# Unset, the whole file runs on the CPU backend, which is what CI does.

using KernelAbstractions: CPU, Backend, get_backend, supports_float64

const DEVICE_NAME = lowercase(get(ENV, "TREEWAVE_TEST_BACKEND", ""))

if DEVICE_NAME == "cuda"
    using CUDA
elseif DEVICE_NAME == "metal"
    using Metal
elseif !isempty(DEVICE_NAME)
    error("TREEWAVE_TEST_BACKEND must be \"cuda\" or \"metal\", got " *
          "\"$DEVICE_NAME\"")
end

# One entry per backend to sweep, with the float types that backend can
# actually run: a device without hardware fp64 gets `Float32` only, which
# is the case the package's type genericity was written for.
const BACKENDS = let out = Any[("CPU", CPU(), (Float64, Float32))]
    device = if DEVICE_NAME == "cuda"
        CUDA.functional() ? CUDABackend() : nothing
    elseif DEVICE_NAME == "metal"
        Metal.functional() ? MetalBackend() : nothing
    end
    if device === nothing
        isempty(DEVICE_NAME) ||
            @info "TREEWAVE_TEST_BACKEND=$DEVICE_NAME is not functional " *
                  "here; CPU only"
    else
        types = supports_float64(device) ? (Float64, Float32) : (Float32,)
        push!(out, (uppercase(DEVICE_NAME), device, types))
    end
    out
end

const DEVICE = length(BACKENDS) > 1 ? BACKENDS[2] : nothing
const DEVOPS = Operators(prolongation=4, restriction=4)

# The ghost width order-4 operators need, which is the one thing about
# these field sets the centering changes: `p/2 - 1` along a stagger,
# `p/2` across one.
devghosts(centering) = all(==(:vertex), centering) ? 1 : 2

"""A 1D pulse on an eight-block mesh, with the far half refined."""
function pulsefieldset(::Type{T}; backend=CPU(),
                       centering=vertexcentered(1)) where {T}
    forest = Forest{T}((8,); N=8, periodic=(true,),
                       extents=((zero(T), one(T)),))
    # Refined blocks far from the pulse, so the criterion has somewhere to
    # say `Coarsen`: a quiet block at level 0 must stay a bare `Keep`, and
    # a quiet block above it must ask to go away.
    refine!(forest, forest.leaves[6:7])
    balance!(forest)
    fs = FieldSet(forest, 2; G=devghosts(centering), centering=centering,
                  backend=backend)
    fill_by_coordinates!(pulse_exact(1, one(T), T(1//4), T(2//25), zero(T)), fs)
    fill_ghosts!(fs, GhostSchedule(fs, DEVOPS))
    return fs
end

"""The blast wave's initial data in 2D, whose `∂ₜu` is exactly zero."""
function blastfieldset(::Type{T}; backend=CPU(),
                       centering=vertexcentered(2)) where {T}
    forest = Forest{T}((4, 4); N=8, periodic=(true, true),
                       extents=ntuple(_ -> (zero(T), one(T)), 2))
    fs = FieldSet(forest, 2; G=devghosts(centering), centering=centering,
                  backend=backend)
    x₀ = (one(T) / 2, one(T) / 2)
    fill_by_coordinates!(blast_initial(2, one(T), x₀, T(2//25)), fs)
    fill_ghosts!(fs, GhostSchedule(fs, DEVOPS))
    return fs
end

@testset "The device criterion chooses the same mesh as the host one" begin
    # `firing_flags` is not a second criterion; it is the same one walked
    # by a kernel, and a flag vector that differed by one box would build
    # a different mesh without anything failing. Compared on the CPU
    # backend, where both forms are available -- on a device
    # `refine_flags` *is* `firing_flags` and there would be nothing to
    # compare it against.
    #
    # Both centerings, because the two forms derive the stored index of an
    # owned point separately -- `cell_indicator` here, TreeAMR's firing
    # kernel there -- and a stagger is exactly what makes those two
    # derivations able to disagree. Nothing else in the suite would catch
    # it: a run that flagged differently would still run.
    for T in (Float64, Float32)
        for fs in (pulsefieldset(T), blastfieldset(T),
                   pulsefieldset(T; centering=cellcentered(1)),
                   blastfieldset(T; centering=cellcentered(2)))
            scales = field_scales(fs)
            # Two caps: at 2 the firing blocks ask to refine, at 0 they are
            # already at the cap and must ask for the equal-level margin
            # instead -- `(Keep, box)`, the case with no other test.
            for cap in (2, 0)
                kw = (; scales=scales, refine_tol=T(3//10),
                      coarsen_tol=T(3//40), maxlevel_cap=cap)
                flags = firing_flags(fs; kw...)
                @test flags == refine_flags(fs; kw...)
                # And the agreement is not vacuous: each cap has to
                # produce the case it exists to exercise, or the
                # comparison above is between two vectors of bare
                # `Keep`s.
                want = cap == 2 ? Refine : Keep
                @test any(f -> f isa Tuple && f[1] === want, flags)
            end
        end
    end

    # And the box really is the one the boxed threshold asks for: a
    # threshold no cell can reach leaves every block quiet.
    fs = pulsefieldset(Float64)
    scales = field_scales(fs)
    quiet = firing_flags(fs; scales=scales, refine_tol=0.99, coarsen_tol=0.98,
                         maxlevel_cap=2)
    @test all(f -> f isa RegridFlag, quiet)          # no boxes at all
    @test any(==(Coarsen), quiet)                    # the refined blocks go
end

for (name, backend, types) in BACKENDS
    @testset "The geometry follows the storage: $name" begin
        # The RHS kernel reads a spacing per block, and that array is the
        # one thing the mesh does not place for the application. A host
        # `Vector` here would be the wrong memory and the failure would
        # be a crash inside the kernel rather than anything nameable.
        for T in types
            fs = pulsefieldset(T; backend=backend)
            problem = WaveProblem(fs, GhostSchedule(fs, DEVOPS))
            @test typeof(get_backend(problem.spacings)) === typeof(backend)
            @test eltype(problem.spacings) === T

            # And the reverse direction, which is how a viewer reads a
            # device run: same forest, same numbers, on the host.
            host = hostcopy(fs)
            @test get_backend(host.work) isa CPU
            @test host.work isa Array
            @test host.forest === fs.forest
            @test host.work == Array(fs.work)
            @test host.G == fs.G                     # the whole layout, not
            @test host.centering == fs.centering     # merely the forest
            backend isa CPU && @test host === fs     # nothing to copy
        end
    end
end

if DEVICE !== nothing
    devname, devbackend, devtypes = DEVICE

    @testset "$devname at $T reproduces the host run" for T in devtypes
        # The whole point of the exercise: the same study, the same mesh.
        # The refinement decisions are threshold comparisons on per-cell
        # arithmetic that is identical on both, so the mesh must agree
        # exactly. The errors must not: the per-block sums inside
        # `volume_weighted_norm` are pairwise on the host and sequential
        # in the kernel, so they differ in their last bits and the
        # difference grows with the number of steps.
        kw = (; N=8, G=1, ops=DEVOPS)
        s0 = wave_errors(T, Val(1); kw...)
        s1 = wave_errors(T, Val(1); kw..., backend=devbackend)
        @test s1.nblocks == s0.nblocks
        @test s1.nsteps == s0.nsteps
        @test s1.l2 isa T
        @test s1.l2 ≈ s0.l2 rtol = 1e-3

        pkw = (; roots=8, N=8, G=1, ops=DEVOPS, σ=T(2//25), chunk=T(1//50))
        p0 = track_pulse(T, Val(1); pkw...)
        p1 = track_pulse(T, Val(1); pkw..., backend=devbackend)
        @test p1.nblocks == p0.nblocks
        @test p1.maxlevel == p0.maxlevel
        @test p1.tracking == p0.tracking
        @test p1.worst ≈ p0.worst rtol = 1e-2

        # The solution itself, per chunk, which is the tight comparison and
        # the one worth making: no cancellation, so this measures the
        # divergence between the two arithmetics directly. Measured 7e-6
        # at Float32 on both centerings.
        norms(backend) = begin
            ns = T[]
            track_blast(T, Val(2); t_end=T(1//10), backend=backend,
                        observer=(fs, t, u) ->
                            push!(ns, volume_weighted_norm(fs, u; p=Inf)))
            ns
        end
        n0, n1 = norms(CPU()), norms(devbackend)
        @test length(n1) == length(n0)
        @test all(isapprox(a, b; rtol=1e-4) for (a, b) in zip(n0, n1))

        b0 = track_blast(T, Val(2); t_end=T(1//10))
        b1 = track_blast(T, Val(2); t_end=T(1//10), backend=devbackend)
        @test b1.nblocks == b0.nblocks
        @test b1.maxlevel == b0.maxlevel
        @test b1.growth == b0.growth
        @test b1.covered == b0.covered               # identical decisions

        # The *error* cannot be held that tightly, and the factor is
        # arithmetic rather than a fudge. `worst` is the L∞ of
        # `u - u_exact` over the whole state, where `∂ₜu` reaches 17 while
        # the error is 0.017 -- a cancellation of three orders. So the 7e-6
        # relative divergence above lands here as ~1e-2, and measured it
        # does: 1.3e-2 vertex-centred, 9e-4 cell-centred, the difference
        # being only where each run's maximum happens to fall. The mesh
        # assertions above are the exact ones; this is the sanity check
        # that the two runs are the same run.
        @test b1.worst ≈ b0.worst rtol = 3e-2
    end

    if !supports_float64(devbackend)
        @testset "Float64 is refused where there is no hardware fp64" begin
            # Not a limitation to work around: it is why the package was
            # made generic in its float type in the first place, and the
            # mesh says so at the one place that can still be fixed --
            # when the storage is allocated, rather than from inside a
            # kernel. See "Precision" in `CODE.md`.
            @test_throws "no hardware Float64" track_pulse(Float64, Val(1);
                                                           roots=8, N=8, G=1,
                                                           ops=DEVOPS,
                                                           backend=devbackend)
        end
    end
end
