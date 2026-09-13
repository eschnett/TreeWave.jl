# Multi-threading (M5).
#
# There is no switch to test: TreeAMR's kernels and host-side passes
# thread themselves, and this package's own loops over blocks are written
# to match. What has to be guarded is the *invariant* that makes that
# safe to rely on — the answer does not move when the thread count does,
# to the last bit — because nothing else in the suite would notice if it
# did. Every assertion here is exact equality, never `≈`.

include("thread_workload.jl")

@testset "A run is bit-identical across thread counts" begin
    # The acceptance test. The thread count is a command-line argument to
    # Julia and cannot be changed from inside a running session, so the
    # comparison is against a subprocess started at a different count.
    # A reduction partitioned by thread rather than by block would pass
    # every other test in this suite and fail here.
    reference = thread_digests()
    @test length(reference) == 8

    other = Threads.nthreads() == 1 ? max(2, min(4, Sys.CPU_THREADS)) : 1
    script = joinpath(@__DIR__, "thread_workload.jl")
    # The *active* project, not `test/`: under `Pkg.test` the tests run in
    # a sandbox and `test/Project.toml` has no manifest of its own.
    project = Base.active_project()
    out = read(`$(Base.julia_cmd()) --threads=$other --project=$project $script`,
               String)

    @test split(chomp(out), '\n') == reference
end

@testset "Threaded reductions reproduce the serial ones exactly" begin
    # The cheap, local half of the claim above, and the one that fails
    # legibly: if `field_scales` or the Hankel contraction is ever
    # rewritten to accumulate into shared state, this says so in
    # milliseconds and names the function, where the subprocess test says
    # only that two long outputs differ.
    forest = wave_forest(Val(2), 8, 2; roots=4)
    fs = FieldSet(forest, 2)
    fill_by_coordinates!(pulse_exact(2, 1.0, 0.25, 0.08, 0.0), fs)

    serial_scales(fs; vars=1:fs.nvars) =
        [maximum(b -> maximum(abs, interiorview(fs, b, v)), 1:nblocks(fs))
         for v in vars]
    @test field_scales(fs) == serial_scales(fs)

    function serial_coverage(fs)
        peak = maximum(b -> maximum(abs, interiorview(fs, b, 1)), 1:nblocks(fs))
        finest = maximum(b -> level(blockkey(fs, b)), 1:nblocks(fs))
        hot, fine = 0, 0
        for b in 1:nblocks(fs)
            isfine = level(blockkey(fs, b)) == finest
            for value in interiorview(fs, b, 1)
                abs(value) > peak / 2 || continue
                hot += 1
                isfine && (fine += 1)
            end
        end
        return hot == 0 ? 1.0 : fine / hot
    end
    @test blast_coverage(fs) === serial_coverage(fs)

    # The quadrature: the table is filled column by column in parallel,
    # and contracted row by row, so both halves are checked against the
    # loops they replaced. Small `nr`/`nk` -- this is about arithmetic
    # order, not about the integral being accurate.
    ref = blast_reference(1.0, (0.5, 0.5), 0.08; rmax=0.5, nr=201, nk=150)
    @test ref.J == [besselj0(k * r) for r in ref.rs, k in ref.ks]

    function serial_table(ref, t)
        nr, nk = size(ref.J)
        us, vs = zeros(Float64, nr), zeros(Float64, nr)
        for j in 1:nk
            w, k = ref.ws[j], ref.ks[j]
            wc, wsn = w * cos(k * t), w * sin(k * t) * k
            for i in 1:nr
                us[i] += ref.J[i, j] * wc
                vs[i] -= ref.J[i, j] * wsn
            end
        end
        return (us=us, vs=vs)
    end
    table = blast_radial_table(ref, 0.2)
    @test table.us == serial_table(ref, 0.2).us
    @test table.vs == serial_table(ref, 0.2).vs
end
