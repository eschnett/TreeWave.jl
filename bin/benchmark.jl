# Thread-scaling measurement (M5).
#
#     julia -t N --project=. bin/benchmark.jl
#
# Note the `--project=.` -- this is the one script in `bin/` that runs
# against the *package* environment rather than `bin/Project.toml`. It
# needs no CairoMakie, and asking a compute node to build Cairo in order
# to time a Laplacian would be absurd.
#
# Prints one tab-separated `threads<TAB>phase<TAB>seconds` line per phase
# for the thread count Julia was started with, preceded by a comment line
# describing the problem. Run it once per thread count and read the
# columns against each other; `bin/benchmark.sbatch` does that on a node.
#
# The sizes are flags rather than constants because the same script has
# to serve a laptop and a 64-core node.

using TreeWave

function main(args)
    dim = 2
    # 1792 blocks of 128^2 on a two-level mesh, about 29M cells: at
    # KernelAbstractions' CPU grain of 1024 cells that is 28672 workgroups
    # per launch, so even 64 threads get hundreds each and the measurement
    # is not reading a partly idle machine.
    n = 128
    roots = 32
    reps = 5
    steps = 10
    σ = 0.08
    # The end-to-end driver run is a separate question from the phase
    # table and costs more than all of it, so it is opt-in -- and it has
    # its own size, because an adaptive run cannot use the phase table's:
    # `refinement_buffer` rejects a chunk that crosses a finest-level
    # block, and sigma has to follow the coarse spacing or the criterion
    # finds nothing to refine.
    driver = false
    driver_roots = 16
    driver_n = 16
    for a in args
        if startswith(a, "--dim=")
            dim = parse(Int, a[7:end])
        elseif startswith(a, "--n=")
            n = parse(Int, a[5:end])
        elseif startswith(a, "--roots=")
            roots = parse(Int, a[9:end])
        elseif startswith(a, "--reps=")
            reps = parse(Int, a[8:end])
        elseif startswith(a, "--steps=")
            steps = parse(Int, a[9:end])
        elseif startswith(a, "--sigma=")
            σ = parse(Float64, a[9:end])
        elseif startswith(a, "--driver-roots=")
            driver_roots = parse(Int, a[16:end])
        elseif startswith(a, "--driver-n=")
            driver_n = parse(Int, a[12:end])
        elseif a == "--driver"
            driver = true
        elseif a == "--no-driver"
            driver = false
        else
            error("unknown argument $a; expected --dim=, --n=, --roots=, \
                   --reps=, --steps=, --sigma=, --driver, --no-driver, \
                   --driver-roots=, --driver-n=")
        end
    end
    dim in (1, 2, 3) || error("--dim must be 1, 2 or 3; got $dim")

    threads = Threads.nthreads()
    result = benchmark_phases(Val(dim); N=n, roots=roots, reps=reps, steps=steps,
                              σ=σ)
    s = result.sizes
    println("# threads=", threads, " D=", s.D, " N=", s.N, " roots=", s.roots,
            " blocks=", s.blocks, " cells=", s.cells,
            " work=", round(s.workbytes / 2^20; digits=1), "MiB")
    for (name, seconds) in result.timings
        println(threads, "\t", name, "\t", round(seconds; sigdigits=4))
    end

    # Cell updates per second of the whole step, which is the throughput
    # an application would quote -- four RHS evaluations and the stage
    # arithmetic included, not the kernel alone.
    step = first(t for (name, t) in result.timings if name == "step")
    println(threads, "\tcell_updates_per_second\t",
            round(s.cells / step; sigdigits=4))

    if driver
        # The calibrated run has sigma/h0 = 5.12 at roots = N = 8; holding
        # that ratio is what makes a bigger run a bigger *hierarchy*
        # rather than a finer mesh the criterion declines to refine.
        h0 = 1.0 / (driver_roots * driver_n)
        run = benchmark_driver(; roots=driver_roots, N=driver_n, σ=5.12 * h0,
                               chunk=0.01, t_end=0.05, reps=2)
        println(threads, "\ttrack_blast\t", round(run.seconds; sigdigits=4))
        println("# track_blast roots=", driver_roots, " N=", driver_n,
                " blocks=", run.nblocks, " growth=", round(run.growth; digits=2),
                " worst=", round(run.worst; sigdigits=4))
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
