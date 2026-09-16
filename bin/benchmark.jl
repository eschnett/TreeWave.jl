# Thread-scaling (M5) and device (M6) measurement.
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
#
# `--backend=cuda` or `--backend=metal` runs the same phases on a device,
# in the same format, so a device column and a host column can be read
# side by side. The device package is *not* a dependency of TreeWave --
# neither CUDA nor Metal is a dependency of TreeAMR either, and the
# cluster should not have to build one to time a Laplacian -- so the
# script needs an environment that has it, which is one command to make:
#
#     julia --project=/tmp/twgpu -e 'using Pkg; Pkg.develop(path = ".");
#                                     Pkg.add("Metal")'
#     julia --project=/tmp/twgpu bin/benchmark.jl --backend=metal --type=f32
#
# `--type=f32` is not optional on a device without hardware fp64: a
# `Float64` field set is refused there, with the reason. See "Running on a
# device" in `CODE.md`.
#
# `--centering=cell` runs the same phases on the cell-centred layout, so
# that column can be read against the vertex one too.

using TreeAMR: cellcentered, vertexcentered
using TreeWave

include(joinpath(@__DIR__, "backend.jl"))

const FLOATTYPES = Dict("f32" => Float32, "f64" => Float64)

# `--centering=` selects where the values sit, defaulting to vertex as the
# rest of the package does. It changes the *stored* array -- a vertex-like
# dimension holds one plane more -- so the reported `work=` differs between
# the two columns even at an identical cell count.
const CENTERINGS = Dict("vertex" => vertexcentered, "cell" => cellcentered)

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
    T = Float64
    makecentering = vertexcentered
    backendname = "cpu"
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
        elseif startswith(a, "--type=")
            tag = a[8:end]
            haskey(FLOATTYPES, tag) ||
                error("unknown --type=$tag; expected f32 or f64")
            T = FLOATTYPES[tag]
        elseif startswith(a, "--centering=")
            tag = a[13:end]
            haskey(CENTERINGS, tag) ||
                error("unknown --centering=$tag; expected vertex or cell")
            makecentering = CENTERINGS[tag]
        elseif startswith(a, "--backend=")
            backendname = a[11:end]
        elseif a == "--driver"
            driver = true
        elseif a == "--no-driver"
            driver = false
        else
            error("unknown argument $a; expected --dim=, --n=, --roots=, \
                   --reps=, --steps=, --sigma=, --type=, --centering=, \
                   --backend=, --driver, --no-driver, --driver-roots=, \
                   --driver-n=")
        end
    end
    dim in (1, 2, 3) || error("--dim must be 1, 2 or 3; got $dim")
    centering = makecentering(dim)

    threads = Threads.nthreads()
    # Everything that touches the storage runs inside `withbackend`, which
    # is what makes a device package loaded a moment ago visible to it.
    return withbackend(backendname, T) do backend
        result = benchmark_phases(T, Val(dim); N=n, roots=roots, reps=reps,
                                  steps=steps, σ=T(σ), centering=centering,
                                  backend=backend)
        s = result.sizes
        println("# threads=", threads, " type=", s.floattype,
                " backend=", s.backend,
                " centering=", all(==(:vertex), s.centering) ? "vertex" : "cell",
                " D=", s.D, " N=", s.N,
                " roots=", s.roots, " blocks=", s.blocks, " cells=", s.cells,
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
            # The calibrated run has sigma/h0 = 5.12 at roots = N = 8;
            # holding that ratio is what makes a bigger run a bigger
            # *hierarchy* rather than a finer mesh the criterion declines
            # to refine.
            h0 = 1.0 / (driver_roots * driver_n)
            run = benchmark_driver(T; roots=driver_roots, N=driver_n,
                                   σ=5.12 * h0, chunk=0.01, t_end=0.05,
                                   reps=2, centering=makecentering(2),
                                   backend=backend)
            println(threads, "\ttrack_blast\t", round(run.seconds; sigdigits=4))
            println("# track_blast roots=", driver_roots, " N=", driver_n,
                    " blocks=", run.nblocks,
                    " growth=", round(run.growth; digits=2),
                    " worst=", round(run.worst; sigdigits=4))
        end
        return nothing
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
