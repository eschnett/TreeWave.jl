# TreeWave.jl

[![CI](https://github.com/eschnett/TreeWave.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeWave.jl/actions/workflows/CI.yml)

`TreeWave` solves the scalar wave equation in 2nd-order form, as a
sample application for
[TreeAMR](https://github.com/eschnett/TreeAMR.jl).

That is `∂ₜu = v`, `∂ₜv = ∇²u` on a periodic box, with three initial
conditions: a standing sine mode, used to measure convergence order; a
travelling super-Gaussian pulse, used to exercise a refined region that
follows it; and a 2D radial blast wave, whose feature loses amplitude as
it spreads and whose refined region therefore has to grow.

Every driver takes the floating-point type to run in as a leading
argument, defaulting to `Float64` — `track_pulse(Float32, Val(1))` — so a
case can be run in single precision on a device with no hardware fp64, or
in a MultiFloats software type. See "Precision" in [CODE.md](CODE.md) for
what that does and does not buy.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'   # the acceptance tests
julia --project=bin bin/visualize.jl           # solution, error, error norms
julia --project=bin bin/visualize.jl --type=f32   # the same run, single precision
julia --project=bin bin/visualize2d.jl         # the blast wave and its mesh
julia -t auto --project=. bin/benchmark.jl     # where the time goes
```

It runs threaded with nothing to configure: start Julia with `-t` and
every loop over blocks — TreeAMR's and this package's — is parallel, and
the answer is bit-identical whatever the thread count. What that does
*not* buy is the whole step, because the integrator's stage arithmetic
stays serial; the measured numbers are in [CODE.md](CODE.md).

It also runs on a GPU with nothing to configure but where the storage
goes: every driver takes a KernelAbstractions `backend` alongside its
type, and both viewers and the benchmark take `--backend=cuda|metal`. No
device package is a dependency of this one. What that buys is the
compute-bound work — 8× on initial data on an M3 Pro — and, on unified
memory, nothing at all on the memory-bound right-hand side; "Running on a
device" in [CODE.md](CODE.md) has the table and the reason.

See [CODE.md](CODE.md) for the design and the measured results.
