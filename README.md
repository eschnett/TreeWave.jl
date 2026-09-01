# TreeWave.jl

[![CI](https://github.com/eschnett/TreeWave.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeWave.jl/actions/workflows/CI.yml)

`TreeWave` solves the scalar wave equation in 2nd-order form, as a
sample application for
[TreeAMR](https://github.com/eschnett/TreeAMR.jl).

That is `∂ₜu = v`, `∂ₜv = ∇²u` on a periodic box, with two initial
conditions: a standing sine mode, used to measure convergence order, and a
travelling super-Gaussian pulse, used to exercise a refined region that
follows it.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'   # the acceptance tests
julia --project=bin bin/visualize.jl           # solution, error, error norms
```

See [CODE.md](CODE.md) for the design and the measured results.
