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

```bash
julia --project=. -e 'using Pkg; Pkg.test()'   # the acceptance tests
julia --project=bin bin/visualize.jl           # solution, error, error norms
julia --project=bin bin/visualize2d.jl         # the blast wave and its mesh
```

See [CODE.md](CODE.md) for the design and the measured results.
