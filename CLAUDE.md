# Working notes for Claude in TreeWave.jl

Read `CODE.md` first — it is the design document and states *why* things
are the way they are. This file is only about mechanics.

**Contents**

- [Where things go](#where-things-go)
- [What this package is](#what-this-package-is)
- [Commands](#commands)
- [Things that have bitten before](#things-that-have-bitten-before)
- [Conventions](#conventions)
- [Repository facts](#repository-facts)

## Where things go

| file | for | holds |
|---|---|---|
| `README.md` | users | what the package does, how to run it, a pointer to `CODE.md` |
| `CLAUDE.md` | agents; loaded into every session, so kept short | what to read, commands, sharp edges as short rules pointing at `CODE.md`, repository facts |
| `CODE.md` | anyone changing the code | the current design and the reasons for it, the tests, current measurements, limitations, possible extensions — in the present tense, with **(open)** or **(proposed)** on what is not settled |
| `HISTORY.md` | anyone asking how it got here | who decided what and when, alternatives dropped, measurements later ones replaced, bugs since fixed, dependency and release history, by topic in `CODE.md`'s order |
| `PLAN.md` | — | does not exist; concrete planned work, with how one knows each step is done, would go there |

The test between `CODE.md` and `HISTORY.md` is whether someone changing
the code today needs it. To amend the documents, fix `CODE.md` to
describe the current state, and record in `HISTORY.md` what changed, when
and why. Keep each file's table of contents current when you add, rename
or move a heading — `##` and `###` in `CODE.md`, `PLAN.md` and
`HISTORY.md`, `##` in `README.md` and here — and update every reference
to a heading you rename, in code comments and docstrings too.

## What this package is

The sample application for [TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl).
TreeAMR is the mesh and contains no physics; TreeWave is the physics. If a
change you are about to make is about trees, ghost cells, or interpolation,
it belongs upstream in TreeAMR, not here (`CODE.md`, "Scope and
non-goals").

## Commands

Run the tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

And, because the RHS kernel is `@inbounds`, also run them with the
checks put back -- this is what CI does, and it is the only thing that
can catch the annotation being wrong:

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--check-bounds=yes"])'
```

To check a change end-to-end the way CI will see it:

```bash
git archive HEAD | tar -x -C /tmp/clean && julia --project=/tmp/clean -e 'using Pkg; Pkg.test()'
```

Run the viewer (separate environment, so CairoMakie is not a package
dependency):

```bash
julia --project=bin bin/visualize.jl
```

`Pkg.test` does not inherit `-t`, so the threaded paths in the suite
itself need it passed explicitly (the thread-independence test spawns its
own subprocess either way):

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

Run anything on a GPU. No device package is a dependency of this package
or of TreeAMR, so this needs an environment of its own -- once, then
reuse it:

```bash
julia --project=/tmp/twgpu -e 'using Pkg; Pkg.develop(path=".")
    Pkg.add(["Metal", "KernelAbstractions", "MultiFloats",
             "OrdinaryDiffEqLowOrderRK", "SciMLBase", "SpecialFunctions",
             "TreeAMR", "Test"])'
TREEWAVE_TEST_BACKEND=metal julia --project=/tmp/twgpu test/runtests.jl
julia --project=/tmp/twgpu bin/benchmark.jl --backend=metal --type=f32
```

`TREEWAVE_TEST_BACKEND` unset runs `test/device_tests.jl` on the CPU
backend, which is what CI does and is where the criterion-equivalence
test lives. The viewers need CairoMakie too, so they want a second such
environment rather than `bin/` (whose `Project.toml` is tracked and must
not gain a device dependency).

Thread scaling (`bin/benchmark.jl`, swept by `bin/benchmark.sbatch`,
which is both a batch job and an ordinary shell script). Note the
`--project=.`: this is the one script in `bin/` that does *not* use the
viewer environment.

```bash
TREEWAVE_THREADS="1 2 4 8" TREEWAVE_ARGS="--dim=2 --n=64 --roots=16" \
    bin/benchmark.sbatch
```

**Julia's precompilation cache is per CPU target.** Precompiling on a
cluster's login node does not help its compute nodes if the two are
different machines -- on Symmetry the login node is Intel and the AMD
nodes are not, so a batch job precompiles everything again on its own
wall clock. Precompile in an interactive `srun` on the same node type
before submitting, and do not submit two jobs that would precompile the
same depot at once.

The full suite takes about 60 seconds, of which the thread-independence
test is 10 -- it runs `test/thread_workload.jl` in a subprocess, which
pays Julia's startup and `OrdinaryDiffEq`'s load time again. The other
slow parts are the type tests, the two convergence sweeps and the 3D
smoke test; the pulse test is about 5 s and the device tests about 8,
nearly all of it compiling the two firing kernels.

## Things that have bitten before

Each is the rule; the `CODE.md` section named after it has the reason.

- **TreeAMR comes from the General registry, not from the local
  checkout.** A `~/src/jl/TreeAMR` working copy is *not* what is being
  tested, and neither is upstream `main` any more. If a TreeAMR change is
  needed, say so — the route is a TreeAMR release, not an edit to that
  checkout and not a branch pin here ("File layout"). Note the remote has
  no `master` branch — only `main` and `gh-pages`.
- **There is exactly one TreeAMR version in the repository, and it is
  `[compat]` in `Project.toml`.** If you raise it, raise it there and
  nowhere else; grep for `TreeAMR` to confirm nothing else names a
  version ("File layout").
- **Deleting a `[sources]` entry does not un-track the branch.** Run
  `Pkg.free("TreeAMR")` in the root *and* in `bin/`. A freed entry says
  `registries = "General"` and carries no `repo-rev`; check for that
  rather than for the absence of an error ("File layout").
- **The floor is Julia 1.13, and CI tests nothing else.** Do not add
  version checks or workarounds for older Julias ("File layout").
- **The pulse's `∂ₜu` sign is load-bearing.** `u = G(x - t)` gives
  `∂ₜu = -G'` ("Travelling super-Gaussian pulse").
- **The super-Gaussian order `n` is not free.** It defaults to 1. If you
  raise `n`, raise the resolution of every pulse run and re-measure — do
  not adjust the tolerances to fit ("Travelling super-Gaussian pulse").
- **A scale-free error indicator is a trap.** The noise floor must be
  referred to a *global* amplitude (`field_scales`). If you touch
  `lohner`, keep that ("The noise floor must be global").
- **The refinement box threshold is `coarsen_tol`, not `refine_tol`**
  ("Two thresholds, meaning two different things").
- **Don't name a keyword `maxlevel`.** It shadows TreeAMR's exported
  `maxlevel(forest)`; the keyword here is `maxlevel_cap` ("Calibrated
  thresholds").
- **Order-4 prolongation is required** for 2nd-order convergence on a
  refined mesh, even though the RHS stencil reaches only one cell.
  Vertex-centred needs `G = 1`; cell-centred needs `G = 2` *and* order-4
  restriction. Never give `ops` or `G` a default that hides it ("Operator
  order").
- **Vertex centring is the default, and cell-centred is kept as the
  comparison.** Do not delete one half to make a change smaller
  ("Centerings").
- **`coordinates(fs, b, idx)` indexes the stored array**, so owned point
  `i` is `idx = i + fs.G[d]`; ask it for a position rather than
  reconstructing one ("Centerings").
- **`fs.G` is an `NTuple{D,Int}`, not an `Int`** ("Centerings").
- **A schedule belongs to a *layout*, and a regrid rebuilds it.** Write
  `GhostSchedule(fs, ops)`, and give `regrid!` `fs => schedule` pairs
  ("Centerings").
- **`Base` is not generic even though the mesh is.** Use `wrap` /
  `ceilint` / `floorint` / `tofloat64` from `src/precision.jl` rather than
  the `Base` spellings; each one throws at `Float64x2` and works at
  `Float64`, so the test suite is the only thing that will tell you
  ("Base does not come along").
- **A decimal literal in a `T` expression is a leak, not a style point.**
  Write `T(1//100)`, not `0.01` ("The rule").
- **Do not add a Float32 convergence-rate assertion** ("What `Float32`
  cannot do").
- **The RHS kernel's `@inbounds` is load-bearing in both directions.**
  `.github/workflows/CI.yml` *states* `check_bounds: 'yes'`; do not drop
  that line to make a CI edit smaller. Run the suite both ways ("What
  bounds checking costs").
- **`@inbounds` does not cross a call that is not
  `@propagate_inbounds`.** If `cell_tau` in `src/refinement.jl` ever
  wants it, annotate inside `cell_tau`; an `@inbounds` at the call site
  would silently do nothing ("What bounds checking costs").
- **A parallel loop here must be bit-identical to the serial one.** Write
  one slot per block and combine the partials in a fixed order, never
  accumulate into shared state. `test/threading_tests.jl` fails if this
  is broken, and nothing else in the suite would ("Bit-identical, not
  merely to roundoff"; `blast_radial_table` under "What was left for the
  application to thread").
- **Never thread anything a TreeAMR callback can reach**: no
  `Threads.@threads` inside `cell_indicator`, `refine_mark` or an
  initial-data closure. `field_scales` is evaluated at `refine_flags`'
  own call site, outside the flagging pass; keep it that way ("The
  callbacks are called concurrently, so they must be pure").
- **A callback must capture no `Type` and no host array.** Write
  `zero(x[1])`, not `zero(T)`; upload a captured `Vector` (`to_backend`)
  or make it a tuple; accumulate in a loop rather than through a
  generator ("Callbacks become kernel arguments").
- **Loading a device package inside `main` is a world-age bug.** Use
  `bin/backend.jl`'s `withbackend`, which puts the check and the work
  inside one `invokelatest`, rather than calling `resolvebackend` and
  carrying on ("Running on a device").
- **The mesh sizes the tests use are two orders of magnitude too small
  for a GPU.** A whole `track_blast` is 7× *slower* on Metal at 52k
  cells; that is launch overhead, not a regression. Measure phases at
  `--n=128 --roots=32` (29.4M cells) ("What it buys on this hardware, and
  what it does not").
- **A reduction that crosses blocks goes through `mesh_mapreduce`, never
  through a `maximum` or `sum` of `block_mapreduce`.** That covers
  `field_scales`, `blast_coverage` and `track_pulse`'s tracking measure;
  a diagnostic that needs the tree passes it as a 0/1 `weight` ("What an
  application has to do that the mesh does not"). Its old, unexported
  name was `TreeAMR.block_partials`; prose here still saying that is
  stale.
- **`julia -t N` asks for `N + 1` threads**; `-t N,0` asks for exactly
  `N`. `bin/benchmark.sbatch` does not pin ("What it buys, and what caps
  it").

## Conventions

Match TreeAMR's style, since the two are read together:

- 4-space indent, wrap at about 80 columns.
- `return` on the last line of any non-trivial function.
- `ntuple(_ -> x, D)` / `ntuple(d -> f(d), Val(D))` rather than
  comprehensions in kernel-adjacent code.
- Unicode in mathematical contexts (`∇²`, `∂ₜ`, `σ`, `ω`, `≈`, `x̄`).
- Keyword-heavy driver signatures, with no default for anything the caller
  must think about.
- Docstrings are prose-first: what it is, then *why* it is that way.
- **Testset names are claims**, not labels — "Energy stays bounded", not
  "energy test". Each testset opens with a comment naming the failure mode
  it guards.
- Record measured numbers in `CODE.md` ("Measured results") when they
  change.

## Repository facts

- `Manifest.toml` is gitignored (both root and `bin/`), as is `TODO.md`,
  the owner's own list: do not edit it, and move nothing into or out of
  it. `CLAUDE.md` is *not* — it is committed, so an edit to it lands in
  the diff and belongs in the commit message like any other change. So
  are `CODE.md`, `HISTORY.md` and `README.md`.
- **`bin/` has its own Manifest**, so `Pkg.update("TreeAMR")` in the root does
  not touch it. After a TreeAMR change, update both or the viewer fails with a
  `MethodError` on an API the tests are already using.
- **`bin/backend.jl` is `include`d by all three scripts in `bin/`**, two of
  which run against `bin/Project.toml` and one against the root project. So
  it may use only what *both* environments have -- today that is
  `KernelAbstractions`, which is why it was added to `bin/Project.toml`.
- `bin/output/` is gitignored; the viewer writes PNGs there.
- `main` is committed and published: `origin` is
  `git@github.com:eschnett/TreeWave.jl.git`, and `main` tracks it. Work on a
  branch; do not push, open a pull request, or merge to `main` without being
  asked.
