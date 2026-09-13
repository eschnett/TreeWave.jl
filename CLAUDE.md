# Working notes for Claude in TreeWave.jl

Read `CODE.md` first — it is the design document and states *why* things
are the way they are. This file is only about mechanics.

## What this package is

The sample application for [TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl).
TreeAMR is the mesh and contains no physics; TreeWave is the physics. If a
change you are about to make is about trees, ghost cells, or interpolation,
it belongs upstream in TreeAMR, not here.

## Commands

Run the tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
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

Thread scaling (`bin/benchmark.jl`, swept by `bin/benchmark.sbatch`,
which is both a batch job and an ordinary shell script). Note the
`--project=.`: this is the one script in `bin/` that does *not* use the
viewer environment.

```bash
TREEWAVE_THREADS="1 2 4 8" TREEWAVE_ARGS="--dim=2 --n=64 --roots=16" \
    bin/benchmark.sbatch
```

The full suite takes about 60 seconds, of which the thread-independence
test is 10 -- it runs `test/thread_workload.jl` in a subprocess, which
pays Julia's startup and `OrdinaryDiffEq`'s load time again. The other
slow parts are the type tests, the two convergence sweeps and the 3D
smoke test; the pulse test is about 5 s.

## Things that have bitten before

- **TreeAMR is pinned to the GitHub `main`, not to the local checkout.** A
  `~/src/jl/TreeAMR` working copy is *not* what is being tested. If a
  TreeAMR change is needed, say so rather than editing that checkout and
  assuming the tests see it. Note the remote has no `master` branch — only
  `main` and `gh-pages`.
- **`[sources]` in `Project.toml` is what makes a clean checkout resolve.**
  TreeAMR is unregistered and `Manifest.toml` is untracked, so without it
  `Pkg.instantiate()` fails with "expected package TreeAMR to be
  registered". That entry is why the Julia floor is 1.11, not 1.10. To check
  a change end-to-end the way CI will see it:
  `git archive HEAD | tar -x -C /tmp/clean && julia --project=/tmp/clean -e 'using Pkg; Pkg.test()'`
- **The pulse's `∂ₜu` sign is load-bearing.** `u = G(x - t)` gives
  `∂ₜu = -G'`. The wrong sign does not reverse the pulse, it splits it,
  and the failure looks like an instability rather than like bad initial
  data. This was a real bug; see `CODE.md`.
- **The super-Gaussian order `n` is not free.** It defaults to 1. At
  `n = 8` the pulse's shoulder is under four cells wide on the mesh the
  tests use and the L∞ error stays O(40) however correct the code is. If
  you raise `n`, raise the resolution of every pulse run and re-measure —
  do not adjust the tolerances to fit.
- **A scale-free error indicator is a trap.** Löhner's τ with its classic
  *local* noise floor scored 0.986 on the pulse's far tail, where `u ~ 1e-16`,
  and refined the whole domain. The floor must be referred to a *global*
  amplitude (`field_scales`). If you touch `lohner`, keep that.
- **The refinement box threshold is `coarsen_tol`, not `refine_tol`.** A block
  at the cap has stopped being under-resolved, so keying the box on
  `refine_tol` silently disables the `(Keep, box)` travelling margin — the
  mesh still works, just worse, which is the hard kind of bug to notice.
- **Don't name a keyword `maxlevel`.** TreeAMR exports `maxlevel(forest)`, and
  a keyword of that name shadows it inside the function body; `maxlevel(forest)`
  then tries to call an integer. The keyword here is `maxlevel_cap`, which also
  says what it is — at calibrated tolerances it never binds.
- **Order-4 operators and `G = 2` are required** for 2nd-order
  convergence on a refined mesh, even though the RHS stencil reaches only
  one cell. This is TreeAMR's interface-order rule; `CODE.md` explains it.
  Never give `ops` or `G` a default that hides it.
- `cell_center(forest, key, idx)` indexes the **stored** array, so
  interior cell `i` is `idx = i + G`. Off-by-`G` here produces plots that
  look almost right.
- **`Base` is not generic even though the mesh is.** MultiFloats defines no
  `rem` (so `mod` throws), no conversion to `Integer` (so `ceil(Int, x)`
  throws), and a conversion only to its own limb type (so
  `Float64(::Float32x2)` throws while `Float32(::Float32x2)` works). Use
  `wrap` / `ceilint` / `floorint` / `tofloat64` from `src/precision.jl`
  rather than the `Base` spellings; each one throws at `Float64x2` and
  works at `Float64`, so the test suite is the only thing that will tell
  you.
- **A decimal literal in a `T` expression is a leak, not a style point.**
  `0.01` is an fp64 operand and widens the whole expression; write
  `T(1//100)`. At `Float64` the two are bit-identical, which is what let
  every measured number stay put when the drivers went generic.
- **Do not add a Float32 convergence-rate assertion.** Roundoff in the
  Laplacian is `eps/h²`, which at `Float32` and `h = 1/256` is comparable
  to the discretization error the sweeps measure. `CODE.md` records this;
  `test/type_tests.jl` asserts the mesh the criterion chooses instead,
  which *is* precision-insensitive.
- **A parallel loop here must be bit-identical to the serial one.** That is
  TreeAMR's M5 invariant and the whole application inherits it: write one
  slot per block and combine the partials in a fixed order, never
  accumulate into shared state. `blast_radial_table` is the instructive
  case — the mode loop *cannot* be split, because every `k` contributes to
  every `r`; partitioning the radii instead keeps each sum in its original
  order. `test/threading_tests.jl` fails if this is broken, and nothing
  else in the suite would.
- **Never thread anything a TreeAMR callback can reach.** `flag_blocks`
  and `fill_by_coordinates!` call their callbacks concurrently, so a
  `Threads.@threads` inside `cell_indicator`, `refine_mark` or an
  initial-data closure would nest one parallel loop inside another.
  `field_scales` is safe only because it is evaluated at `refine_flags`'
  own call site, outside the flagging pass; keep it that way.
- **`julia -t N` asks for `N + 1` threads.** The interactive thread is
  added on top of the count given, so with `JULIA_EXCLUSIVE=1` pinning one
  thread per core, `-t 64` on a 64-core node dies with "Too many threads
  requested for JULIA_EXCLUSIVE option" -- at the *end* of a sweep that
  had already run for eight minutes. `-t N,0` asks for exactly `N` on
  1.13 and is rejected by 1.11 ("n and m must be integers >= 1"), which
  is the cluster's version, so `bin/benchmark.sbatch` does not pin at
  all rather than pin every point of the sweep but the last.
- **Julia's precompilation cache is per CPU target.** Precompiling on a
  cluster's login node does not help its compute nodes if the two are
  different machines -- on Symmetry the login node is Intel and the AMD
  nodes are not, so a batch job precompiles everything again on its own
  wall clock. Precompile in an interactive `srun` on the same node type
  before submitting, and do not submit two jobs that would precompile the
  same depot at once.

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
- Record measured numbers in `CODE.md` when they change, so a regression
  shows up as a changed number and not as a test that merely still passes.

## Repository facts

- `Manifest.toml` is gitignored (both root and `bin/`), as is `TODO.md`. This
  file is *not* — `CLAUDE.md` is committed, so an edit to it lands in the diff
  and belongs in the commit message like any other change.
- **`bin/` has its own Manifest**, so `Pkg.update("TreeAMR")` in the root does
  not touch it. After a TreeAMR change, update both or the viewer fails with a
  `MethodError` on an API the tests are already using.
- `bin/output/` is gitignored; the viewer writes PNGs there.
- **`bin/benchmark.jl` is the one script in `bin/` that runs against the
  root project**, not `bin/Project.toml`: it needs no CairoMakie, and
  asking a compute node to build Cairo in order to time a Laplacian would
  be absurd. It is therefore not part of the viewer CI job.
- `CODE.md` is committed. `README.md` is the short public blurb.
- `main` is committed and published: `origin` is
  `git@github.com:eschnett/TreeWave.jl.git`, and `main` tracks it. Work on a
  branch; do not push, open a pull request, or merge to `main` without being
  asked.
