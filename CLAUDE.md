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

And, because the RHS kernel is `@inbounds`, also run them with the
checks put back -- this is what CI does, and it is the only thing that
can catch the annotation being wrong:

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--check-bounds=yes"])'
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

The full suite takes about 60 seconds, of which the thread-independence
test is 10 -- it runs `test/thread_workload.jl` in a subprocess, which
pays Julia's startup and `OrdinaryDiffEq`'s load time again. The other
slow parts are the type tests, the two convergence sweeps and the 3D
smoke test; the pulse test is about 5 s and the device tests about 8,
nearly all of it compiling the two firing kernels.

## Things that have bitten before

- **TreeAMR comes from the General registry, not from the local
  checkout.** A `~/src/jl/TreeAMR` working copy is *not* what is being
  tested, and neither is upstream `main` any more. If a TreeAMR change is
  needed, say so — the route is a TreeAMR release, not an edit to that
  checkout and not a branch pin here. Note the remote has no `master`
  branch — only `main` and `gh-pages`.
- **There is exactly one TreeAMR version in the repository, and it is
  `[compat]` in `Project.toml`.** It was two `rev = "main"` pins under
  `[sources]`, one here and one in `bin/Project.toml`, and the standing
  hazard was updating the first and leaving the viewer resolving a branch
  that no longer existed. Both are gone: `bin/` declares TreeAMR as an
  ordinary dependency with no bound of its own and inherits this one
  through its `TreeWave = {path = ".."}` source. The bound is a *floor*
  over the `0.1` series — `"0.1.0"` admits every `0.1.x`, so a new
  TreeAMR patch arrives on the next resolve with nothing to edit, and
  the bound is raised only when this package comes to need something a
  newer release added. If you do raise it, raise it here and nowhere
  else; grep for `TreeAMR` to confirm nothing else names a version.
- **Deleting a `[sources]` entry does not un-track the branch —
  `Pkg.resolve()` says "no packages added or removed" and leaves
  `repo-rev = "main"` sitting in the manifest.** Both manifests are
  untracked, so CI resolves from scratch and never saw this; a working
  copy keeps building against the branch until someone runs
  `Pkg.free("TreeAMR")` in the root *and* in `bin/`, which is the quiet
  way to measure a TreeAMR that is not the one CI measures. A freed
  entry says `registries = "General"` and carries no `repo-rev`; check
  for that rather than for the absence of an error.
- **The floor is Julia 1.10 and no longer has anything to do with
  `[sources]`.** It was 1.11 because TreeAMR was unregistered and
  `Project.toml` had to locate it with a `[sources]` entry, which is a
  1.11 key; TreeAMR was registered on 2026-09-21 and 0.1.1 released the
  same day, the entry went, and the floor is now just the LTS. To check a
  change end-to-end the way CI will see it, at both ends of the matrix:
  `git archive HEAD | tar -x -C /tmp/clean && julia --project=/tmp/clean -e 'using Pkg; Pkg.test()'`
  and the same with `julia +1.10`.
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
- **Order-4 prolongation is required** for 2nd-order convergence on a
  refined mesh, even though the RHS stencil reaches only one cell. This is
  TreeAMR's interface-order rule; `CODE.md` explains it. What it costs
  depends on the centering: vertex-centred needs `G = 1` and is indifferent
  to the restriction order (injection along a stagger has no order to
  raise); cell-centred needs `G = 2` *and* order-4 restriction, and refuses
  `G = 1` outright. Never give `ops` or `G` a default that hides it.
- **Vertex centring is the default, and cell-centred is kept as the
  comparison.** Every driver takes `centering`, defaulting to
  `vertexcentered(D)`; the `test/*_cell_tests.jl` files run the same
  studies at `cellcentered(D)` with the numbers that were measured before
  the switch. Do not delete one half to make a change smaller — the
  agreement between them is a measurement, and it needs both halves to
  keep making it.
- `coordinates(fs, b, idx)` — which replaced `cell_center` in M8 — indexes
  the **stored** array, so owned point `i` is `idx = i + fs.G[d]`. It takes
  the *field set* because the answer depends on the ghost width and the
  centering and the forest carries neither. Off-by-`G`, or reconstructing a
  position as `origin + (i - 1/2)h` instead of asking, produces plots that
  look almost right — and on a vertex mesh the second one is wrong by half
  a spacing everywhere.
- **`fs.G` is an `NTuple{D,Int}`, not an `Int`.** Every misuse happens to
  fail loudly (`Int + NTuple` is a `MethodError`), which is the only
  pleasant thing about the migration. `fs.forest.G` no longer exists.
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
- **The RHS kernel's `@inbounds` is load-bearing in both directions.**
  It is 6.5x on `wave_rhs_kernel!` (the checks stop the stencil
  vectorizing, so this is not a constant factor per load), and it is an
  assertion that only `--check-bounds=yes` can falsify -- which is why
  `.github/workflows/CI.yml` *states* `check_bounds: 'yes'` instead of
  inheriting the action's default. Do not drop that line to make a CI
  edit smaller. Run the suite both ways: the checked run never executes
  the optimized code, and the plain run is the only one that can catch a
  wrong answer rather than a wrong index.
- **`@inbounds` does not cross a call that is not
  `@propagate_inbounds`.** It reaches an inlined callee only if that
  callee says so, and an anonymous closure never does, so
  `@inbounds ntuple(v -> work[...], ...)` keeps its checks while
  `ntuple(v -> @inbounds(work[...]), ...)` does not. `cell_tau` in
  `src/refinement.jl` is an `@inline` helper that reads `work` for its
  callers and is *not* marked, and is deliberately left that way -- the
  criterion is regrid-frequency, not per-evaluation. If that ever
  changes, annotate inside `cell_tau`; an `@inbounds` at the call site
  would silently do nothing.
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
- **A callback must capture no `Type` and no host array.** Every
  initial-data, boundary and flagging callback becomes a kernel argument,
  so everything it closes over must be `isbits`. `zero(T)` inside a
  closure whose `T` is a *local* puts a `Type` in a kernel argument;
  write `zero(x[1])`. A captured `Vector` must become a device array
  (`to_backend`) or a tuple -- which is why `cell_tau` takes tuples and
  `blast_exact` uploads its two profiles. Generators are the quiet
  version of the same problem: `prod(… for d in 1:D)` was replaced by an
  accumulating loop, which is the shape TreeAMR's own device-tested
  closures use.
- **A schedule belongs to a *layout*, and a regrid rebuilds it.** Write
  `GhostSchedule(fs, ops)`, which takes the ghost width, centering, element
  type and backend from the field set; the forest form still exists but now
  needs all four spelled out and is the way to get a subtly wrong schedule.
  `regrid!` takes `fs => schedule` pairs for the same reason — a bare field
  set no longer says which schedule moves it — and rejects the M6 spelling
  by name rather than with a `MethodError`.
- **Loading a device package inside `main` is a world-age bug**, and it
  has two symptoms, not one. `allocate` falls through to
  KernelAbstractions' generic method and throws a `MethodError` naming a
  method the same message lists as a candidate -- that one is loud.
  `supports_float64` falls through to the generic `true`, which is
  *silent*: the `--type=f64 --backend=metal` guard passed and the
  objection arrived from the mesh instead. `bin/backend.jl`'s
  `withbackend` puts the check and the work inside one `invokelatest`;
  use it rather than calling `resolvebackend` and carrying on.
- **The mesh sizes the tests use are two orders of magnitude too small
  for a GPU.** A whole `track_blast` is 7× *slower* on Metal at 52k
  cells. That is launch overhead, not a regression: measure phases at
  `--n=128 --roots=32` (29.4M cells), which is what `CODE.md` records.
- **`block_mapreduce` is where the per-block reductions go.**
  `field_scales`, `blast_coverage` and `track_pulse`'s tracking measure
  all reduce field data one block at a time, and all three go through
  TreeAMR rather than looping here — the thread-count determinism is
  upstream's invariant and a second copy of that argument is a second
  thing to get wrong. It was the unexported `TreeAMR.block_partials`
  until upstream exported it; if you find prose here still saying that,
  it is stale.
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
- **`bin/Project.toml` still needs Julia 1.11**, because its
  `TreeWave = {path = ".."}` source is a real dependence on `[sources]`
  and is the one that cannot go away; only the root project dropped to
  1.10. The viewer CI job runs on `"1"`, so this is invisible until
  someone tries the viewer on the LTS.
- **`bin/` has its own Manifest**, so `Pkg.update("TreeAMR")` in the root does
  not touch it. After a TreeAMR change, update both or the viewer fails with a
  `MethodError` on an API the tests are already using.
- **`bin/backend.jl` is `include`d by all three scripts in `bin/`**, two of
  which run against `bin/Project.toml` and one against the root project. So
  it may use only what *both* environments have -- today that is
  `KernelAbstractions`, which is why it was added to `bin/Project.toml`.
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
