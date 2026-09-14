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

The full suite takes about 20 seconds. The slow parts are the two
convergence sweeps and the 3D smoke test; the pulse test is about 5 s.

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
- `CODE.md` is committed. `README.md` is the short public blurb.
- `main` is committed and published: `origin` is
  `git@github.com:eschnett/TreeWave.jl.git`, and `main` tracks it. Work on a
  branch; do not push, open a pull request, or merge to `main` without being
  asked.
