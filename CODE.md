# TreeWave.jl — Design

TreeWave.jl solves the scalar wave equation in 2nd-order form on an
adaptively refined mesh. It exists to be the worked example for
[TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl): TreeAMR provides the
mesh, the storage, and the inter-grid operations and deliberately contains
no physics, so its own wave-equation code lives in its test suite and
cannot be pointed at as a downstream application. This package is that
application.

## Goals

- Show the whole path from mesh to solution: initial data, right-hand
  side, time integration, regridding, error measurement, visualization.
- Use only TreeAMR's public API. Nothing here reaches into TreeAMR's
  internals, and nothing here is mesh machinery that belongs upstream.
- Be small enough to read in one sitting.

## Scope and non-goals

- **No mesh machinery.** If something is about trees, ghosts, or
  interpolation, it belongs in TreeAMR.
- **One equation, two initial conditions.** Not a framework; there is no
  abstraction over equations or initial data, because with two of each an
  abstraction would only hide what the example is meant to show.
- **Non-conservative at coarse-fine interfaces.** The wave equation does
  not need flux matching, and TreeAMR defers it to its M8 anyway.

## The equation

    ∂ₜu = v
    ∂ₜv = ∇²u                    (wave speed c = 1)

Variable 1 is `u`, variable 2 is `v`. The Laplacian is the standard
2nd-order centered 3-point stencil per dimension, evaluated by a single
KernelAbstractions kernel over all blocks
([`src/evolution.jl`](src/evolution.jl)).

Second-order form rather than first-order is deliberate: it makes the
initial data trivially exact for both test problems, and it puts a
*second* derivative in the RHS, which is what makes the interface-order
rule below bite. A first-order system would hide it.

## The right-hand-side contract

Each evaluation of `wave_rhs!` does exactly three things, in order:

    scatter!(p.fs, u)          # flat state vector -> working-array interiors
    fill_ghosts!(p.fs, p.schedule)
    map_blocks!(wave_rhs_kernel!, ...)   # writes du in state layout

This is the pattern TreeAMR specifies, with no `semidiscretize`-style
wrapper. Two properties of it matter and are easy to break:

- **`u` is authoritative and is never mutated.** The working array is
  scratch, refreshed at every evaluation. A RHS that wrote back into `u`
  would corrupt a multi-stage method like RK4, and would do so silently.
- **The RHS is a pure function of `(u, t)`.** No state is carried in the
  working array between calls, so re-evaluating at the same `u` gives the
  same `du`.

`WaveProblem` is the container the integrator carries. It exists to hoist
everything invariant out of the per-evaluation path: the field set, the
ghost schedule, the per-block spacings, and — as `Val{D}` and `Val{G}` —
the two parameters the kernel specializes on. Building the `Val`s per
evaluation instead would recompile or dynamically dispatch the kernel on
every RK stage. `G` is read from a runtime field, so the *constructor* is
type-unstable by design; it is called once per chunk, never per step.

## Initial conditions

The two are not variations on a theme. Each measures something the other
cannot.

### Standing sine mode ([`src/sinewave.jl`](src/sinewave.jl))

    u(x,t) = cos(ωt) ∏_d sin(2πm x_d / L),   ω = 2πm √D / L

An exact solution of the continuum equation on a periodic box in any
number of dimensions. Because the answer is known at every time and
everywhere, this is what pins down the **order** of the discretization:
run it at several `N` on a hierarchy held fixed in physical space, and the
slope of the error against `h` is a number that either is 2 or is not.

`wave_forest` builds that hierarchy — a `roots^D` periodic box with the
middle sub-box refined once — and takes `refined=false` to produce the
same box uniform, as a control. Holding the refined *region* fixed while
`N` varies is what makes the study measure `h` and nothing else.

### Travelling super-Gaussian pulse ([`src/supergaussian.jl`](src/supergaussian.jl))

    u = G(d),  ∂ₜu = -G'(d),   d = x - x₀ - t  (wrapped),
    G(x) = exp(-(x/σ)^(2n))

Exact for the 1D equation, and in higher dimensions a plane pulse uniform
in the transverse directions, so `∇²u = ∂ₓ²u` and it stays exact. Because
the feature is *localized* and *moves*, this is what exercises a refined
region that has to follow it — something a standing mode cannot test at
all.

The minus sign on `∂ₜu` is load-bearing: for `u = G(x - t)` the chain rule
gives `∂ₜu = -G'`, and initial data with the wrong sign matches neither
characteristic, so the pulse splits rather than travelling. (This was a
real bug in the first draft of this package, introduced when TreeAMR's
closed-form Gaussian derivative was replaced by a call to
`dsupergaussian` and the sign that had been folded into the algebra was
lost. The measured L∞ error at the pulse test's finest mesh was 110
before the fix and 43 after, against 0.09 once the resolution question
below was also settled — the sign fix alone was not enough to make the
error small, which is why both had to be found.)

**The order `n` is a keyword, defaulting to 1** (the ordinary Gaussian),
because raising it trades localization against resolvability and the mesh
pays for both. Measured shoulder width — where `G` falls from 0.9 to 0.1 —
at `σ = 0.08`, in cells of the finest mesh the pulse tests use
(`h = 1/256`):

| n | shoulder | cells | max abs G' |
|---|---|---|---|
| 1 | 0.0954 | 24.4 | 10.7 |
| 2 | 0.0530 | 13.6 | 19.0 |
| 4 | 0.0284 | 7.3 | 37.1 |
| 8 | 0.0148 | 3.8 | 73.7 |

Two things worsen together as `n` rises: the shoulder needs more cells,
and `∂ₜu = -G'` grows like `n/σ`, so the state itself is larger and the
error scales with it. At `n = 8` the shoulder is under four cells wide on
that mesh and the pulse is simply not represented — the L∞ error stays
O(40) no matter how correct the code is. Raising `n` means raising the
resolution of every pulse run to match; the flat-top shape is available,
but it is not free.

## Operator order: the constraint inherited from TreeAMR

TreeAMR's interface-order rule is why every driver here takes `ops` and
`G` together rather than defaulting them. A ghost filled by an order-`p`
operator carries an `O(hᵖ)` error; the 2nd-order Laplacian divides it by
`h²`, leaving `O(h^(p-2))` along every coarse-fine interface. With `p = 2`
that is `O(1)` and drags the global rate to **first** order even though
the interior scheme is second order.

So 2nd-order global convergence on a refined mesh needs **order-4
prolongation and restriction, and `G = 2`** — even though the RHS stencil
itself only ever reaches one cell. Raising one operator alone does not
help, because the two sides of an interface get their ghosts from
different operators. `test/sinewave_tests.jl` asserts all four
combinations, so the rule is guarded here and not merely documented.

## The refinement criterion

Refinement is for **resolution**, not amplitude. The first version of this
package refined where `|u|` was large, which worked only because the pulse sat on
a zero background — on the sine mode, whose amplitude is O(1) everywhere, it would
have refined the whole domain. It also fixed the depth by fiat (`maxlevel_wanted=2`,
so every block was at level 2 or level 0) and read only `u`, missing features
carried by `∂ₜu`, which has an order of magnitude more amplitude.

[`src/refinement.jl`](src/refinement.jl) replaces it with a per-cell
[Löhner](https://doi.org/10.1016/0961-3552(91)90006-T)-style indicator,

    τ = |uᵢ₊₁ - 2uᵢ + uᵢ₋₁| / (|uᵢ₊₁ - uᵢ| + |uᵢ - uᵢ₋₁| + 4ε·scale)

taken over every interior cell, dimension, and variable. The differences are
undivided, so the spacing enters implicitly: τ measures how well the *mesh*
resolves what it holds, not the data's curvature. For smooth data τ therefore
*falls* as `h` shrinks, and that is what makes refinement terminate on its own.
τ ∈ [0, 1], since the numerator is bounded by the two first differences.

### The noise floor must be global

Löhner's classic form floors the denominator with the local
`ε(|uᵢ₊₁| + 2|uᵢ| + |uᵢ₋₁|)`. That is scale-free, and scale-free is fatal here: in
the pulse's far tail three consecutive values of `3.9e-16, 3.6e-17, 3.6e-17` — ten
orders of magnitude below the peak — score `τ = 0.986`, because the floor shrinks
along with them. Measured consequence: the criterion refined the entire domain at
every threshold tried, 32 blocks where 16 were wanted. Referring the floor to a
global per-variable amplitude ([`field_scales`](src/refinement.jl)) fixes it —
negligible regions then get a negligible numerator against a fixed floor. After
the fix the pulse's tail blocks score τ = 0.0000 against the feature's 0.79.

Since the wave equation conserves amplitude, the scale is measured once from the
initial data and reused; a problem that grows or decays by orders of magnitude
would have to refresh it.

### Two thresholds, meaning two different things

- `refine_tol` is *"under-resolved here"* — go finer.
- `coarsen_tol` is *"there is something here at all"* — the feature is present,
  even if adequately resolved.

Which gives the four marks in [`refine_mark`](src/refinement.jl): `(Refine, box)`
below the cap, `(Keep, box)` at it, a bare `Coarsen` where nothing fired, and a
bare `Keep` otherwise. The gap between the thresholds is also the hysteresis dead
band, so a block near one threshold cannot flip on alternate regrids.

The box is the bounding box of cells above **`coarsen_tol`**, not `refine_tol`, and
that choice is load-bearing. A block refined to the cap has by construction stopped
being under-resolved — its τ fell below `refine_tol`, which is precisely why
refinement stopped there — so keying the box on `refine_tol` would make a
feature-holding block at the cap report nothing, and the `(Keep, box)` margin below
would be unreachable in the one case it exists for.

τ vanishes wherever the second difference does, so a Gaussian's inflection points
score zero even though the feature is right there. The reduction to a block verdict
is `max` over cells rather than a vote, and the box is convex, so both close that
notch by construction.

### Calibrated thresholds

Löhner's canonical `τ > 0.8` is a shock detector; smooth data never comes close, so
the thresholds had to be measured. Max τ over the pulse (σ = 0.08) on uniform
meshes at `roots = 8`, against the depth the criterion then reaches when the cap is
set to 6 so that the *indicator* has to be what stops it:

| level | h | measured max τ |
|---|---|---|
| 0 | 1/64 | 0.794 |
| 1 | 1/128 | 0.478 |
| 2 | 1/256 | 0.192 |

| refine_tol | depth reached (cap 6) | blocks | cells |
|---|---|---|---|
| 0.15 | 3 | 20 | 160 |
| 0.20 | 2 | 16 | 128 |
| 0.30 | 2 | 16 | 128 |
| 0.45 | 2 | 16 | 128 |

Any `refine_tol` in `[0.20, 0.45]` terminates at level 2, so the defaults are
`refine_tol = 0.30`, `coarsen_tol = 0.075` — mid-plateau, with the cap never
binding. Note that an earlier closed-form estimate for τ *at the pulse peak*,
`(h²/σ²)/(h²/σ² + 4ε)`, gave 0.49 / 0.19 / 0.056: the right shape but about 3×
low, because the maximum over a block is not at the peak. The measured column is
the one to trust.

### Buffering: TreeAMR's job, the width ours

TreeAMR dilates the reported boxes by a `buffer` given in cells. The width is the
application's to choose, because it is physics — feature speed times regrid
cadence. [`refinement_buffer`](src/refinement.jl) derives it: the wave speed is 1,
so the pulse travels `chunk` per regrid, and

    buffer = ceil(chunk / spacing(forest, maxlevel_cap)) + 1

The `+ 1` is because TreeAMR's guidance is that the margin must *exceed* the motion
it covers. It uses the spacing at the cap rather than `minimum_spacing`, which
reports the *current* finest spacing — coarse while the hierarchy is still being
built, and so would derive a uselessly narrow margin on the first pass. Since
recruitment reaches exactly one ring of neighbours, `buffer ≤ N`; that cap is
really a statement about cadence — the feature may not cross a whole finest-level
block between regrids — and the derivation throws naming that rather than letting
the caller meet an opaque rejection inside `regrid!`.

## Regridding: restart per chunk

Regridding changes both the length and the meaning of the state vector.
Rather than fight that inside a DiffEq callback — where multistep history
and dense output become invalid the moment entries are reinterpreted —
`track_pulse` stops at the end of each `chunk`, regrids, rebuilds the
schedule and the state vector, and starts a fresh `solve`. This is the
pattern TreeAMR prescribes for anything beyond a one-step method, and at
`chunk = 0.02` the restart cost is not measurable against the step cost.

The refinement criterion is deliberately crude — refine a block if its
peak `|u|` exceeds a threshold — because the point is to test whether the
*mesh* follows a moving feature, not to design a good error estimator.

## Watching a run: the `observer` keyword

`wave_errors` and `track_pulse` return summary numbers, which is what the
tests want. The viewer wants the run itself, so both take an
`observer=nothing` keyword, called as `observer(fs, t, u)` with `fs`
already scattered from `u`: per chunk for `track_pulse` (before the regrid
that would invalidate `fs`), and at `nsnapshots` times via `saveat` for
`wave_errors`. Left `nothing`, neither stores an intermediate solution.

This is the alternative to the viewer re-implementing a driver. There are
already three near-identical time-stepping loops in `src/`; a fourth in
`bin/` would be the one that silently drifts out of step with the rest.

## File layout

| file | contents |
|---|---|
| `src/TreeWave.jl` | module shell: `using`s, exports, includes |
| `src/evolution.jl` | the RHS kernel, `WaveProblem`, `wave_rhs!`, `convergence_rate` |
| `src/refinement.jl` | the per-cell refinement indicator and its reduction to block marks |
| `src/sinewave.jl` | the standing mode and its convergence driver |
| `src/supergaussian.jl` | the travelling pulse, its AMR driver, and the uniform reference |
| `test/sinewave_tests.jl` | convergence order, the interface-order rule, 3D smoke test, energy drift |
| `test/refinement_tests.jl` | claims about the indicator itself |
| `test/supergaussian_tests.jl` | a moving refined region tracks the pulse |
| `bin/visualize.jl` | CairoMakie viewer (own environment; see `bin/Project.toml`) |

The viewer draws three panels per case — the solution, the pointwise error
in `u`, and the volume-weighted L2/L∞ norms over the whole state against
time — with one line per block colored by refinement level. `--ops=2`
reruns with order-2 operators, which is the quickest way to see the
interface-order rule rather than read about it. Figures are written as PNGs
and, when stdout is a terminal, drawn inline via SixelTerm.

The tests are ported from TreeAMR's own `test/wave_tests.jl`, minus its
"RHS does not mutate the state vector" testset — that one guards TreeAMR's
`scatter!`/`gather!` contract rather than anything about the wave
equation, and belongs upstream where it already lives.

Visualization lives in `bin/` and not in the package because CairoMakie is
a heavy dependency that nothing in `src/` needs. `bin/` carries its own
`Project.toml` with a `[sources]` entry pointing at the package root, so
`julia --project=bin bin/visualize.jl` works from a fresh checkout.

## Measured results

Recorded so that a regression is visible as a change in a number rather
than as a test that merely still passes.

- Sine mode, uniform mesh, order-2 operators: L2 rate 2.0 (D = 1, 2).
- Sine mode, two-level mesh, order-4 operators and `G = 2`: L2 and L∞
  rates both 2.0 (D = 1, 2).
- Sine mode, two-level mesh, D = 1: L2 rate 1.0 at (2,2), below 1.5 at
  (4,2) and at (2,4), and 2.0 at (4,4) — the interface-order rule.
- Sine mode, four periods on a two-level mesh: L∞ amplitude within 5% of
  its initial value.
- Pulse, `n = 1`, `σ = 0.08`, `roots = 8`: uniform-coarse (`N = 8`) L∞
  error 1.433, uniform-fine (`N = 32`) 0.0928. The adaptive run (`N = 8`,
  two levels) gives 0.0961 — a ratio to the fine reference of 1.035 — at
  **128** cells against the fine mesh's 256. The pulse peak never leaves a
  refined block over the whole run.
- Old amplitude criterion vs new Löhner criterion on that same run: 0.0926
  at 176 cells against 0.0961 at 128 cells. The resolution criterion buys a
  27% cell saving for a 4% error increase, and unlike the old one it is not
  told the depth — it discovers level 2 and stops there.
- Buffer width, same run (derived width is 7 cells for `chunk = 0.02`):

  | buffer | worst L∞ | ratio to uniform-fine | cells |
  |---|---|---|---|
  | 7 (derived) | 0.0961 | 1.035 | 128 |
  | 2 (too narrow) | 0.1071 | 1.154 | 128 |
  | 0 (none) | 0.1454 | 1.566 | 112 |

  Wider is monotonically better here, and only the derived width meets the
  test's `rtol = 0.1` against the uniform-fine reference. This **does not**
  reproduce TreeAMR's recorded observation that a margin narrower than the
  motion measures *slightly worse than no buffer at all*: at `buffer = 2`
  the error is clearly better than at `buffer = 0`, not worse. Recorded as a
  contradiction rather than smoothed over — it may be geometry-specific, and
  TreeAMR measured it on a different setup.
- Sine mode at 0.9 periods, `N = 16`, `roots = 4`, two levels: final L∞
  0.0063 with order-4 operators against 0.122 with order-2 — a factor of
  19 for a change that touches only the ghost cells. This is the pair the
  viewer draws side by side (`--ops=2`), and the pointwise error goes from
  smooth across the coarse-fine interfaces to visibly kinked at them.

## Possible extensions

Not planned, listed because they are the obvious next questions:

- Kreiss–Oliger dissipation, to see what it does to interface modes.
- An adaptive integrator, once TreeAMR's `volume_weighted_norm` is wired
  in as `internalnorm`.
- First-order form, as a second application of the same mesh.
