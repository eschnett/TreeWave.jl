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
- Run in the caller's floating-point type, not only `Float64` — TreeAMR's
  mesh is generic and an application that is not would give the genericity
  nowhere to go. See Precision.

## Scope and non-goals

- **No mesh machinery.** If something is about trees, ghosts, or
  interpolation, it belongs in TreeAMR.
- **One equation, three initial conditions.** Not a framework; there is no
  abstraction over equations or initial data, because with three of the
  latter an abstraction would only hide what the example is meant to show.
  The third one earns its place by breaking something: it is the only case
  whose feature *loses amplitude*, and that is what puts the refinement
  criterion's noise floor under strain rather than merely exercising it.
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

The three are not variations on a theme. Each measures something the
others cannot: the sine mode has an exact answer everywhere and so fixes
the *order*; the pulse is localized and *moves*, so the refined region has
to follow it; the blast wave *spreads*, so its amplitude falls and its
refined region grows.

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

### Radial blast wave ([`src/blast.jl`](src/blast.jl))

    u = G(r),  ∂ₜu = 0,   r = |x - x₀|  (wrapped)

A super-Gaussian peak released at rest in the middle of the box, which
spreads as a ring at the wave speed. This is the Sedov blast-wave test in
the only form this package can run: there is no shock and no Riemann
solver, it is still the wave equation, but the mesh problem is the one a
Sedov test poses — a front expanding at a known speed, *losing amplitude
to geometric spreading*, with the refined region tracking it outward and
the interior coarsening back behind it. It is the first case here in which
coarsening does any work at all.

Released at rest is forced, not chosen. A purely outgoing radial wave is
`P(r-t)/r`, singular at the origin unless `P(0) = 0`, so a peak *at the
centre* cannot be outgoing: it splits, the ingoing half passes through the
origin at once, and a single expanding ring is what survives.

#### Why 2D, when 3D is where the closed form lives

In three dimensions Huygens' principle holds, the ring is a clean shell
with exactly nothing behind it, and the radial solution is elementary:

    u(r,t) = [h(r+t) + h(r-t)] / 2r,   h(s) = s·G(s)

Measured against the solver at `σ = 0.07`, that gives L∞ `1.43 / 0.76 /
0.33` at `h = 1/24, 1/32, 1/48` — second order, so the formula is right.

In two dimensions there is no Huygens principle. A wake trails the ring
and never clears — measured `u(r = 0.05, t = 0.3) = -0.0152` in 2D against
`0.0` to roundoff in 3D — and there is no elementary closed form.

3D is nonetheless the wrong choice, on cost. An adaptive *shell* is
minutes, not seconds: `track_pulse(Val(3))` already costs 20 s for
`t ≤ 0.06` at 2752 blocks, and a shell whose area grows is far worse. That
is outside what the CI figure job can render, and a 3D viewer would have
to slice to a plane anyway. The 2D adaptive run costs 1.4 s to `t = 0.4`.
So the 3D formula is recorded here and not implemented.

#### The exact solution is a quadrature

In 2D the radially symmetric solution is the inverse Hankel transform of
the initial profile, propagated mode by mode:

    u(r,t)  =  ∫₀^∞ f̂(k) J₀(kr) cos(kt) k dk
    ∂ₜu(r,t) = -∫₀^∞ f̂(k) J₀(kr) sin(kt) k² dk

For the ordinary Gaussian `f̂(k) = (σ²/2)·exp(-k²σ²/4)`, which is why an
exact solution exists at all — and why it exists **only at `n = 1`**.
[`blast_exact`](src/blast.jl) throws for any other order rather than
returning a silently wrong answer; the case itself still runs at higher
`n`, it just has to be measured against a uniform reference mesh instead.

Two things make the radial table finite and correct. It is **zero beyond
`t + 6σ`** — the solution has no support ahead of the front and a
Gaussian's tail at six standard deviations is `e^(-36)`. And the closure
**sums over periodic images**, which `pulse_exact` never has to do: a
plane pulse of width `σ ≪ L` is always far from its own images, but a ring
reaches the corners of the box, where the nearest image lies exactly as
far away as the source. The number of image rings is derived from
`t + 6σ`, so raising `t_end` cannot quietly invalidate it.

`J₀(kr)` does not depend on `t`, so [`blast_reference`](src/blast.jl)
tabulates it once on the `(r, k)` grid and every later time is a
contraction against it. That is not a micro-optimization: evaluating the
transform afresh costs 0.65 s, and a run measuring its error once per
chunk wants twenty of them, so caching turns 13 s into 1 at a cost of
32 MB.

Against a uniform mesh the whole construction reproduces second order —
L2 rate **1.99**, L∞ rate **1.95**, tabulated under Measured results. The
quadrature's own error is ~9e-7, four orders below the finest
discretization error, so what the comparison measures is the scheme and
not the table.

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

## Precision

TreeAMR's mesh is generic in its floating-point type: `Forest{D,T}` carries
the type the *geometry is computed in*, not merely stored in, and `FieldSet`
and `GhostSchedule` default their element type to the forest's. This package
follows it, so that a run can be done at `Float32` — the point being test runs
on a device with no hardware fp64, which is exactly what a low-end GPU is —
or at a MultiFloats type, which is a software float and therefore evidence
that no fp64 path is load-bearing anywhere.

Every driver takes the type as a **leading positional argument**, spelled as
TreeAMR spells `spacing(T, forest, level)`:

    wave_errors(Float32, Val(1); N=16, G=2, ops=ops)
    track_pulse(Float32, Val(1); roots=8, N=8)
    track_blast(Float32, Val(2))

It defaults to `Float64`, so every existing call site — the tests, the
viewers, the numbers below — is unchanged. It is said once, to the `Forest`,
and reaches the field set, the schedule, the state vector and the coordinates
handed to every initial-data callback from there.

### The rule

**Physical quantities carry `T`; counts and ratios of counts do not.** Errors,
norms, spacings, times, amplitudes and τ are `T`. `nblocks`, `nsteps`,
`maxlevel`, `growth` (a ratio of leaf counts) and `blast_coverage` (a fraction
of cells) stay `Int` or `Float64`. `track_pulse`'s `tracking` is a ratio of
*amplitudes*, so it is `T`.

**No floating-point literal may appear where `T` is in play.** `0.01` is an
fp64 operand and drags the expression with it; `T(1//100)` is exact and folds
at compile time. Every keyword default is written that way — `cfl=T(1//4)`,
`σ=T(2//25)`, `coarsen_tol=T(3//40)` — and at `Float64` each is bit-identical
to the decimal it replaced, which is why no measured number below moved.

Two literals were not merely a type leak but a bug waiting for a change of
precision:

- `ε=0.01`, the Löhner noise floor, multiplied straight into the denominator,
  so **every** τ came out `Float64` however the field was stored.
- `while t < t_end - 1e-12`, the chunk-loop guard. At `Float32` `1e-12` is far
  below one ulp of `t`, so the guard degenerates into `t < t_end` and whether
  the final chunk runs turns on rounding. Both loops now count chunks, which
  is exact in every type.

### Base does not come along

The mesh is generic; `Base` is not. MultiFloats.jl implements `floor`, `ceil`,
`sqrt`, `exp`, `log` and integer powers, but **no `rem`** — so `mod` is a
`MethodError` — and **no conversion to `Integer`**, nor to `Float64` (only to
its own limb type, so `Float32(::Float32x2)` works and `Float64(::Float32x2)`
does not). Every one of those sits on a path a driver takes. `src/precision.jl`
bridges them, and is the whole of the workaround:

| spelling | why not the obvious one |
|---|---|
| `wrap(x, L)` | `mod` goes through `rem`. `x - L·floor(x/L)` is the same function for `L > 0`, and at `L = 1` — every run here — bit-identical. |
| `ceilint`, `floorint` | `ceil(Int, x)` closes through a conversion to `Integer`. The fallback goes via `BigFloat`, the one conversion every `AbstractFloat` has, and only once the value is already an exact integer. |
| `tofloat64` | the bridge to the blast wave's reference table. Same story. |

The `BigFloat` fallbacks allocate, which is why they are confined to what they
convert: loop counts, evaluated a handful of times per run, and one subscript
into a `Float64` lookup table. A hardware float never reaches them.

### What each type is for, and what is not available

| type | what it catches |
|---|---|
| `Float64` | the default; where every convergence order is measured |
| `Float32` | the **leak detector** — a stray `Float64` operand widens the result, so a returned `Float64` names the leak |
| `Float32x2` | the **software-float** probe: two `Float32` limbs, which no fp64 fast path can serve. It cannot detect leaks (MultiFloats promotes `Float64` *downward*, absorbing them silently); it tests that nothing depends on a hardware float at all. `Float64x2` behaves the same way and is what a run wanting more precision than `Float64` would use. |

Two things are **not** available at a MultiFloat, both recorded rather than
worked around:

- **The sine mode**, because it is built on `sin` and `cos`, which MultiFloats
  does not implement: they `error`, pointing at
  `MultiFloats.use_bigfloat_transcendentals()`, which evals BigFloat-backed
  methods into `Base`. Neither this package nor TreeAMR calls that; a caller
  who wants it may.
- **A reference more accurate than `Float64`.** The blast wave's exact ring is
  a Hankel quadrature over `besselj0`, which SpecialFunctions defines for
  hardware floats only. [`blast_reference`](src/blast.jl) therefore keeps its
  table in `Float64` on the host at every precision, and
  [`blast_exact`](src/blast.jl) converts it into `T` **once, outside the
  per-cell closure** — so the closure a field set is filled from is arithmetic
  in `T` alone and no fp64 array would ever be uploaded to a device. The
  quadrature's own ~9e-7 error is four orders below the finest discretization
  error measured against it, so nothing this package can resolve notices the
  cap.

### What `Float32` cannot do

Roundoff in the Laplacian is `ε/h²`, because the second difference is divided
by `h²`. At `Float32` and `h = 1/256` that is ~8e-3 — *comparable to the
discretization error the finest mesh of the convergence sweeps measures*. So
**the measured second-order rates cannot be reproduced at `Float32`**, and
`test/type_tests.jl` does not assert them: moving the meshes until a rate came
out right would be fitting the test to the answer. What it asserts instead is
what survives a change of precision — the types, boundedness, and the mesh the
criterion chooses.

That last one is the claim worth having, and it holds. Measured, at the
calibrated tolerances:

| run | `Float64` | `Float32` |
|---|---|---|
| pulse, `t_end = 0.5` | L∞ 0.09609, 16 blocks, level 2 | L∞ 0.09687, 16 blocks, level 2 |
| blast, `t_end = 0.4` | L∞ 0.0297, 136 → 820 blocks, 91% covered | L∞ 0.0298, 136 → 820 blocks, 91% covered |
| sine, `N = 8`, two levels | L2 0.0033392, L∞ 0.0074446 | L2 0.0033391, L∞ 0.0074439 |

The blast wave agrees on the block count, the depth *and* the coverage
fraction — the refinement criterion's decisions are precision-insensitive even
where the error is not, which is what makes a `Float32` run a rehearsal for
the `Float64` one rather than a different experiment.

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

#### The blast wave is that problem, and it fails two ways

The radial blast wave is the case that note anticipated, so the hypothetical is
now a measurement. Geometric spreading takes the peak from 1.0 to 0.109 over the
run. Measured in 2D at `σ = 0.05`, `roots = 8`, `N = 8`, cap 2:

| `field_scales` policy | max τ at t=0.4 | blocks at t=0.4 | outcome |
|---|---|---|---|
| refreshed at every regrid | 0.429 | 784 | tracks the ring; refined area grows with it |
| frozen after one chunk | 0.208 | 472 | τ halves as the amplitude decays 6.5×, falls below `refine_tol`, and the ring stops recruiting blocks ahead of itself |
| frozen at `t = 0`, as `track_pulse` does | 1.000 | 1024 = **the whole domain at level 2** | total failure |

The last row is a **second, sharper trap and not the decay effect**. This initial
data has `∂ₜu ≡ 0`, so `field_scales` returns `[0.952, 0.0]`: the variable-2 scale
is *exactly* zero. At `t = 0` that is harmless by accident — the numerator
vanishes with it and `lohner` returns 0, which is why the initial adaptation is
sane at 136 blocks — but one chunk later `∂ₜu` is numerical dust in the far field
measured against a floor of zero, which is precisely the scale-free pathology
above. τ pins to 1.0 everywhere.

Both are fixed by the same line: [`track_blast`](src/blast.jl) recomputes
`field_scales` at every regrid. It keeps `refresh_scales=false` so that the table
is a test (`test/blast_tests.jl`) and a figure (`--frozen-scales`) rather than a
remark. Note that τ is a *ratio*, so the indicator is otherwise amplitude-blind
by construction — which is exactly why a resolution criterion survives geometric
spreading at all, where the old `|u| > threshold` criterion could not have.

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

Written out, the constraint is `chunk ≤ (N-1)·spacing(forest, maxlevel_cap)`, and
it tightens by a factor of two with every level. At `N = 8` and `chunk = 0.02` a
cap of 2 needs 7 cells and fits; a cap of 3 would need 12 and a cap of 4 would
need 14, so a deeper hierarchy is not a free choice — it has to be paid for with
a shorter `chunk`. Both of those are throws today, not surprises.

## Regridding: restart per chunk

Regridding changes both the length and the meaning of the state vector.
Rather than fight that inside a DiffEq callback — where multistep history
and dense output become invalid the moment entries are reinterpreted —
`track_pulse` stops at the end of each `chunk`, regrids, rebuilds the
schedule and the state vector, and starts a fresh `solve`. This is the
pattern TreeAMR prescribes for anything beyond a one-step method, and at
`chunk = 0.02` the restart cost is not measurable against the step cost.

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
| `src/precision.jl` | the `Base` operations a software float type does not provide, bridged — see Precision |
| `src/evolution.jl` | the RHS kernel, `WaveProblem`, `wave_rhs!`, `convergence_rate` |
| `src/refinement.jl` | the per-cell refinement indicator and its reduction to block marks |
| `src/sinewave.jl` | the standing mode and its convergence driver |
| `src/supergaussian.jl` | the travelling pulse, its AMR driver, and the uniform reference |
| `src/blast.jl` | the radial blast wave, its Hankel-quadrature exact solution, its AMR driver, and the uniform reference |
| `test/sinewave_tests.jl` | convergence order, the interface-order rule, 3D smoke test, energy drift |
| `test/refinement_tests.jl` | claims about the indicator itself |
| `test/supergaussian_tests.jl` | a moving refined region tracks the pulse |
| `test/blast_tests.jl` | the exact ring, 2nd-order convergence to it, a growing refined region, and what a frozen amplitude scale costs |
| `test/type_tests.jl` | the drivers at `Float32` and at MultiFloats' `Float32x2` — see Precision |
| `bin/visualize.jl` | CairoMakie viewer for the 1D cases (own environment; see `bin/Project.toml`) |
| `bin/visualize2d.jl` | CairoMakie viewer for the blast wave — a different figure, not a flag on the other one |
| `.github/workflows/CI.yml` | tests on a Julia matrix, plus a job that renders the figures |

`bin/visualize.jl` draws four panels per 1D case — the solution, the
pointwise error in `u`, the indicator τ, and the volume-weighted L2/L∞
norms over the whole state against time — with one line per block colored
by refinement level. `--ops=2` reruns with order-2 operators, which is the
quickest way to see the interface-order rule rather than read about it.
`--type=f32` reruns in single precision, which is the quickest way to see
that it is the *same run* — same blocks, same levels, same τ — rather than
read that either. Both viewers take it; only the two hardware types are
offered, because Makie cannot plot a MultiFloat.

`bin/visualize2d.jl` is a separate script and not a `--dim=2` flag,
because nothing transfers: a line per block against `x` is not a worse
picture in 2D, it is no picture. It draws a filmstrip of `u` at four
times with the block boundaries colored by level, then `u` against radius
for every cell, then the block count and max τ against time. The radial
panel is the standard Sedov diagnostic, and sharper than it looks: the
exact solution depends on `r` alone, so every cell of a perfect solution
lands on one curve and vertical spread is the error the Cartesian mesh
introduces by not being radial. It is readable only inside `r = L/2`,
where the ring is far enough from its own periodic images that the *exact*
solution is radial too; the figure draws that rule. `--frozen-scales`
renders the failure mode from the refinement section above, and that pair
of figures is the argument for refreshing the scale.

Figures are written as PNGs and, when stdout is a terminal, drawn inline
via SixelTerm. PNG is not an idle default for the 2D viewer: CairoMakie's
fast image path is what lets 820 abutting per-block heatmaps meet without
hairline seams, and vector output disables it, degrading each block into
one polygon per cell.

The tests are ported from TreeAMR's own `test/wave_tests.jl`, minus its
"RHS does not mutate the state vector" testset — that one guards TreeAMR's
`scatter!`/`gather!` contract rather than anything about the wave
equation, and belongs upstream where it already lives.

CI runs the test suite on Julia 1.11 and release, on Linux and macOS, and
separately renders all three figures — plus the pulse at `Float32`, whose
conversions the default render does not exercise — and keeps them as
artifacts. The second job
exists because `bin/` carries its own environment and therefore its own copy
of the TreeAMR dependency: during development that let the viewer keep
building against an older TreeAMR than the tests, until it failed on an API
the tests were already using. Nothing in the test job could have caught that.

Julia 1.11 is the floor, and not by preference: TreeAMR is unregistered, so
`Project.toml` locates it with a `[sources]` entry, which 1.11 introduced.
Without it a clean checkout cannot resolve at all — `Manifest.toml` is not
tracked, and Project.toml alone carries only a UUID — which is also why CI
was impossible before this.

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
- Blast wave, `σ = 0.08`, `roots = 8`, `t_end = 0.4`. Uniform meshes against
  the Hankel-quadrature exact solution:

  | N | h | L2 | L∞ | cells |
  |---|---|---|---|---|
  | 8 | 1/64 | 0.0772 | 0.3082 | 4096 |
  | 16 | 1/128 | 0.0196 | 0.0811 | 16384 |
  | 32 | 1/256 | 0.0049 | 0.0205 | 65536 |

  L2 rate 1.99, L∞ rate 1.95 — the quadrature is right to well past what the
  scheme can see.
- Blast wave, adaptive (`N = 8`, cap 2) against those: L∞ 0.0297 at 52480
  cells, so **10× better than uniform-coarse and 1.45× worse than
  uniform-fine at 80% of its cells**. That is a weaker claim than the
  pulse's `rtol = 0.1` match, and it is the honest one: 9% of the ring's
  cells sit on level-1 blocks whose τ fell below `refine_tol`, which is the
  criterion trading accuracy for cells rather than failing to. The pulse
  matched uniform-fine only because its refined region was, relatively, far
  more generous.
- Blast wave, mesh growth: 136 blocks after the initial adaptation to 820 at
  `t = 0.4`, a factor of 6.0, with the depth an output — at `σ = 0.08` the
  indicator's τ falls to 0.222 at level 2, below `refine_tol = 0.30`, so it
  stops there and `maxlevel_cap` never binds. Max τ on uniform meshes, the
  calibration that fixes σ:

  | σ | h=1/64 | h=1/128 | h=1/256 | h=1/512 |
  |---|---|---|---|---|
  | 0.05 | 0.909 | 0.745 | 0.424 | 0.157 |
  | 0.08 | 0.812 | 0.529 | **0.222** | 0.067 |

  At `σ = 0.05` the indicator wants level 3 and the cap binds instead, which
  is why the blast uses the pulse's σ and not a smaller one. Any
  `refine_tol` in `[0.20, 0.30]` yields the identical mesh.
- Blast wave with the amplitude scale frozen at `t = 0`: 1024 blocks — the
  whole domain at level 2 — against 208 for the refreshed run at `t = 0.1`.
  See the refinement section for why it fails two separate ways.
- Reduced precision, same runs: the `Float64`/`Float32` table under Precision
  above. The pulse and the blast wave reach the *same mesh* at both — same
  block count, same depth, and for the blast the same 91% ring coverage — with
  L∞ agreeing to under 1%.
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
