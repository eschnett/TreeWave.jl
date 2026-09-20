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
- Use TreeAMR's public API. Nothing here is mesh machinery that belongs
  upstream, and there are now no exceptions. There was exactly one, and
  recording it rather than quietly taking it is what got it fixed:
  `TreeAMR.block_partials`, the per-block reduction every diagnostic
  upstream is built on, was unexported and ought not to have been.
  Upstream agreed, and it is now public as `block_mapreduce` — with a
  better signature than the one this package was reaching for. See
  [Running on a device](#running-on-a-device).
- Be small enough to read in one sitting.
- Run in the caller's floating-point type, not only `Float64` — TreeAMR's
  mesh is generic and an application that is not would give the genericity
  nowhere to go. See Precision.
- Run where the caller's storage is, not only on the host — same argument,
  one milestone later. See Running on a device.
- Run where the caller's *values* are, not only at cell centres — the same
  argument a milestone later again, and the one that pays back rather than
  merely generalizing: the wave equation's natural layout is vertex
  centring, and adopting it makes the interface-order rule cheaper. See
  Centerings.

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
  not need flux matching. TreeAMR's M8 delivers it — `InterfaceSchedule`
  and `restrict_interfaces!`, measured upstream on Burgers' equation — and
  this package deliberately does not use it: there is no flux here to
  restrict. Second-order form is not a conservation law, and reaching for
  the machinery anyway would misrepresent what it is for. The half of M8
  this package *does* use is the other one, Centerings.

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

The kernel takes no centering, and that is a claim rather than an
omission: it reads its own point and its two neighbours a spacing away,
which is the same stencil wherever those points sit. Everything the
stagger changes happens at a coarse-fine interface, inside the mesh. See
Centerings.

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
every RK stage. `G` is read from a runtime field — `fs.G`, an
`NTuple{D,Int}` since M8, one width per dimension — so the *constructor*
is type-unstable by design; it is called once per chunk, never per step.
There is no `Val{C}`: the centering never reaches the kernel, because the
stencil does not depend on it. See Centerings.

## What bounds checking costs

`wave_rhs_kernel!`'s reads are `@inbounds`, and that annotation is worth
more than it looks. On the two-level mesh `wave_forest` builds at
`D = 3`, `N = 16`, 3 roots per edge -- 34 blocks, one thread, Julia
1.13.0 on an Apple M3 Pro -- with the kernel timed through `map_blocks!`
as best of 30:

| | without `@inbounds` | with |
|---|---|---|
| `wave_rhs_kernel!` via `map_blocks!` | 0.727 ms | **0.111 ms** |
| the whole `wave_rhs!` | 3.41 ms | 2.74 ms |

**6.5x on the kernel, 20% on the right-hand side.** The two numbers
differ by that much because the ghost fill dominates a refined mesh, and
that is the honest framing of what this buys: the kernel is not where
most of an evaluation goes. On a uniform mesh, or at larger `N` where
the interior-to-boundary ratio grows, its share is larger and so is the
gain.

The 6.5x is out of proportion to the number of checks removed, and that
is the part worth understanding. A 2nd-order Laplacian is a stencil that
vectorizes cleanly, and the bounds checks were preventing it. This is
not a constant factor saved on each load; it is the difference between a
scalar loop and a vector one. `du` is bit-identical either way -- the
same `sum(abs, du)` to the last digit -- which is the only acceptable
outcome for a change that removes no arithmetic.

**Why it is safe, and how that stays checkable.** Every index the kernel
forms comes from `map_blocks!`, whose contract is that the global index
runs over the owned range and nothing else; the kernel adds `G[d]` to
reach the working array and reads one point either side, which is what
`G >= 1` guarantees is in bounds and what `cell_indicator` already
refuses to run without. That makes `@inbounds` an assertion, and an
assertion wants to be falsifiable: `--check-bounds=yes` overrides
`@inbounds` package-wide, so a run under it re-checks every index the
kernel forms. `.github/workflows/CI.yml` therefore *states*
`check_bounds: 'yes'` rather than leaning on the action's default. The
flag is already that default, so this changes no behaviour today; the
point is that the package now depends on it, and a default that quietly
changed would turn the assertion into memory corruption rather than a
test failure.

The two test runs prove different things, and neither covers the other.
`Pkg.test(; julia_args = ["--check-bounds=yes"])` re-checks every index
and, precisely because it disables `@inbounds`, never runs the code the
annotation actually produces. Plain `Pkg.test()` is the only one that
exercises the optimized path, and so the only one that can catch a wrong
*answer* rather than an out-of-range index. A change that passes the
first and not the second has moved the numbers; one that passes the
second and not the first is reading out of bounds and getting away with
it today. Run both.

**One trap before generalising this.** `@inbounds` propagates into an
inlined callee only if that callee is marked `@propagate_inbounds`. An
anonymous closure is not, so

    vals = @inbounds ntuple(v -> work[c..., v, b], Val(NV))   # checks remain
    vals = ntuple(v -> @inbounds(work[c..., v, b]), Val(NV))  # checks removed

are not the same thing. The RHS kernel reads its array directly and does
not hit this, but the same rule bites through ordinary calls and this
package has one: `cell_tau` in `src/refinement.jl` is an `@inline`
helper that reads `work` on behalf of its callers and is not marked
`@propagate_inbounds`, so an `@inbounds` at a call site would not reach
inside it. It is deliberately left alone -- the criterion runs at regrid
frequency rather than per evaluation, so the case for it is much weaker
-- and if it is ever worth doing, the annotation belongs *in* `cell_tau`,
or `cell_tau` needs `@propagate_inbounds` so that its callers can
decide. Not at the call site, on the assumption that it reaches.

This is the one finding that transferred from TreeAMR's ghost-exchange
performance pass; see "What the ghost fill costs" in its `CODE.md`. The
other three there are specific to the mesh. TreeWave's kernels go
through `map_blocks!`, which already launches a `D + 1`-dimensional
ndrange, so there is no flattened box to un-flatten, and no hot kernel
here loops `CartesianIndices` over a compile-time constant.

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

## Centerings

A field set carries, per dimension, whether its values sit at cell centres
(`:cell`) or on cell boundaries (`:vertex`). This package runs
**vertex-centred by default**, and keeps the cell-centred layout
reachable, tested and measured beside it.

Vertex centring is not a generalization taken for its own sake. It is the
layout a wave equation wants, and the reason is the interface-order rule
below: along a vertex-like dimension restriction is exact **injection**,
so the restriction order stops entering the global rate at all, and
prolongation reaches one plane less, so `G = 1` suffices at order 4 where
cell centring needs 2. A smaller ghost width on the same mesh is a
smaller working array — `N + 2G + 1` stored points against `N + 2G`, which
at `N = 8` is 11 against 12 per dimension — and one fewer operator whose
order has to be got right.

**What a stagger changes, and what it does not.** A vertex-like dimension
stores one plane more, the boundary plane a block *shares* with its
high-side neighbour, and the exchange fills it exactly as it fills a
ghost. Ownership stays half-open: a block owns its points `0 … N-1` in
every dimension, so the state vector still holds `N^D` values per block
per variable. That is why so little here moved:

| | cell-centred | vertex-centred |
|---|---|---|
| owned points per block per dimension | `N` | `N` |
| stored points per dimension | `N + 2G` | `N + 2G + 1` |
| state-vector length | `N^D·nvars·nblocks` | the same |
| position of owned point `i` | `origin + (i - ½)h` | `origin + (i-1)h` |
| restriction at an interface | order-`p` stencil | injection |
| `G` needed at prolongation order `p` | `p/2` | `p/2 - 1` |

Nothing in the physics knows which it is. `wave_rhs_kernel!` takes no
`Val(C)`, `WaveProblem` carries none, the Löhner sweep walks the same
`1:N` owned points, and the boxes it reports mean the same thing. Every
driver gained a `centering` keyword and passes it to its field sets, and
that is all any of them do with it.

The rest of the port is M8's *other* change and would have been needed
with no stagger at all: `fs.G` is an `NTuple{D,Int}` rather than an `Int`
and lives on the field set, a `GhostSchedule` is built from the field set
because it belongs to a layout, `regrid!` takes `fs => schedule` pairs
because a bare field set no longer says which schedule moves it, and
`hostcopy` has to reproduce the whole layout rather than just the forest.
Every one of those fails loudly if missed, which is the pleasant half.

Three places did have to learn the difference, and all three are about
*where a value is*, not about what is done to it:

- `coordinates(fs, b, idx)` replaced `cell_center`, and takes the field
  set rather than the forest, because the answer depends on the ghost
  width and the centering and the forest carries neither.
- The 1D viewer asks `coordinates` for its x values instead of adding
  `h/2` to a block origin.
- The 2D viewer draws each sample's own dual cell — half a spacing either
  side of the sample — rather than spanning the block extent with `N+1`
  edges. On a vertex mesh those cells are offset half a spacing from the
  block outlines, which is what half-open ownership looks like when it is
  drawn, and the cells still tile exactly between same-level neighbours.
  The radial panel is the check: a half-cell error there fans the curve
  out, which is indistinguishable at a glance from a discretization
  defect.

**What it costs.** Nothing measurable. Every convergence rate, every block
count, every coverage fraction comes out the same on both layouts — the
blast wave reaches 820 blocks with growth `6.029411764705882` on each, to
the last digit — and the errors differ in the third significant figure.
The two columns are tabulated under [Measured
results](#measured-results). The cell-centred studies are kept as
`test/sinewave_cell_tests.jl`, `test/supergaussian_cell_tests.jl` and
`test/blast_cell_tests.jl`, running the identical assertions with the
centering said out loud, and both viewers take `--centering=cell`.

Keeping them is not caution. The pulse is the control — a travelling
Gaussian aligned with nothing, where the two layouts must agree, so a
disagreement there is the port and not the physics — and the blast wave
is the case where they might not have: its peak sits at the centre of the
box, which is a grid *point* on a vertex mesh, sampled at exactly 1.0, and
a block corner on a cell-centred one. That they agree anyway is a
measurement, and one that needs both halves present to keep making.

**A word on words.** The mesh still has *cells* — a block is `N^D` of
them per dimension, and `h` is their width — whatever the centering; what
changes is whether a stored value sits at a cell's centre or on its
corner. This document says "point" where the distinction matters (an
owned point, a stored point, the `N^D` the state vector holds) and
"cell" where it does not (the mesh, the spacing, a per-cell kernel).
`refinement.jl` reports boxes in cell indices `1:N` and means the owned
points with those indices; the two numberings coincide because a block
owns exactly `N` of each.

**What this package does not use.** M8's other half is conservation at
coarse-fine faces — a flux field set with a vertex-like centring in the
face dimension, `G = 0`, and `restrict_interfaces!`. There is no flux in
second-order form, so none of it appears here. See Scope and non-goals.

## Operator order: the constraint inherited from TreeAMR

TreeAMR's interface-order rule is why every driver here takes `ops` and
`G` together rather than defaulting them. A ghost filled by an order-`p`
operator carries an `O(hᵖ)` error; the 2nd-order Laplacian divides it by
`h²`, leaving `O(h^(p-2))` along every coarse-fine interface. With `p = 2`
that is `O(1)` and drags the global rate to **first** order even though
the interior scheme is second order. So the global rate is
`min(2, p - 1)`: **first** order at `p = 2`, second at `p = 4`, on either
layout.

What the centering changes is which `p`, and how much ghost it costs:

| | needs | because |
|---|---|---|
| cell-centred | order-4 prolongation **and** restriction, `G = 2` | the two sides of an interface get their ghosts from different operators, so raising one alone leaves the other first order |
| vertex-centred | order-4 prolongation, **any** restriction, `G = 1` | restriction along a stagger is injection — a coincident fine point copied, exact for arbitrary data, with no order to raise |

Both are asserted rather than documented. `test/sinewave_cell_tests.jl`
checks all four operator combinations at `G = 2`, and
`test/sinewave_tests.jl` checks them at `G = 1` and then makes the
stronger claim the vertex row implies: the two restriction orders are not
merely equally accurate, they are **the same computation**, `l2 === l2`.
It also asserts that `G = 1` and `G = 2` give bit-identical answers at
order 4 — the second ghost plane buys nothing, not almost nothing — and
that `G = 1` with `cellcentered(D)` is refused outright.

Measured rates, `D = 1` and `D = 2`, two-level mesh:

| prolongation | restriction | vertex, `G = 1` | cell, `G = 2` |
|---|---|---|---|
| 2 | 2 | 0.99 / 1.01 | 1.0 |
| 2 | 4 | 0.99 / 1.01 | below 1.5 |
| 4 | 2 | 1.99 / 1.99 | below 1.5 |
| 4 | 4 | 1.99 / 1.99 | 2.0 |

The vertex column gives `D = 1` and `D = 2`; the cell column is `D = 1`
only, which is what its testset has always asserted, and the two
off-diagonal rows there are asserted as "below 1.5" rather than as a
number because what matters is that raising one order alone does not
reach 2. The vertex rows come in pairs because they are the same run.

## Precision

TreeAMR's mesh is generic in its floating-point type: `Forest{D,T}` carries
the type the *geometry is computed in*, not merely stored in, and `FieldSet`
defaults its element type to the forest's, with `GhostSchedule` taking the
field set's. This package follows it, so that a run can be done at
`Float32` — the point being test runs
on a device with no hardware fp64, which is exactly what a low-end GPU is —
or at a MultiFloats type, which is a software float and therefore evidence
that no fp64 path is load-bearing anywhere.

That first reason stopped being hypothetical: the mesh refuses a `Float64`
field set on a backend without hardware fp64, so `Float32` is not a
reduced-precision option on such a device but the only one there is. See
[Running on a device](#running-on-a-device).

Every driver takes the type as a **leading positional argument**, spelled as
TreeAMR spells `spacing(T, forest, level)`:

    wave_errors(Float32, Val(1); N=16, G=1, ops=ops)
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

One further gap is not bridged, because nothing in the package should be
relying on it: **MultiFloats' division is not correctly rounded**, so `x / x`
is not always exactly one. Measured, it differs from one by an ulp for about
an eighth of `Float32x2` values. That surfaced in exactly one place —
`track_pulse`'s `tracking`, a ratio of two amplitudes which are the *same
number* whenever the peak sits in a refined block — where the exact claim
`tracking == 1` is true of the run and false of the arithmetic.
`test/type_tests.jl` therefore makes the exact claim at a hardware float and
the approximate one at a MultiFloat, naming the reason. It is a property of
the software float, and a fair thing for the software-float probe to find.

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

Vertex-centred, which is the default:

| run | `Float64` | `Float32` |
|---|---|---|
| pulse, `t_end = 0.5` | L∞ 0.09605, 16 blocks, level 2 | L∞ 0.09704, 16 blocks, level 2 |
| blast, `t_end = 0.4` | L∞ 0.029104, 136 → 820 blocks, 91.05% covered | L∞ 0.029168, 136 → 820 blocks, 91.05% covered |
| sine, `N = 8`, two levels | L2 0.0033486, L∞ 0.0077755 | L2 0.0033486, L∞ 0.0077748 |

Cell-centred, the same runs:

| run | `Float64` | `Float32` |
|---|---|---|
| pulse, `t_end = 0.5` | L∞ 0.09609, 16 blocks, level 2 | L∞ 0.09687, 16 blocks, level 2 |
| blast, `t_end = 0.4` | L∞ 0.029713, 136 → 820 blocks, 91.10% covered | L∞ 0.029778, 136 → 820 blocks, 91.10% covered |
| sine, `N = 8`, two levels | L2 0.0033392, L∞ 0.0074446 | L2 0.0033391, L∞ 0.0074439 |

The blast wave agrees on the block count, the depth *and* the coverage
fraction — the refinement criterion's decisions are precision-insensitive even
where the error is not, which is what makes a `Float32` run a rehearsal for
the `Float64` one rather than a different experiment. The coverage fractions
are identical to their last digit *within* a column and differ between the
columns only in the fourth: `0.9105206073752712` vertex against
`0.9109663409337676` cell-centred. So the criterion's decisions are
layout-insensitive too, very nearly but not exactly — which is the honest
statement, since the two layouts do sample different points.

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
the thresholds had to be measured. Max τ over the pulse's **initial data**
(σ = 0.08) on uniform meshes at `roots = 8`, against the depth the criterion then
reaches when the cap is set to 6 so that the *indicator* has to be what stops it:

| level | h | max τ, vertex | max τ, cell |
|---|---|---|---|
| 0 | 1/64 | 0.770 | 0.794 |
| 1 | 1/128 | 0.495 | 0.478 |
| 2 | 1/256 | 0.197 | 0.192 |

| refine_tol | depth reached (cap 6) | blocks | cells |
|---|---|---|---|
| 0.15 | 3 | 20 | 160 |
| 0.20 | 2 | 16 | 128 |
| 0.30 | 2 | 16 | 128 |
| 0.45 | 2 | 16 | 128 |

The depth table is the same on both layouts, row for row.

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

## Multi-threading

There is no switch. Start Julia with `-t` and the whole application is
parallel, because TreeAMR's M5 threads both halves of what it owns: every
per-cell kernel (KernelAbstractions' CPU backend spreads a launch over
`Threads.nthreads()`) and every host-side pass over blocks. Two of those
passes call code that lives *here* — the initial-data callback given to
`fill_by_coordinates!`, which upstream turned into a kernel, and the flag
function given to `flag_blocks`, which is where the Löhner sweep of
[`cell_indicator`](src/refinement.jl) runs. Both were parallelized by
updating the dependency and changing nothing.

That also made the package most of the way ready for TreeAMR's M6: its
initial data already ran inside a kernel that reaches the geometry
through per-block `block_origins`/`block_spacings` arrays rather than
through the tree, which is the shape a device demands. What was left, and
what it cost, is [Running on a device](#running-on-a-device).

### The callbacks are called concurrently, so they must be pure

`pulse_exact`, `wave_exact`, `blast_initial`, `blast_exact` and
`refine_mark` are now invoked from inside parallel loops. They are pure
functions of their arguments and must stay that way — a closure that
memoized, counted calls, or wrote into a captured buffer would be a data
race that no test in this suite would report as one.

The subtle case is the one that looks safe. `field_scales` is itself a
threaded reduction, so it must not be evaluated *inside* the flagging
pass, or one parallel loop would nest in another. It is not:
[`refine_flags`](src/refinement.jl) evaluates that keyword at its own call
site, and the drivers hoist it further still, to once per regrid. That is
load-bearing rather than incidental.

### What was left for the application to thread

Four loops here were reached by none of the above, and they were measured
before they were touched — 0.98×, 1.01× and 0.98× at eight threads for
the three the benchmark already covered, which is the flat line a serial
loop should give and the check that the benchmark reads what it claims
to. Each was then written the way upstream writes its own: one slot per
block (or column, or radius), combined afterwards in a fixed order, never
an accumulator shared between tasks.

Two of the four have since gone back upstream. `field_scales` and
`blast_coverage` were per-block reductions over field data, which is
exactly what `TreeAMR.block_mapreduce` is, and a per-block reduction over
field data is the one shape that has to change on a device. They now call
it and the discipline above is upstream's to keep; the two that remain
here are the Hankel table and its contraction, which are host `Float64`
tables and not field data at all. See
[Running on a device](#running-on-a-device).

**The Hankel contraction is the instructive one.** Its natural outer loop
is over modes, and that loop cannot be split: every `k` contributes to
every `r`, so `us` and `vs` would be shared and the summation order — and
therefore the last bits of the answer — would follow the thread count.
Partitioning the *radii* instead leaves each `us[i]` summed over all modes
in the order it always was, so the result is bit-identical for any
partition at all. It is also the loop that barely speeds up: it streams
the whole 32 MB `J₀(kr)` table and is bandwidth-bound, and the table is
first-touched by column in `blast_reference` and read by row here, so no
page placement serves both passes. It is threaded anyway — the partition
that keeps it exact costs three lines.

### Bit-identical, not merely to roundoff

TreeAMR's M5 asks only that threaded results match serial to roundoff, and
delivers exact equality instead. An application can give that property
away in one line, so this package holds itself to the same standard and
tests it: [`test/thread_workload.jl`](test/thread_workload.jl) runs
`track_pulse` and `track_blast` — adaptation, evolution, flagging,
regridding, the quadrature and the reductions — and prints a digest of the
state vector at every chunk through the `observer` keyword, while
[`test/threading_tests.jl`](test/threading_tests.jl) compares a subprocess
at a different thread count against the run in hand, character for
character.

Two decisions in that test are worth keeping. The comparison is between
two *live* runs and never against a committed digest: bit-identity holds
across thread counts, not across TreeAMR versions — upstream's M5 changed
`volume_weighted_norm` from a running total to a sum of per-block
partials, which is a different summation order and may move the last bits
of any recorded L2 number. And the workload uses only `Base` and the
package (`hash`, `repr`), so it runs from the root environment as well as
from `test/`, which is what makes a mismatch bisectable.

The workload runs vertex-centred only, and that is deliberate rather than
an omission. Thread determinism is a property of the *shapes* of the
reductions — one slot per block, combined in a fixed order — and a
centering changes none of them: the same blocks, the same `N^D` owned
points, the same partial arrays. A second workload would double the
test's ten seconds, most of which is a subprocess paying Julia's startup
again, and buy no claim that the first does not already make.

### What it buys, and what caps it

Measured on one exclusive node of Symmetry's `amddebugq` — 64-core AMD
EPYC, 8 NUMA domains, no SMT — on a two-level 2D mesh of **1792 blocks of
128², 29.4M points**, a 484 MiB working array and a 448 MiB state vector,
vertex-centred at `Float64`. Speedups against one thread at the same page
placement, with the default first touch and with the pages interleaved
(`numactl --interleave=all`). Both placements come from the same node in
the same job, because two nominally identical nodes measured 20% apart:

| phase | 8t | 16t | 32t | 64t | 64t interleaved |
|---|---|---|---|---|---|
| RK4 step (`solve`) | 3.32 | 3.56 | 3.62 | **3.58** | 3.50 |
| RHS evaluation | 7.84 | 12.2 | 14.0 | **12.7** | 16.7 |
| stage broadcast | 1.01 | 1.01 | 1.08 | **0.99** | 0.99 |
| initial data | 8.17 | 16.4 | 31.5 | **40.2** | 63.5 |
| refinement flags | 3.93 | 4.04 | 3.70 | **3.35** | 3.37 |
| amplitude scales | 7.99 | 15.7 | 9.72 | **8.13** | 6.18 |
| exact ring (`blast_exact`) | 7.99 | 15.6 | 31.0 | **48.4** | 46.6 |
| Hankel table | 7.62 | 14.6 | 29.6 | **38.3** | 54.6 |
| Hankel contraction | 1.63 | 2.15 | 1.28 | **2.28** | 2.70 |
| coverage reduction | 8.02 | 14.4 | 10.6 | **12.3** | 13.9 |

The runs behind it are vertex-centred, and the sweep was not repeated
cell-centred: the per-phase device table measures both layouts on the same
machine and finds them equal to within the run-to-run spread, and
parallelism has no more reason than arithmetic does to care where inside a
cell a value sits.

**The mesh scales and the step does not, and that is the result.** The
right-hand side — scatter, ghost fill and the Laplacian kernel, which is
all this application asks the mesh for — reaches 12.7× (16.7×
interleaved). A whole RK4 step reaches 3.6× and stops there, by 16
threads. The arithmetic is not mysterious: at one thread, four RHS
evaluations are 2.44 s of a 3.15 s step and the four stage broadcasts are
0.28 s, so the rest is the integrator's own copying; at 64 threads the
same step is 0.878 s, of which the RHS is 0.19 and the other 0.69 — 79% —
does not scale. `OrdinaryDiffEq` does its stage arithmetic as ordinary
broadcasts over the state vector, and the benchmark times one of them
directly (`stage_broadcast`, flat at 1.0× across the whole sweep)
precisely so the ceiling is a measured number and not an inference.

The obvious fix is one keyword — `RK4(thread = True())`, which sends those
same broadcasts through Polyester — and it is **worse**: measured at eight
threads, 0.030 s per step becomes 0.064, reproducibly and in both
orderings. Polyester spins up a second thread pool alongside the tasks
KernelAbstractions is already using, and the two oversubscribe the cores.
It also costs two new direct dependencies (`Static` for the keyword's
value, `Polyester` for the extension that implements it). So the honest
statement is not "we could turn it on" but "the stage updates have to join
the pool that is already running" — which means a time integrator written
as kernels, listed under Possible extensions rather than done.

**Interleaving the pages helps, and an earlier reading of this table said
it did not.** That earlier entry recorded "the RHS is *slower*
interleaved (0.033 s against 0.047)" and concluded that TreeAMR's 2–6×
did not reproduce downstream. The two seconds in it are almost exactly
what this sweep measures — 0.0365 interleaved against 0.0479 default at
64 threads — but they are the *interleaved* figure first, so they say the
opposite of the sentence around them. The conclusion was drawn from a
transposition, and it is withdrawn: at 64 threads interleaving makes the
RHS 24% faster, initial data 37% faster (0.0484 against 0.0763) and the
Hankel table 29% faster, while the whole step is 2% slower and the
per-block reductions are a wash. That is smaller than upstream's 2–6× and
in the same direction, which is the unremarkable answer the first reading
turned into a puzzle.

The reason the effect is modest here is still worth keeping: every large
array gets a good first touch for free, because `FieldSet` and
`statevector` allocate untouched pages and the first thing to write them
is a threaded kernel partitioned by block, which is how every later
kernel partitions them too. The two-placement sweep stays in
`bin/benchmark.sbatch` so the question can be re-asked on another machine
rather than assumed — and, on this evidence, so that a sign error in
reading it is caught the next time rather than the time after.

Three phases do not scale, and the reasons differ. The stage broadcast is
serial by construction, above. The Hankel contraction is a memory-bound
pass over tens of megabytes, which saturates at 2× however many threads
are added. And the amplitude scales and the coverage reduction are erratic
at high counts — 15.7× at 16 threads and 8.1× at 64 for the scales —
because at that point the pass itself is a few milliseconds and the cost
of spawning is not negligible against it; both are once per regrid, so
this is left alone rather than given a grain-size heuristic.

**The refinement flags are the row that changed, and the port is not
why.** They reach 3.9× at 8 threads and then *fall* — 4.04, 3.70, 3.35 —
where the entry here previously recorded 8.00× and 33.9×. That was worth
attributing rather than overwriting, so it was measured directly: the
pre-M8 package, at its own TreeAMR pin and cell-centred, was run beside
this one **in a single job on a single node**, and

| phase | pre-M8, cell | this, vertex |
|---|---|---|
| `refine_flags`, 1 → 16 threads | 2.378 → 0.664 s (3.58×) | 2.388 → 0.697 s (3.43×) |
| `initial_data`, 1 → 16 | 3.071 → 0.191 s (16.1×) | 3.069 → 0.193 s (15.9×) |
| `rhs`, 1 → 16 | 0.582 → 0.0474 s (12.3×) | 0.613 → 0.0533 s (11.5×) |

— so the old figure does not reproduce for the *old code* either, on this
machine today. Whatever changed is the machine or the Julia version, not
M8 and not the stagger. The shape of the curve says a serial remainder of
roughly 0.6 s in a 2.4 s pass, which is what `flag_blocks` boxing 1792
`(flag, box)` tuples into a `Vector{Any}` and then narrowing it would look
like; that code is upstream's and unchanged. Recorded as a number that
moved and is only half explained, rather than quietly replaced.

The compute-bound phases behave as they should: initial data 40×, the
exact ring 48×, the Hankel table 38× — all more than the memory-bound RHS,
and the first two more than linear, which is what happens when one
thread's working set is limited by one NUMA domain's bandwidth and 64
threads' is not.

## Running on a device

There is no switch here either. TreeAMR's M6 makes the *storage* decide:
`backend` is a keyword on `FieldSet` and on nothing else — since M8 a
schedule is built from the field set and takes the backend with the rest
of the layout — after which `statevector` allocates where the field set
lives, `regrid!` reallocates there, and every kernel takes its backend
from the array it is handed. Every driver therefore takes `backend`
alongside its `T`, defaulting to `CPU()`:

    track_blast(Float32, Val(2); backend = MetalBackend())
    wave_errors(Float32, Val(1); N = 16, G = 1, ops = ops, backend = CUDABackend())

`T` and `backend` travel together because on a device they are not
independent: a `Float64` field set on a backend without hardware fp64 is
refused at construction, with the reason. That is what
[Precision](#precision) was *for*, and the order the two milestones were
done in was not arbitrary — a device port on top of a `Float64`-only
application would have been a rewrite rather than a keyword, and the
`Float32` work was where the stray `Float64` literals were found while
they were still cheap to find.

No device package becomes a dependency. Neither CUDA nor Metal is a
dependency of TreeAMR, `KernelAbstractions.allocate` is the whole of the
interface, and this package adds nothing to that. What needs one is the
caller: the scripts in `bin/` load it on demand (see `bin/backend.jl`)
and the device tests are opt-in behind an environment variable.

Loading it on demand has one trap worth recording, because it cost time
and because *anything* that resolves a backend from a command-line flag
will hit it. A `using` issued from inside a running function adds methods
to a later world, so every dispatch on the new backend type still
resolves to whatever generic method KernelAbstractions defines:
`allocate` throws a `MethodError` naming a method the same message lists
as a candidate, and `supports_float64` quietly answers `true`, so a
`--type=f64` guard passes and the mesh raises the objection instead.
`bin/backend.jl` therefore hands the work to `invokelatest` — the check
and the run together, not the run alone.

### What an application has to do that the mesh does not

Four things in this package reached into block storage from the host, and
they are the whole of the port.

**The RHS kernel reads a spacing per block.** That array is the one piece
of geometry the kernel needs and the one thing the mesh does not place for
the application: `block_spacings` returns a host `Vector`, and
[`WaveProblem`](src/evolution.jl) uploads it to the field set's backend
when it is built. `to_backend` in [`src/device.jl`](src/device.jl) is six
lines and is deliberately not `TreeAMR.todevice` — this package stands
for a downstream user of the public API, and the point is that an
application needs nothing privileged. **The RHS kernel itself did not
change**, which is the argument for having written it as a kernel in the
first place.

**The refinement criterion walked cells on the host.** This is the one
that needed a second implementation, because `flag_blocks` calls
`f(b, key)` on the host and a realistic criterion reads its block's data.
`firing_boxes` is the device form: the mesh evaluates a per-point predicate
over every block in one kernel and returns each block's firing count and
bounding box, and the application turns that into flags. See below. The
two forms derive a point's *stored* index separately — `fs.G[d]` here,
inside the kernel there — which is exactly what a stagger could make them
disagree about, so `test/device_tests.jl` compares them on both
centerings.

**Three diagnostics were per-block reductions over field data.**
`field_scales`, [`blast_coverage`](src/blast.jl) and `track_pulse`'s
tracking measure go through `block_mapreduce` — threaded host views on
the CPU, one kernel work item per block otherwise, per-block values
combined in block order either way. They call it rather than reimplement
it because a second copy of a determinism argument is a second thing to
get wrong. What stayed here is the part that needs the *tree* — which
blocks are refined, which are at the finest level — reducing one number
per block.

That name was `TreeAMR.block_partials` when this was written, and using
it was the one documented exception to the "public API only" rule, with a
request to make upstream. Upstream took the request and, in taking it,
found three things wrong with the signature this package had been
working around — worth recording, because finding them is the argument
for reporting an internal you depend on rather than quietly depending on
it:

- It took a host block-reducer *and* a kernel-form fold, which had to
  agree and which nothing checked. Every call here passed both, spelled
  twice (`w -> maximum(abs, w)` and `(a, x) -> max(a, abs(x))`). They can
  disagree: upstream measured a 3.0e-15 divergence in `D = 1` with a
  scalar variable selection, where the host view goes `IndexLinear` and
  `sum` turns pairwise. The public form is `mapreduce`-shaped, so the
  four call sites here each collapsed to one line.
- It took a bare array plus the matching ghost offset. Every call here
  passed `fs.work, fs; g=G`, and getting `g` wrong would have read the
  ghosts silently. The public form takes the field set and works it out.
- Its variable selection meant different things on the two backends. The
  public form refuses anything a kernel cannot take, with a reason.

**One callback closed over arrays.** [`blast_exact`](src/blast.jl) closes
over the two radial profiles of the Hankel table, so on a device they are
converted to `T` and uploaded once per error measurement. The table
itself stays a host `Float64` object, which was already the design (see
[Precision](#precision)) and is now load-bearing rather than tidy.

### Callbacks become kernel arguments

Everything a callback captures must be `isbits`, and a captured **`Type`**
is the trip. `zero(T)` inside a closure whose `T` is a local variable puts
a `Type` in a kernel argument; `zero(x[1])` reads the type off the
coordinate the mesh already handed over. Where a `Vector` was captured it
becomes a device array or a tuple. And two generators —
`prod(… for d in 1:D)` in `wave_exact` and `sum(d -> …, 1:D)` in
`blast_initial` — became accumulating loops, the shape TreeAMR's own
device-tested closures use; multiplying by one and adding zero are exact,
so no recorded number moved.

This is the same purity requirement the concurrent callbacks of
[Multi-threading](#multi-threading) already imposed, tightened: a closure
that was merely *racy* on threads does not compile at all for a device,
which is the better failure.

### The criterion has two forms, and keeps both

[`refine_flags`](src/refinement.jl) dispatches on `get_backend(fs)`: the
host loop over points on the CPU, `firing_flags` — built on
`firing_boxes` — on a device. The choice lives there so that no *driver*
has to make it: a driver passes `backend` to its field set and then calls
`refine_flags`, and nothing in between knows which form ran.

The host form is kept rather than retired in favour of the one that runs
everywhere, and not out of caution:

- it returns `τmax`, which the 2D viewer draws and
  `test/refinement_tests.jl` makes claims about, where a firing count
  does not;
- it is what every number recorded under
  [Measured results](#measured-results) was taken with;
- it is the readable statement of the criterion; and
- it is what the device form is **tested against**. Both are available on
  the CPU backend, so `test/device_tests.jl` compares them there, flag
  for flag and box for box, on pulse data and on blast data, at two caps,
  at two precisions and — since M8 — on both centerings. Nothing else in
  the suite would catch a drift: a device run that flagged differently
  would still run, and would merely build a different mesh. The centering
  belongs in that product because the two forms work out a point's stored
  index independently, and a stagger is what makes those two derivations
  able to disagree.

The two share their arithmetic — one `cell_tau`, whose argument list is
not this package's choice but exactly what `firing_boxes` hands a
predicate — so what is duplicated is the walk over cells, not the
criterion.

**The device form needs two sweeps, because the criterion has two
thresholds.** A firing count answers one yes-or-no question per block and
[`refine_mark`](src/refinement.jl) asks two: is any cell above
`refine_tol` (a count of zero is exactly `τmax ≤ refine_tol`, since the
count is over cells and `τmax` is their maximum), and where are the cells
above `coarsen_tol`. The second is not recoverable from the first, so the
cells are walked twice. That is the cost of the form, paid once per
regrid against an evolution of many steps, and it still measures faster
than the host loop.

### What stays on the host, and why that is not a gap

- **The tree.** It is the mesh's own bookkeeping, and upstream keeps it on
  the host; `regrid!` completes marks and rebuilds the leaf array there.
- **The Hankel quadrature.** `besselj0` has hardware-float methods only,
  and a reference table is the last thing worth uploading as fp64. It is
  four million Bessel evaluations once per run against an evolution of
  thousands of steps, and it is the most favourable shape a *threaded*
  loop can have — which is where it already is.
- **The viewer.** CairoMakie is host code and reads single points.
  `hostcopy` brings a field set down in one call at the top of each
  `snapshot`, and no line of figure code below it knows the difference. It
  has to copy the whole *layout* and not merely the forest — ghost width
  and centering both, since M8 — or the host array is a different shape
  from the device one and `copyto!` is the least of the problems.
  That is for a consumer that *cannot* move, not one that has not been
  moved: the numeric diagnostics deliberately do not work this way,
  because a copy of the whole state per chunk is hundreds of megabytes on
  a run worth putting on a device at all.

### What it buys on this hardware, and what it does not

Measured on an Apple M3 Pro — 12 CPU cores, 18 GPU cores, unified memory
— at `Float32`, on the same two-level 2D mesh the thread scaling uses:
**1792 blocks of 128², 29.4M cells**, a 242 MB working array
vertex-centred and 238 MB cell-centred (`N + 2G + 1` stored points per
dimension against `N + 2G`, at `G = 2` in both columns so that the two
are a like-for-like comparison). The host column is the same machine at
eight threads, so this is a device against a *threaded* host and not
against a serial one.

Vertex-centred:

| phase | CPU, 1 thread | CPU, 8 threads | Metal | Metal vs CPU 8t |
|---|---|---|---|---|
| `step` (one RK4 step) | 0.889 | 0.244 | 0.363 | 0.7× |
| `rhs` | 0.188 | 0.0405 | 0.0658 | **0.6×** |
| `stage_broadcast` | 0.0176 | 0.0171 | 0.0121 | 1.4× |
| `initial_data` | 1.585 | 0.258 | 0.0492 | **5.2×** |
| `blast_exact` | 1.000 | 0.176 | 0.0753 | 2.3× |
| `refine_flags` | 1.015 | 0.264 | 0.148 | 1.8× |
| `field_scales` | 0.0679 | 0.0116 | 0.0161 | 0.7× |
| `blast_coverage` | 0.0681 | 0.0119 | 0.0163 | 0.7× |
| `blast_reference` | 0.620 | 0.0987 | 0.615 | host |
| `blast_radial_table` | 0.0011 | 0.0008 | 0.0010 | host |

Cell-centred, the same machine in the same sweep:

| phase | CPU, 1 thread | CPU, 8 threads | Metal |
|---|---|---|---|
| `step` | 0.846 | 0.257 | 0.373 |
| `rhs` | 0.183 | 0.0377 | 0.0651 |
| `stage_broadcast` | 0.0182 | 0.0167 | 0.0195 |
| `initial_data` | 1.563 | 0.259 | 0.0447 |
| `blast_exact` | 1.012 | 0.197 | 0.0758 |
| `refine_flags` | 1.018 | 0.294 | 0.147 |
| `field_scales` | 0.0679 | 0.0107 | 0.0163 |
| `blast_coverage` | 0.0676 | 0.0153 | 0.0154 |

Seconds; the minimum of several, each synchronized.

**The centering costs a few per cent on the RHS and nothing anywhere
else.** A vertex-like dimension stores one plane more — 133 against 132
per dimension at `N = 128, G = 2`, 1.5% more bytes — and the one phase
that is pure bandwidth pays a little more than that: the RHS is 2.6%
slower vertex-centred at one host thread and 7% slower at eight, on Metal
1%. The compute-bound phases and the reductions agree between the layouts
to within the run-to-run spread. This is the honest version of "the
stagger is free": it is free where the arithmetic dominates, and costs
about what the extra plane weighs where the memory does. The ratios below
are quoted from the vertex column and hold for either.

These numbers are lower across the board than the ones recorded here
before M8, on both layouts and on both backends — the eight-thread host
column most of all (a `step` of 0.244 against 0.348). That is the machine
and the Julia version, not the port: the cell-centred column *is* the
pre-M8 configuration, and it moved by the same amount. Ratios between
columns of one sweep are the thing to read; absolute seconds across
sweeps are not.

Read the rest as two populations:

**The compute-bound callbacks win, by 2–5×.** `initial_data` and
`blast_exact` do real arithmetic per point — an `exp` and integer powers,
or nine periodic images and two interpolations — and that is what a GPU
is for. `refine_flags` wins 1.8× *despite* walking every point twice, so
the two-sweep form of the criterion is not what holds it back; what does
is upstream's deliberate one-work-item-per-block reduction, which
TreeAMR measures at 5.5× for a single `firing_boxes` and explains as the
price of determinism.

The three per-block reductions — `field_scales`, `blast_coverage` and
the norms — are flat or slightly *behind* the threaded host for that same
reason, 0.7×. None is on the per-evaluation path.

**The bandwidth-bound kernels do not, and cannot on this machine.** The
RHS is a 3-point stencil per dimension: in 2D six reads and two writes
per point, with almost no arithmetic between them. On Apple silicon the
CPU and the GPU share one memory controller, so there is no bandwidth
ratio to win — and measured, there is none: 0.6×, the device *behind* the
threaded host. That is not a finding
about the code. TreeAMR measured **35.5×** on this same RHS on an H200
against 16 host cores, against a triad bandwidth ratio of 18.7; the
number here is what the same code does when the ratio is 1. A
memory-bound kernel moves to a device to get its memory, and on unified
memory it is already there.

#### The same code on a discrete GPU

The paragraph above says the Metal result is a statement about unified
memory and not about the code, and quotes TreeAMR's H200 number as the
evidence. That quotation is now a measurement of *this* package, on
Symmetry's `h200q`: one NVIDIA H200 against 16 AMD EPYC cores of the same
node, `Float32`, the identical 29.4M-point mesh.

| phase | CPU, 16 threads | H200 | speedup |
|---|---|---|---|
| `step` (one RK4 step) | 0.599 | 0.00807 | 74× |
| `rhs` | 0.0524 | 0.00147 | **36×** |
| `stage_broadcast` | 0.0230 | 0.000383 | 60× |
| `initial_data` | 0.171 | 0.000949 | **180×** |
| `blast_exact` | 0.290 | 0.00224 | 129× |
| `refine_flags` | 0.468 | 0.122 | 3.8× |
| `field_scales` | 0.0293 | 0.0127 | 2.3× |
| `blast_coverage` | 0.0206 | 0.0128 | 1.6× |
| `blast_reference` | 0.0296 | 0.218 | host |

Cell-centred on the same node and in the same job: `rhs` 0.0547 against
0.00148, `initial_data` 0.187 against 0.000950, `refine_flags` 0.454
against 0.167 — the same conclusion, and the layouts again agree to within
the run-to-run spread.

**36× on the RHS**, against TreeAMR's 35.5× for the same kernel on the
same hardware, and against 0.6× on the M3 Pro. Nothing in the application
differs between those two rows: the same kernel, the same field set, the
same `--backend=` flag. What differs is whether the device has memory the
host does not, and a memory-bound kernel is worth moving only when it
does. Recording both is the point — one machine alone would have supported
either "GPUs do not help this" or "GPUs give 36×", and neither is the
finding.

The two rows that do *not* scale are the same two as everywhere else, and
for the reason already given: `field_scales` and `blast_coverage` are
upstream's one-work-item-per-block reductions, whose determinism is bought
with parallelism, and `refine_flags` is built on the same shape. They cost
3.8× and less here while the per-point kernels cost two orders — which is
the sharpest version of that trade this package has measured.

`stage_broadcast` is worth a second look on both machines. On the M3 Pro
8 host threads are barely faster than one (0.0171 against 0.0176),
because five state-sized streams saturate the bus — the same ceiling
[Multi-threading](#multi-threading) measures as most of a 64-thread step.
Metal does it in 0.0121 — barely a win, and for the same reason. The H200
does it in 0.000383, 60×, because there the bus is a different bus: the
Amdahl term that caps a threaded step is not intrinsic to the broadcast,
it is intrinsic to *host* bandwidth.

**And a whole adaptive run is slower, by design of the problem and not of
the code.** `track_blast` at the calibrated size — 820 blocks of 8², 52k
points — takes 1.07 s on eight host threads and 6.98 s on Metal; at
`roots = 16, N = 16, chunk = 0.01` (256 blocks of 16², 66k points) the gap
narrows to 1.1 s against 3.6 s. A run that small is launch-bound: an RK4
step is four RHS evaluations, each a scatter, several ghost phases and a
kernel, each synchronized, over blocks of 64 points. The phase table above
is at 29.4M points for exactly this reason, and the honest summary is that
the mesh sizes this package's *tests* use are two orders of magnitude
below where a device begins to pay. (`chunk` has to shrink with `h` in the
second run: `refinement_buffer` refuses a margin wider than a block, and
at `N = 16, roots = 16` the default `chunk = 0.02` would need 22 cells.)

### Reproducing it

`bin/benchmark.jl --backend=metal --type=f32` prints the table in the
format `--backend=cpu` prints, so the two can be diffed. The device
package must be in the environment the script is run against, which is
one command and is in the script's header — the package environment
deliberately does not have it. `--centering=cell` gives the other column
of the same table, and the header line names the centering so two files
can be told apart. Both viewers take `--backend=` and `--centering=` too,
and that is the quickest way to see that a device run is the *same run*:
same blocks, same levels, same τ, the same figure.

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
| `src/device.jl` | placing an application's own arrays on a backend, and bringing a field set back to the host — see Running on a device |
| `src/evolution.jl` | the RHS kernel, `WaveProblem`, `wave_rhs!`, `convergence_rate` |
| `src/refinement.jl` | the per-cell refinement indicator and its reduction to block marks |
| `src/sinewave.jl` | the standing mode and its convergence driver |
| `src/supergaussian.jl` | the travelling pulse, its AMR driver, and the uniform reference |
| `src/blast.jl` | the radial blast wave, its Hankel-quadrature exact solution, its AMR driver, and the uniform reference |
| `src/benchmark.jl` | per-phase and end-to-end timings, for the thread-scaling measurement |
| `test/sinewave_tests.jl` | convergence order, the interface-order rule, the stagger's two consequences, 3D smoke test, energy drift |
| `test/refinement_tests.jl` | claims about the indicator itself |
| `test/supergaussian_tests.jl` | a moving refined region tracks the pulse |
| `test/blast_tests.jl` | the exact ring, 2nd-order convergence to it, a growing refined region, and what a frozen amplitude scale costs |
| `test/sinewave_cell_tests.jl` | the three above, cell-centred: the identical assertions with `centering = cellcentered(D)` said out loud, so the numbers measured before vertex became the default stay under test — see Centerings |
| `test/supergaussian_cell_tests.jl` | " |
| `test/blast_cell_tests.jl` | " |
| `test/type_tests.jl` | the drivers at `Float32` and at MultiFloats' `Float32x2` — see Precision |
| `test/threading_tests.jl` | the answer does not move with the thread count |
| `test/device_tests.jl` | the two forms of the criterion agree, the geometry follows the storage, and — with a device — a whole run reproduces the host run |
| `test/thread_workload.jl` | not a test: the standalone run whose digests the above compares across thread counts |
| `bin/visualize.jl` | CairoMakie viewer for the 1D cases (own environment; see `bin/Project.toml`) |
| `bin/visualize2d.jl` | CairoMakie viewer for the blast wave — a different figure, not a flag on the other one |
| `bin/benchmark.jl` | the benchmark's CLI — the one script in `bin/` that uses the *package* environment, since it needs no CairoMakie |
| `bin/benchmark.sbatch` | the thread and page-placement sweep; a SLURM job and an ordinary shell script at once |
| `bin/backend.jl` | `--backend=`, shared by all three scripts: loads a device package on demand and runs the work in the world that load created |
| `.github/workflows/CI.yml` | tests on a Julia matrix, at one thread and at four, plus a job that renders the figures on both layouts; coverage is collected and uploaded from the single-threaded cells only |

`bin/visualize.jl` draws four panels per 1D case — the solution, the
pointwise error in `u`, the indicator τ, and the volume-weighted L2/L∞
norms over the whole state against time — with one line per block colored
by refinement level. `--ops=2` reruns with order-2 operators, which is the
quickest way to see the interface-order rule rather than read about it.
`--type=f32` reruns in single precision, which is the quickest way to see
that it is the *same run* — same blocks, same levels, same τ — rather than
read that either. Both viewers take it; only the two hardware types are
offered, because Makie cannot plot a MultiFloat. `--backend=metal` (or
`cuda`) makes the same argument about the *storage*, and needs
`--type=f32` on a device without hardware fp64. `--centering=cell` makes
it about the *layout*, and is the one of the three where something does
visibly move: the samples shift half a spacing, and nothing else does.
All three flags keep the default spelling's filename, so a comparison
render never overwrites the figure CI checks.

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
conversions the default render does not exercise, and all three again
cell-centred, which is the one comparison where something visibly moves
and the only check that both viewers' geometry still comes from
`coordinates` — and keeps them as artifacts. One extra entry of the matrix runs the suite on **four
threads**: the thread-independence test spawns a subprocess at the *other*
count either way, so one threaded job and the serial ones between them
cover both directions, while a whole extra dimension of the matrix would
only buy the same guarantee four times over. The figure job
exists because `bin/` carries its own environment and therefore its own copy
of the TreeAMR dependency: during development that let the viewer keep
building against an older TreeAMR than the tests, until it failed on an API
the tests were already using. Nothing in the test job could have caught that.

**No CI runner has a GPU**, so `test/device_tests.jl` runs there on the
CPU backend, which needs no device package and is the default. That is
less of a gap than it sounds, because the assertion most worth guarding
is the one that needs no device: the two forms of the refinement
criterion must agree, on both centerings, and both forms are available on
the CPU backend. What CI
cannot check is that a device run *works* — for that, add a device
package to an environment of your own and set `TREEWAVE_TEST_BACKEND`;
see [Running on a device](#running-on-a-device).

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
- Sine mode, two-level mesh, order-4 operators: L2 and L∞ rates both 2.0
  (D = 1, 2), at `G = 1` vertex-centred and `G = 2` cell-centred.
  Vertex-centred L2 rates 1.99 (D = 1) and 1.99 (D = 2); cell-centred the
  same to two figures.
- Sine mode, two-level mesh: the interface-order rule, and the two forms
  it takes. Cell-centred, D = 1: L2 rate 1.0 at (2,2), below 1.5 at (4,2)
  and at (2,4), and 2.0 at (4,4) — *both* orders matter. Vertex-centred,
  `G = 1`: 0.99 / 0.99 / 1.99 / 1.99 in D = 1 and 1.01 / 1.01 / 1.99 /
  1.99 in D = 2 — only the prolongation matters, and the pairs are
  bit-identical because restriction along a stagger is injection. Also
  bit-identical: `G = 1` against `G = 2` at order 4 on the vertex layout.
- Sine mode, four periods on a two-level mesh: L∞ amplitude within 5% of
  its initial value.
- Pulse, `n = 1`, `σ = 0.08`, `roots = 8`, vertex-centred:
  uniform-coarse (`N = 8`) L∞ error 1.4063, uniform-fine (`N = 32`)
  0.0930. The adaptive run (`N = 8`, two levels) gives 0.09605 — a ratio
  to the fine reference of 1.033 — at **128** points against the fine
  mesh's 256. The pulse peak never leaves a refined block over the whole
  run. Cell-centred: 1.4330 / 0.09282 / 0.09609, ratio 1.035, the same
  16 blocks at level 2.
- Old amplitude criterion vs new Löhner criterion on that same run: 0.0926
  at 176 cells against 0.0961 at 128 cells. The resolution criterion buys a
  27% cell saving for a 4% error increase, and unlike the old one it is not
  told the depth — it discovers level 2 and stops there.
- Buffer width, same run (derived width is 7 cells for `chunk = 0.02`):

  | buffer | worst L∞, vertex | ratio | worst L∞, cell | ratio |
  |---|---|---|---|---|
  | 7 (derived) | 0.09605 | 1.033 | 0.09609 | 1.035 |
  | 2 (too narrow) | 0.11575 | 1.245 | 0.10708 | 1.154 |
  | 0 (none) | 0.14307 | 1.538 | 0.14536 | 1.566 |

  128 points in every row but the cell-centred `buffer = 0`, which is 112.
  Wider is monotonically better here, and only the derived width meets the
  test's `rtol = 0.1` against the uniform-fine reference. This **does not**
  reproduce TreeAMR's recorded observation that a margin narrower than the
  motion measures *slightly worse than no buffer at all*: at `buffer = 2`
  the error is clearly better than at `buffer = 0`, not worse. Recorded as a
  contradiction rather than smoothed over — it may be geometry-specific, and
  TreeAMR measured it on a different setup.
- Blast wave, `σ = 0.08`, `roots = 8`, `t_end = 0.4`. Uniform meshes against
  the Hankel-quadrature exact solution:

  | N | h | L2, vertex | L∞, vertex | L2, cell | L∞, cell | points |
  |---|---|---|---|---|---|---|
  | 8 | 1/64 | 0.07717 | 0.30982 | 0.07717 | 0.30819 | 4096 |
  | 16 | 1/128 | 0.019626 | 0.081591 | 0.019627 | 0.081086 | 16384 |
  | 32 | 1/256 | 0.0049269 | 0.020605 | 0.0049269 | 0.020540 | 65536 |

  Rates 1.98 (L2) and 1.96 / 1.95 (L∞) — the quadrature is right to well
  past what the scheme can see. (The 1.99 recorded here before was the
  same 1.985, rounded the other way.)
- Blast wave, adaptive (`N = 8`, cap 2) against those: L∞ 0.029104
  vertex-centred at 52480 points, so **10× better than uniform-coarse and
  1.41× worse than uniform-fine at 80% of its points**; cell-centred
  0.029713, ratio 1.45. That is a weaker claim than the
  pulse's `rtol = 0.1` match, and it is the honest one: 9% of the ring's
  cells sit on level-1 blocks whose τ fell below `refine_tol`, which is the
  criterion trading accuracy for cells rather than failing to. The pulse
  matched uniform-fine only because its refined region was, relatively, far
  more generous.
- Blast wave, mesh growth: 136 blocks after the initial adaptation to 820 at
  `t = 0.4`, a factor of `6.029411764705882` — **the same digits on both
  layouts** — with the depth an output: at `σ = 0.08` the indicator's τ
  falls to 0.23 at level 2, below `refine_tol = 0.30`, so it stops there and
  `maxlevel_cap` never binds. Max τ on uniform meshes at `t = 0.4`, which is
  the calibration that fixes σ — the ring at its widest and faintest, not
  the initial peak:

  | σ | h=1/64 | h=1/128 | h=1/256 | h=1/512 |
  |---|---|---|---|---|
  | 0.05, vertex | 0.905 | 0.747 | 0.431 | 0.159 |
  | 0.05, cell | 0.906 | 0.746 | 0.428 | 0.160 |
  | 0.08, vertex | 0.818 | 0.544 | **0.227** | 0.069 |
  | 0.08, cell | 0.820 | 0.540 | **0.229** | 0.069 |

  The two layouts agree to 0.005 everywhere, so the calibration is not one
  that had to be redone — but it had to be *re*measured to say so, and the
  recipe is recorded here because the earlier entry did not state it and
  reproducing it took a second attempt: it is the evolved solution at
  `t_end`, not the initial data. (On the initial data the same table reads
  0.65 / 0.32 / 0.11 / 0.03 at σ = 0.08, which is a different question and
  the wrong one for choosing σ.)

  At `σ = 0.05` the indicator wants level 3 and the cap binds instead, which
  is why the blast uses the pulse's σ and not a smaller one. Any
  `refine_tol` in `[0.20, 0.30]` yields the identical mesh.
- Blast wave with the amplitude scale frozen at `t = 0`: 1024 blocks — the
  whole domain at level 2 — against 220 vertex-centred and 208 cell-centred
  for the refreshed run at `t = 0.1`. See the refinement section for why it
  fails two separate ways.
- Reduced precision, same runs: the `Float64`/`Float32` tables under Precision
  above, one per layout. The pulse and the blast wave reach the *same mesh* at
  both — same block count, same depth, and for the blast the same 91% ring
  coverage — with L∞ agreeing to under 1%.
- Centring, same runs: the two layouts reach the *same mesh* as each
  other too — the pulse 16 blocks at level 2, the blast 136 → 820 with an
  identical growth factor to sixteen digits, coverage 0.91052 against
  0.91097, and L∞ within 2%. They also cost the same to run, except on the
  bandwidth-bound RHS, where the vertex layout's extra stored plane makes
  it 2.6% slower at one host thread and 7% at eight. The two columns are
  side by side in the tables above and in
  Centerings.
- Sine mode at 0.9 periods, `N = 16`, `roots = 4`, two levels: final L∞
  0.00641 with order-4 operators against 0.1238 with order-2 — a factor of
  19 for a change that touches only the ghost points (cell-centred: 0.00632
  against 0.1225, the same factor). This is the pair the
  viewer draws side by side (`--ops=2`), and the pointwise error goes from
  smooth across the coarse-fine interfaces to visibly kinked at them.
- `@inbounds` on the RHS kernel, `D = 3`, `N = 16`, 3 roots, 34 blocks,
  one thread: `wave_rhs_kernel!` 0.727 ms without against **0.111 ms**
  with, and the whole `wave_rhs!` 3.41 ms against 2.74 ms -- 6.5x on the
  kernel and 20% on the evaluation, with `du` bit-identical. The gap
  between the two ratios is the ghost fill, which dominates a refined
  mesh. The full argument, and why CI states `check_bounds: 'yes'`, is
  under [What bounds checking costs](#what-bounds-checking-costs).
- Thread scaling on 64 cores (AMD EPYC, 8 NUMA domains; 1792 blocks of
  `128²`, 29.4M points): **12.7× on the RHS path (16.7× with the pages
  interleaved), 3.6× on a whole RK4 step**, 38–48× on the compute-bound
  passes. The gap between the first two numbers is the integrator's serial
  stage arithmetic, and it is 79% of a 64-thread step. Two entries here
  changed and are flagged as such: interleaving the pages *does* help, and
  the claim that it did not came from reading a pair of seconds the wrong
  way round; and the refinement-flag pass now scales 3.4× where 33.9× was
  recorded — measured to be equally true of the *pre-M8 code*, run beside
  this one on the same node in the same job, so it is the machine and not
  this work. The full table is under
  [Multi-threading](#multi-threading). `bin/benchmark.sbatch` reproduces
  the measurement.
- The four loops that were this package's own to thread, at 8 threads on
  a 12-core laptop and 1.8M cells: the Hankel table 0.98× → 6.1×, the
  amplitude scales 1.01× → 4.1×, the coverage reduction → 1.8×, the
  Hankel contraction 0.98× → 1.1×. The last two are memory-bound and
  saturate; the numbers are recorded so that a future "why is this only
  2×" has an answer already measured.
- `RK4(thread = True())`, which threads the stage broadcasts through
  Polyester, measured **2.1× slower** than the default at 8 threads
  (0.030 s → 0.064 s per step): a second thread pool alongside the one
  KernelAbstractions is already using. Reproduced with the two
  measurements in either order.
- Results are **bit-identical** across thread counts — every digit of
  every digest in `test/thread_workload.jl`, at 1 and at 4 threads, on
  both a full pulse run and a full blast run.
- On a device (Apple M3 Pro, Metal, `Float32`), the same three runs reach
  the **same mesh** as the host does, to the last digit of every mesh
  number: the pulse 16 blocks at level 2 with the peak never leaving a
  refined block, the blast wave 136 → 820 blocks (×6.029411764705882) at
  level 2 with 0.9109663409337676 of the ring at the finest level — the
  identical fraction, not a close one, because the criterion's decisions
  are threshold comparisons on per-cell arithmetic that is the same on
  both. The errors agree to about 1e-4 relative (pulse L∞ 0.09687 against
  0.09703, blast 0.029778 against 0.029781), which is the per-block
  summation order inside `volume_weighted_norm` — pairwise on the host,
  sequential in the kernel — and not anything else.
- Per-phase device against host on that machine, at 29.4M points: **5.2×
  on initial data, 2.3× on the exact ring, 1.8× on the refinement
  criterion, 0.6× on the RHS**. And the same code on one **NVIDIA H200**
  against 16 EPYC cores of the same node: **180× on initial data, 129× on
  the exact ring, 3.8× on the refinement criterion, 36× on the RHS**. The
  RHS is the pair worth reading together — the M3 Pro's 0.6× is one
  memory controller shared by CPU and GPU, so there is no bandwidth ratio
  to win; the H200's 36× is what the same kernel does when there is, and
  it matches TreeAMR's own 35.5× for it. Both tables are under
  [Running on a device](#running-on-a-device).
- A whole adaptive run on that device is **slower** at the sizes this
  package's tests use: `track_blast` 1.07 s on eight host threads against
  6.98 s on Metal at 52k points, 1.1 s against 3.6 s at 66k. Launch
  overhead over blocks of 64 points, and recorded because it is the first
  thing anyone will measure.

## Possible extensions

Not planned, listed because they are the obvious next questions:

- Kreiss–Oliger dissipation, to see what it does to interface modes.
- An adaptive integrator, once TreeAMR's `volume_weighted_norm` is wired
  in as `internalnorm`.
- First-order form, as a second application of the same mesh — and the
  one that would reach the half of M8 this package leaves alone. A
  first-order system has fluxes, so it could carry a face-centred field
  set at `G = 0` and make the scheme conservative across coarse-fine
  faces with `InterfaceSchedule` and `restrict_interfaces!`. That is a
  different application rather than a flag on this one: second-order form
  has nothing to restrict, and pretending otherwise is what Scope and
  non-goals refuses.
- A time integrator whose stage arithmetic is a KernelAbstractions kernel
  like everything else, which is what the Amdahl term under
  [Multi-threading](#multi-threading) actually asks for. The cheap
  version of this — `RK4(thread = True())`, which threads the same
  broadcasts through Polyester — is measured there and is *worse*, so the
  extension is a real one: the stage updates have to join the pool that
  is already running, not start a second. On a device the same term is
  the one phase that never leaves the host's control flow, and it is
  measured in [Running on a device](#running-on-a-device) as bandwidth
  bound on both.
- An adaptive run large enough for a device to pay for itself. The phase
  table shows where that is; what stands in the way is not the code but
  the calibration — `σ / h₀` has to be held fixed and the regrid cadence
  with it, which is the same constraint `benchmark_driver` documents. The
  H200 numbers make this the most interesting of these: a whole run there
  should be a win rather than the 6× loss Metal measures, and nothing but
  the mesh size stands between the two.
- A staggered *system*, rather than one field set that happens to be
  vertex-centred everywhere. M8's centering is per dimension, so
  `facecentered` and `edgecentered` layouts exist and are what a
  constrained-transport scheme needs; the wave equation has no use for
  them, which is why nothing here exercises them and why saying so is
  better than inventing a use.
