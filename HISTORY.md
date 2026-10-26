# TreeWave.jl — History

How the package got where it is. [CODE.md](CODE.md) describes the
current design; this file keeps what changed, when and why, the
measurements that later ones replaced, and the dependency history,
ordered by `CODE.md`'s sections.

The milestone numbers here and in `CODE.md` are TreeAMR's, not this
package's. **M5** threads both halves of what TreeAMR owns: every
per-cell kernel and every host-side pass over blocks. **M6** makes the
*storage* decide where the work runs: `backend` is a keyword on
`FieldSet`. **M7** (TreeAMR 0.1.5) is distributed meshes. **M8** is
centerings — `fs.G` per dimension on the field set, and vertex-like
layouts — and, as its other half, conservation at coarse-fine faces
(`InterfaceSchedule`, `restrict_interfaces!`).

**Contents**

- [Goals](#goals)
  - [The one exception to "public API only"](#the-one-exception-to-public-api-only)
  - [The order of the milestones](#the-order-of-the-milestones)
- [Initial conditions](#initial-conditions)
  - [The pulse's sign bug](#the-pulses-sign-bug)
- [Centerings](#centerings)
  - [What the M8 port changed](#what-the-m8-port-changed)
- [The refinement criterion](#the-refinement-criterion)
  - [The blast wave's calibration recipe](#the-blast-waves-calibration-recipe)
- [Multi-threading](#multi-threading)
  - [Threading the application's own loops](#threading-the-applications-own-loops)
  - [Page interleaving, read the wrong way round](#page-interleaving-read-the-wrong-way-round)
  - [The refinement flags' scaling](#the-refinement-flags-scaling)
- [Running on a device](#running-on-a-device)
  - [The diagnostics move to `mesh_mapreduce`](#the-diagnostics-move-to-mesh_mapreduce)
  - [The diagnostics move to `mesh_fold`](#the-diagnostics-move-to-mesh_fold)
  - [M3 Pro timings before M8](#m3-pro-timings-before-m8)
- [Measured results](#measured-results)
  - [The blast wave's L2 rate](#the-blast-waves-l2-rate)
- [Dependencies and the Julia floor](#dependencies-and-the-julia-floor)
  - [The Julia floor](#the-julia-floor)
  - [The TreeAMR bound](#the-treeamr-bound)
  - [From two branch pins to one bound](#from-two-branch-pins-to-one-bound)

## Goals

### The one exception to "public API only"

Nothing here uses anything but TreeAMR's public API
([CODE.md](CODE.md#goals)). There was exactly one exception, and
recording it rather than quietly taking it is what got it fixed:
`TreeAMR.block_partials`, the per-block reduction every diagnostic
upstream is built on, was unexported and ought not to have been.
Upstream agreed, and it is now public as `block_mapreduce` — with a
better signature than the one this package was reaching for — and its
whole-mesh form `mesh_mapreduce` is what the diagnostics here now call.

Using it was the one documented exception to the "public API only" rule,
with a request to make upstream. Upstream took the request and, in taking
it, found three things wrong with the signature this package had been
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

### The order of the milestones

The last three goals in [CODE.md](CODE.md#goals) were reached in order:
the caller's floating-point type first, the caller's storage one
milestone later, and the caller's values — centerings — a milestone later
again. The order of the first two was not arbitrary — a device port on
top of a `Float64`-only application would have been a rewrite rather than
a keyword, and the `Float32` work was where the stray `Float64` literals
were found while they were still cheap to find.

## Initial conditions

### The pulse's sign bug

The load-bearing minus sign on the pulse's `∂ₜu`
([CODE.md](CODE.md#travelling-super-gaussian-pulse-srcsupergaussianjl))
was a real bug in the first draft of this package, introduced when
TreeAMR's closed-form Gaussian derivative was replaced by a call to
`dsupergaussian` and the sign that had been folded into the algebra was
lost. The measured L∞ error at the pulse test's finest mesh was 110
before the fix and 43 after, against 0.09 once the resolution question —
the order `n`, in the same section of `CODE.md` — was also settled — the
sign fix alone was not enough to make the error small, which is why both
had to be found.

## Centerings

### What the M8 port changed

`coordinates(fs, b, idx)` replaced `cell_center` in M8. Since M8, `fs.G`
is an `NTuple{D,Int}`; a schedule is built from the field set and takes
the backend with the rest of the layout; `hostcopy` copies the ghost
width and the centering as well as the forest; and `test/device_tests.jl`
compares the two forms of the refinement criterion on both centerings as
well as at two precisions. [CODE.md](CODE.md#centerings) describes all of
these as they are.

## The refinement criterion

### The blast wave's calibration recipe

The blast wave's calibration — max τ on uniform meshes at `t = 0.4`,
under [Measured results](CODE.md#measured-results) in `CODE.md` — did not
have to be redone for the second layout, but it had to be *re*measured to
say so, and the recipe is recorded there because the earlier entry did
not state it and reproducing it took a second attempt.

## Multi-threading

### Threading the application's own loops

With TreeAMR's M5, the two passes that call code living here — the
initial-data callback given to `fill_by_coordinates!` and the flag
function given to `flag_blocks` — were parallelized by updating the
dependency and changing nothing. That also made the package most of the
way ready for TreeAMR's M6: its initial data already ran inside a kernel
that reaches the geometry through per-block
`block_origins`/`block_spacings` arrays rather than through the tree,
which is the shape a device demands.

The four loops M5 did not reach
([CODE.md](CODE.md#what-was-left-for-the-application-to-thread)) were
measured before they were touched — 0.98×, 1.01× and 0.98× at eight
threads for the three the benchmark already covered, which is the flat
line a serial loop should give and the check that the benchmark reads
what it claims to.

The four loops that were this package's own to thread, at 8 threads on a
12-core laptop and 1.8M cells: the Hankel table 0.98× → 6.1×, the
amplitude scales 1.01× → 4.1×, the coverage reduction → 1.8×, the Hankel
contraction 0.98× → 1.1×. The last two are memory-bound and saturate; the
numbers are recorded so that a future "why is this only 2×" has an answer
already measured.

### Page interleaving, read the wrong way round

An earlier reading of the thread-scaling table
([CODE.md](CODE.md#what-it-buys-and-what-caps-it)) said interleaving the
pages did not help. That earlier entry recorded "the RHS is *slower*
interleaved (0.033 s against 0.047)" and concluded that TreeAMR's 2–6×
did not reproduce downstream. The two seconds in it are almost exactly
what the current sweep measures for the RHS at 64 threads — but they are
the *interleaved* figure first, so they say the opposite of the sentence
around them. The conclusion was drawn from a transposition, and it is
withdrawn. What the sweep shows — smaller than upstream's 2–6× and in the
same direction — is the unremarkable answer the first reading turned
into a puzzle.

### The refinement flags' scaling

The refinement flags are the row of the thread-scaling table that
changed, and the port is not why: where it now has 3.9× at 8 threads,
falling to 3.35 at 64, the entry previously recorded 8.00× and 33.9×.
That was worth attributing rather than overwriting, so it was measured
directly: the pre-M8 package, at its own TreeAMR pin and cell-centred,
was run beside the post-M8 one **in a single job on a single node**, and

| phase | pre-M8, cell | this, vertex |
|---|---|---|
| `refine_flags`, 1 → 16 threads | 2.378 → 0.664 s (3.58×) | 2.388 → 0.697 s (3.43×) |
| `initial_data`, 1 → 16 | 3.071 → 0.191 s (16.1×) | 3.069 → 0.193 s (15.9×) |
| `rhs`, 1 → 16 | 0.582 → 0.0474 s (12.3×) | 0.613 → 0.0533 s (11.5×) |

— so the old figure does not reproduce for the *old code* either, on this
machine today. Whatever changed is the machine or the Julia version, not
M8 and not the stagger. Recorded as a number that moved and is only half
explained, rather than quietly replaced.

## Running on a device

### The diagnostics move to `mesh_mapreduce`

`field_scales`, `blast_coverage` and `track_pulse`'s tracking measure
called `block_mapreduce` and combined its per-block vector here until
TreeAMR 0.1.5, and the part that needed the *tree* — which blocks are
refined, which are at the finest level — indexed that vector by block.
On one process that is the same answer, which is why the host-side
combine went unnoticed until TreeAMR's M7 (0.1.5, distributed meshes)
audited the downstreams and found that it would not stay so;
[CODE.md](CODE.md#what-an-application-has-to-do-that-the-mesh-does-not)
has why. Every replacement is exact, so the numbers did not move. The
fine count in `blast_coverage` used to reuse the hot count's per-block
vector, and is now a third pass.

`block_mapreduce` was the unexported `TreeAMR.block_partials` until
upstream exported it; see
[The one exception to "public API only"](#the-one-exception-to-public-api-only).

### The diagnostics move to `mesh_fold`

Until TreeAMR 0.2.6 a reduction saw one value at a time and nothing else,
so every number a diagnostic needed was a sweep of its own, and the
part that needed the tree entered as a 0/1 per-block `weight` that
scaled a block's value after the fact: `field_scales` was one
`mesh_mapreduce` per variable, `track_pulse`'s tracking measure two (the
whole peak, and the refined peak with `weight = level(key) > 0`), and
`blast_coverage` three (the peak, the hot count, and the fine count with
`weight = Int(level(key) == finest)`). The fine count's weight had to be
an `Int` with an integer `init` — TreeAMR's docstring asked for a
floating-point one, since a weight is usually a volume, and a `0.0`
would have put a `Float64` accumulator in a kernel Metal cannot run.

TreeAMR 0.2.6 (M15, 2026-10-22, designed with TreeGeneralizedHarmonic)
added folds over points, which hand the callback every selected variable
and a per-block tag, so on 2026-10-26 each diagnostic became one fold:
`blast_coverage` keeps a separate sweep for its peak, which the counts
need first, and the weights became `Bool` tags. Every value is a maximum
or an integer count, so nothing moved, and `test/threading_tests.jl`'s
bit-for-bit comparison against the serial loops passed unchanged. The
bound went to `"0.2.6"` with it ([The TreeAMR bound](#the-treeamr-bound)).

### M3 Pro timings before M8

The M3 Pro tables in
[CODE.md](CODE.md#what-it-buys-on-this-hardware-and-what-it-does-not)
are lower across the board than the ones recorded there before M8, on
both layouts and on both backends — the eight-thread host column most of
all (a `step` of 0.244 against 0.348). That is the machine and the Julia
version, not the port: the cell-centred column *is* the pre-M8
configuration, and it moved by the same amount.

## Measured results

### The blast wave's L2 rate

The blast wave's uniform-mesh L2 rate, 1.98 under
[Measured results](CODE.md#measured-results) in `CODE.md`, was recorded
there as 1.99 before: the same 1.985, rounded the other way.

## Dependencies and the Julia floor

### The Julia floor

Erik raised the floor to Julia 1.13 from 1.11 on 2026-10-16 across all
the Tree* packages: supporting the older versions slowed development and
was not useful at the moment. Nothing in the code was specific to 1.11;
the raise removed the second end of the CI matrix, and moved
`bin/benchmark.sbatch` from the cluster's `julia/1.11.2` module to
juliaup's 1.13. Before that, the floor followed `[sources]`, a 1.11 key.
It was 1.11 first while TreeAMR was unregistered: a clean checkout could
not resolve at all without a `[sources]` entry to locate TreeAMR —
`Manifest.toml` is not tracked, and `Project.toml` alone carries only a
UUID — which is also why CI was impossible before it. When TreeAMR was
registered the entry went and the floor dropped to the LTS, 1.10, and it
went back to 1.11 on 2026-09-25 across all the Tree* packages, so that
any of them could depend on an unregistered package through `[sources]`
without touching the floor again; 1.13 keeps that.

### The TreeAMR bound

The `[compat]` bound on TreeAMR ([CODE.md](CODE.md#file-layout)) was
`"0.1.0"` until the diagnostics moved to `mesh_mapreduce`, which 0.1.2
added, and `"0.1.2"` until TreeAMR 0.2.0, whose only break moved the
checkpoint functions out into the companion package TreeIOHDF5; nothing
here writes checkpoints, so the change was the bound alone. It was
`"0.2"` until 2026-10-26, when the diagnostics moved to the folds over
points that 0.2.6 added ([The diagnostics move to
`mesh_fold`](#the-diagnostics-move-to-mesh_fold)). The other floors
(SciMLBase, OrdinaryDiffEqLowOrderRK, SpecialFunctions, and MultiFloats
for the tests) were left where they were, since nothing newer from them
is used; the same update resolved SciMLBase 3.58.0,
OrdinaryDiffEqLowOrderRK 2.2.6, SpecialFunctions 2.9.1 and TreeAMR
0.2.6. KernelAbstractions stayed at 0.9.45: 0.10 was out, but every
TreeAMR release so far admits only `0.9.42-0.9` and `1`.

### From two branch pins to one bound

Before the single `[compat]` bound, the version lived in two `[sources]`
entries that had to be changed together: two `rev = "main"` pins, one in
the root `Project.toml` and one in `bin/Project.toml`, and the standing
hazard was updating the first and leaving the viewer resolving a branch
that no longer existed. Both are gone.
