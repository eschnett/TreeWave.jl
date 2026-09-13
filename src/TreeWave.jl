"""
    TreeWave

The scalar wave equation in 2nd-order form, as a sample application for
[TreeAMR](https://github.com/eschnett/TreeAMR.jl):

    ∂ₜu = v
    ∂ₜv = ∇²u                    (wave speed c = 1)

TreeAMR supplies the mesh and its operations and deliberately contains no
physics, so this package is where the physics lives — and where the
`scatter!` → `fill_ghosts!` → `map_blocks!` right-hand-side pattern,
the operator-order requirement, and the regrid-and-restart loop are shown
end to end.

Three initial conditions are provided, each measuring something the
others cannot: a standing sine mode (an exact solution, so convergence
order can be measured) in `sinewave.jl`; a travelling super-Gaussian pulse
(a localized feature, so a moving refined region can be exercised) in
`supergaussian.jl`; and a radial blast wave (a feature that loses
amplitude as it spreads, so the refinement criterion's fixed noise floor
is put under real strain) in `blast.jl`. The refinement criterion that
drives the adaptive runs is in `refinement.jl`.

See `CODE.md` in the package root for the design document, and `bin/` for
a CairoMakie viewer.
"""
module TreeWave

using TreeAMR

using KernelAbstractions: @kernel, @index, @Const
using OrdinaryDiffEqLowOrderRK: RK4
using SciMLBase: ODEProblem, solve
using SpecialFunctions: besselj0

# Evolution system
export WaveProblem, wave_rhs!, convergence_rate

# Initial condition: standing sine mode
export wave_omega, wave_exact, wave_forest, wave_errors

# Initial condition: travelling super-Gaussian pulse
export supergaussian, dsupergaussian, pulse_exact, track_pulse, uniform_pulse

# Initial condition: radial blast wave
export blast_initial, blast_reference, blast_radial_table, blast_exact,
       blast_coverage, track_blast, uniform_blast

# Refinement criterion
export lohner, field_scales, cell_indicator, refine_mark, refine_flags,
       refinement_buffer

# Thread-scaling measurement
export benchmark_phases, benchmark_driver

include("precision.jl")
include("evolution.jl")
include("refinement.jl")
include("sinewave.jl")
include("supergaussian.jl")
include("blast.jl")
include("benchmark.jl")

end
