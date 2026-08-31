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

Two initial conditions are provided, each measuring something different:
a standing sine mode (an exact solution, so convergence order can be
measured) in `sinewave.jl`, and a travelling super-Gaussian pulse (a
localized feature, so a moving refined region can be exercised) in
`supergaussian.jl`.

See `CODE.md` in the package root for the design document, and `bin/` for
a CairoMakie viewer.
"""
module TreeWave

using TreeAMR

using KernelAbstractions: @kernel, @index, @Const
using OrdinaryDiffEqLowOrderRK: RK4
using SciMLBase: ODEProblem, solve

# Evolution system
export WaveProblem, wave_rhs!, convergence_rate

# Initial condition: standing sine mode
export wave_omega, wave_exact, wave_forest, wave_errors

# Initial condition: travelling super-Gaussian pulse
export supergaussian, dsupergaussian, pulse_exact, track_pulse, uniform_pulse

include("evolution.jl")
include("sinewave.jl")
include("supergaussian.jl")

end
