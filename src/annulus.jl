# The rotating Bessel mode on TreeMultiPatch's four-patch annulus, the
# exact solution the wave equation on patches is measured against (step 3
# of TreeAMR's M20). On `r0 ≤ r ≤ r1` with solid (Neumann) walls,
#
#     u = R(kr) cos(mθ - kt),   R(z) = J_m(z) Y_m′(kr0) - Y_m(z) J_m′(kr0)
#
# solves the wave equation when `k` is a root of
# `J_m′(kr0) Y_m′(kr1) - J_m′(kr1) Y_m′(kr0)`, so that `R′` vanishes on
# both walls. The pattern turns at the angular speed `k/m`, so it carries
# flux through every glued face of the ring. The annulus's glues are
# identities and its maps join smoothly, `θ` linear through every glue,
# so TreeAMR's copied ghosts are the interpolation there. See "Patches"
# in `CODE.md`.

# `Z_m′(z)` for `Z = J` or `Y`, by `Z_m′ = (Z_{m-1} - Z_{m+1})/2`.
dbesselj(m, z) = (besselj(m - 1, z) - besselj(m + 1, z)) / 2
dbessely(m, z) = (bessely(m - 1, z) - bessely(m + 1, z)) / 2

annulus_cross(m, k, r0, r1) =
    dbesselj(m, k * r0) * dbessely(m, k * r1) - dbesselj(m, k * r1) * dbessely(m, k * r0)

"""
    annulus_wavenumber(m, n, r0, r1)

The `n`-th positive root `k` of `J_m′(kr0) Y_m′(kr1) - J_m′(kr1) Y_m′(kr0)`:
the wavenumber, and the angular frequency, of the Neumann mode of angular
number `m` with radial number `n` on the annulus `r0 ≤ r ≤ r1`. Bracketed
by a scan in steps of a fiftieth of `π/(r1 - r0)`, the roots' spacing far
out, and bisected to the last bit, in `Float64`.
"""
function annulus_wavenumber(m::Integer, n::Integer, r0::Real, r1::Real)
    m >= 0 && n >= 1 || throw(ArgumentError(
        "the mode needs m ≥ 0 and n ≥ 1, got m = $m and n = $n"))
    r0, r1 = Float64(r0), Float64(r1)
    0 < r0 < r1 || throw(ArgumentError("the annulus needs 0 < r0 < r1"))
    f(k) = annulus_cross(m, k, r0, r1)
    step = π / (r1 - r0) / 50
    a = step / 8
    fa = f(a)
    found = 0
    while true
        b = a + step
        fb = f(b)
        if fa == 0 || sign(fa) != sign(fb)
            found += 1
            if found == n
                lo, hi, flo = a, b, fa
                while true
                    mid = (lo + hi) / 2
                    (mid == lo || mid == hi) && return lo
                    fm = f(mid)
                    if fm == 0
                        return mid
                    elseif sign(fm) == sign(flo)
                        lo, flo = mid, fm
                    else
                        hi = mid
                    end
                end
            end
        end
        a, fa = b, fb
    end
end

"""
    AnnulusMode

The rotating mode `u = R(kr) cos(mθ - kt)`, `v = k R(kr) sin(mθ - kt)`, as
a callback `(ξ, p, var) -> value` of the logical position `ξ` of patch
`p`: wrap it in `WithPatch` for `fill_by_coordinates!`, as
[`annulus_exact`](@ref) does. The physical point is the layout's map of
`ξ`, so the mode does not know how the patches are numbered. `R` is
scaled to `R(kr0) = 1`. It evaluates the Bessel functions in `Float64`,
so it runs on the CPU only.
"""
struct AnnulusMode{L}
    layout::L
    m::Int
    k::Float64
    jr0::Float64                 # J_m′(kr0)
    yr0::Float64                 # Y_m′(kr0)
    scale::Float64
    t::Float64
end

function (mode::AnnulusMode)(ξ, p, var)
    X, Y = chart_map(mode.layout, ξ, p)
    r = hypot(Float64(X), Float64(Y))
    θ = atan(Float64(Y), Float64(X))
    kr = mode.k * r
    R = mode.scale * (besselj(mode.m, kr) * mode.yr0 - bessely(mode.m, kr) * mode.jr0)
    phase = mode.m * θ - mode.k * mode.t
    value = var == 1 ? R * cos(phase) : mode.k * R * sin(phase)
    return oftype(ξ[1], value)
end

"""
    annulus_exact(layout::PolarAnnulus, m, k, t)

The rotating Neumann mode of angular number `m` and wavenumber `k` (see
[`annulus_wavenumber`](@ref)) at time `t` on `layout`, as a `WithPatch`
callback for `fill_by_coordinates!`.
"""
function annulus_exact(layout::PolarAnnulus, m::Integer, k::Real, t::Real)
    k = Float64(k)
    r0 = Float64(layout.r0)
    jr0, yr0 = dbesselj(m, k * r0), dbessely(m, k * r0)
    # `J_m Y_m′ - Y_m J_m′ = 2/(πz)`, so `R(kr0) = 2/(π k r0)`.
    scale = π * k * r0 / 2
    return WithPatch(AnnulusMode(layout, Int(m), k, jr0, yr0, scale, Float64(t)))
end

"""
    annulus_forest([T,] layout::PolarAnnulus; N)

TreeAMR's forest of the four-patch annulus `layout`, `N` cells per block
along each dimension, with both walls reflecting: a vertex-centred
reflecting face mirrors the ghosts about the wall plane, `u(r0 - h) =
u(r0 + h)`, which is the Neumann condition `∂_r u = 0` to second order.
Both walls, `r0` and `r1`, are own points of their blocks and evolved
(TreeAMR 0.3, "Closed blocks" in its `CODE.md`).
"""
annulus_forest(::Type{T}, layout::PolarAnnulus; N) where {T} =
    layout_forest(T, layout; N=N, reflecting=((false, false), (true, true)))
annulus_forest(layout::PolarAnnulus{T}; N) where {T} = annulus_forest(T, layout; N=N)

# `u` and `v` are scalars, even in the walls; the angle has no
# reflecting face.
const ANNULUS_PARITY = [(NoParity, EvenParity), (NoParity, EvenParity)]

"""
    annulus_errors([T]; N, roots=(2, 1), r0=1, r1=2, m=1, n=1, G=1,
                   ops=Operators(prolongation=4, restriction=4), cfl=1/4,
                   tend=nothing, alg=RK4())

Evolve the rotating Neumann mode ([`annulus_exact`](@ref)) on the
four-patch annulus `r0 ≤ r ≤ r1` of `roots` roots per patch, `N` cells
per block, to `tend`, by default a quarter turn of its pattern,
`(π/2)m/k`, after which every point of it has crossed a glued face.
Return the `√g`-weighted L2 error of `u` and `v` together, the L∞ error,
the physical radial spacing `h`, the step count and the number of blocks.

The time step is [`patch_timestep`](@ref). `ops` are the ghost
schedule's operators; the mesh has no refinement. (Before TreeAMR 0.3
they also set the order of the derived high wall point, which closed
blocks removed: the high wall is evolved like the low one.)
Vertex-centred: see [`PatchWaveProblem`](@ref).
"""
function annulus_errors(::Type{T}=Float64; N, roots=(2, 1), r0=1, r1=2, m::Integer=1,
                        n::Integer=1, G=1,
                        ops=Operators(prolongation=4, restriction=4),
                        cfl=T(1//4), tend=nothing, alg=RK4()) where {T}
    layout = PolarAnnulus{T}(r0, r1, roots)
    forest = annulus_forest(T, layout; N=N)
    fs = FieldSet(forest, 2; G=G, centering=vertexcentered(2), parity=ANNULUS_PARITY)
    problem = PatchWaveProblem(fs, GhostSchedule(fs, ops), layout; dissipation=0)
    k = annulus_wavenumber(m, n, r0, r1)

    fill_by_coordinates!(annulus_exact(layout, m, k, 0), fs)
    u0 = statevector(fs)
    gather!(u0, fs)

    t_end = tend === nothing ? T(π / 2 * max(m, 1) / k) : T(tend)
    nsteps = ceilint(t_end / patch_timestep(problem; cfl=cfl))
    dt = t_end / nsteps
    prob = ODEProblem(wave_rhs!, u0, (zero(T), t_end), problem)
    sol = solve(prob, alg; dt=dt, adaptive=false, save_everystep=false)

    exact = FieldSet(forest, 2; G=G, centering=vertexcentered(2), parity=ANNULUS_PARITY)
    fill_by_coordinates!(annulus_exact(layout, m, k, t_end), exact)
    uexact = statevector(exact)
    gather!(uexact, exact)
    err = unified(sol.u[end], fs, problem.schedule) .- uexact

    # `√g`-weighted, through the square root of `√g` at each owned point
    # as a state vector of `fs`'s layout: the norm of `w e` over the norm
    # of `w` is the mean of `√g e²` over the mean of `√g`.
    weights = FieldSet(forest, 2; G=G, centering=vertexcentered(2), parity=ANNULUS_PARITY)
    fill_by_coordinates!(WithPatch((ξ, p, v) -> sqrt(volume_element(layout, ξ, p))),
                         weights)
    w = statevector(weights)
    gather!(w, weights)
    l2 = volume_weighted_norm(fs, w .* err) / volume_weighted_norm(fs, w)

    # The physical radial spacing, `r′/N`, the same at every angle.
    h = (T(r1) - T(r0)) / roots[2] * minimum_spacing(forest)
    return (l2=l2, linf=volume_weighted_norm(fs, err; p=Inf), h=h, nsteps=nsteps,
            dt=dt, tend=t_end, nblocks=nleaves(forest), k=k)
end
