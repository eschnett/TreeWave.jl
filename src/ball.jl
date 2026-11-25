# The wave equation on TreeMultiPatch's seven-patch ball (step 5 of
# TreeAMR's M20): a plane wave crossing every glue, with its own values as
# Dirichlet data on the outer sphere, and the measurements the step asks
# for — convergence, the energy over many crossing times, and the
# symmetry under point inversion and under a mirror. The cube and the six
# shells meet at kinks, so the ghosts across a glue are TreeAMR's
# interpolated ones (`glue = :interpolate`), and every glued face point is
# evolved by each patch holding it, its high glued face planes in the
# state's appendix. See "The seven-patch ball" in `CODE.md`.

"""
    PlaneWave(k, φ)

The plane wave `u = sin(k ⋅ x - |k| t + φ)` of wave vector `k`, a
`D`-tuple, and phase `φ`: a solution of the wave equation in physical
coordinates, the same expression in every patch. `isbits`.
"""
struct PlaneWave{T,D}
    k::NTuple{D,T}
    ω::T
    φ::T
end

PlaneWave(k, φ) = (T = float(promote_type(eltype(k), typeof(φ)));
                   PlaneWave{T,length(k)}(T.(Tuple(k)), T(sqrt(sum(abs2, k))), T(φ)))

# `(u, v)` of a plane wave at the physical point `x` and time `t`.
@inline function plane_wave(w::PlaneWave{T,D}, x, t) where {T,D}
    θ = w.φ - w.ω * t
    for d in 1:D
        θ += w.k[d] * x[d]
    end
    s, c = sincos(θ)
    return (s, -w.ω * c)
end

# A plane wave at logical `ξ` of patch `p`, through the layout's map: the
# state for `fill_by_coordinates!` and the Dirichlet data for the hook.
struct PlaneWaveOn{L,W,T}
    layout::L
    wave::W
    t::T
end

@inline (s::PlaneWaveOn)(ξ, p) =
    oftype.(ξ[1], plane_wave(s.wave, chart_map(s.layout, ξ, p), s.t))

"""
    ball_exact(layout, wave::PlaneWave, t)

`u` and `v` of the plane wave `wave` at time `t` on the patches of
`layout`, as a `WithPatch(AllVariables(…))` callback for
`fill_by_coordinates!`.
"""
ball_exact(layout, wave, t) =
    WithPatch(AllVariables(PlaneWaveOn(layout, wave, t)))

# The boundary hook of the plane wave at time `t`: the wave itself beyond
# the outer sphere.
struct PlaneWaveHook{L,W}
    layout::L
    wave::W
end

(h::PlaneWaveHook)(t) = CellBoundary(WithPatch(AllVariables(
    (ξ, p, δ) -> PlaneWaveOn(h.layout, h.wave, oftype(ξ[1], t))(ξ, p))))

"""
    SymmetricWave(wave::PlaneWave, mirrors, turn)

The sum of `wave`'s images under the group a reduced ball's symmetry
generates (TreeAMR's M21): the mirrors `x_a → -x_a` of `mirrors` and the
`turn` quarter turns about `z` (1 a quarter turn, 2 a half turn, 0 none),
each image a plane wave of the turned wave vector. An exact solution
with that symmetry, so a reduced ball's run of it is the full ball's,
restricted. `isbits`.
"""
struct SymmetricWave{T,M}
    k::NTuple{M,NTuple{3,T}}
    ω::T
    φ::T
end

function SymmetricWave(w::PlaneWave{T,3}, mirrors::NTuple{3,Bool}, turn::Integer) where {T}
    ks = NTuple{3,T}[]
    for flips in Iterators.product(ntuple(a -> mirrors[a] ? (false, true) : (false,), 3)...),
        r in 0:(turn == 1 ? 3 : turn == 2 ? 1 : 0)
        k = ntuple(a -> flips[a] ? -w.k[a] : w.k[a], 3)
        for _ in 1:(turn == 2 ? 2r : r)
            k = (-k[2], k[1], k[3])
        end
        push!(ks, k)
    end
    return SymmetricWave{T,length(ks)}(Tuple(ks), w.ω, w.φ)
end

@inline function plane_wave(w::SymmetricWave{T,M}, x, t) where {T,M}
    u, v = zero(T), zero(T)
    for i in 1:M
        θ = w.φ - w.ω * t + w.k[i][1] * x[1] + w.k[i][2] * x[2] + w.k[i][3] * x[3]
        s, c = sincos(θ)
        u += s
        v -= w.ω * c
    end
    return (u, v)
end

# Homogeneous Dirichlet data, `u = v = 0` beyond the outer sphere, at any
# time: the closed ball, whose energy the scheme can only lose.
zero_hook(t) = CellBoundary(AllVariables((x, δ) -> (zero(x[1]), zero(x[1]))))

"""
    ball_problem([T]; N, roots=(1, 1), Lc=1, R1=3, G=2, glue=4,
                 dissipation=1/2, boundary)

The wave equation on the seven-patch ball `SevenPatchBall(Lc, R1, roots)`,
`N` cells per block, interpolated across every glue at order `glue`, with
ghost width `G` and Kreiss–Oliger `dissipation`: a [`PatchWaveProblem`](@ref)
and its layout. `boundary(layout)` gives the hook's function of the
time: `layout -> PlaneWaveHook(layout, wave)` for a plane wave's
Dirichlet data, `_ -> zero_hook` for the closed ball.

The interpolation order is the scheme's to say: a second derivative of
order 2 needs ghosts exact to degree 3, `glue = 4`, and a 4-point stencil
needs `G ≥ 2` (TreeAMR's `k ≤ 2(G + 1)`), as the dissipation does.
"""
function ball_problem(::Type{T}=Float64; N, roots=(1, 1), Lc=1, R1=3, G=2, glue=4,
                      dissipation=T(1//2), boundary, symmetry=nothing,
                      refine=nothing) where {T}
    layout = SevenPatchBall(T(Lc), T(R1), roots; symmetry=symmetry)
    forest = layout_forest(T, layout; N=N)
    refine === nothing || refine_about_origin!(forest, layout, T(refine))
    # On a reduced ball (M21) `u` and `v` are scalars: even under every
    # mirror, and carried onto themselves by a turn.
    kw = symmetry === nothing ? (;) :
         (; (any(layout.mirrors) ? (parity=[ntuple(_ -> EvenParity, 3) for _ in 1:2],) :
             (;))...,
          (layout.turn == 1 ? (rotation=(1, 2),) :
           layout.turn == 2 ? (halfturn=(1, 2),) : (;))...)
    fs = FieldSet(forest, 2; G=G, centering=vertexcentered(3), glue=:interpolate, kw...)
    ops = Operators(prolongation=4, restriction=4, glue=glue)
    schedule = GhostSchedule(fs, ops; atlas=layout)
    return PatchWaveProblem(fs, schedule, layout; dissipation=dissipation,
                            boundary=boundary(layout)),
           layout
end

# Refine, once, every leaf of the cube whose box lies within `|x|∞ ≤ r` of
# the origin, and balance: the refinement a black hole at the centre wants,
# away from the cube's faces, whose kinked glues keep their levels (M20),
# and across a reduced ball's seams, which regrid freely (M21).
function refine_about_origin!(forest, layout, r)
    marked = filter(forest.leaves) do k
        patch(forest, k) == 1 || return false
        ext = block_extent(forest, k)
        corners = Iterators.product(ntuple(d -> (ext[d][1], ext[d][2]), 3)...)
        return all(ξ -> maximum(abs, chart_map(layout, ξ, 1)) <= r + 1e-12, corners)
    end
    isempty(marked) || refine!(forest, marked)
    balance!(forest)
    return forest
end

# Evolve `u0` under `p` from `t0` to `t1` with fixed-step RK4 at `cfl`,
# landing exactly on `t1`; return the state and the step count.
function patch_evolve(p::PatchWaveProblem{T}, u0, t0, t1; cfl, alg=RK4(),
                      nsteps=max(1, ceilint((t1 - t0) / patch_timestep(p; cfl=cfl)))) where {T}
    dt = (t1 - t0) / nsteps
    prob = ODEProblem(wave_rhs!, u0, (T(t0), T(t1)), p)
    sol = solve(prob, alg; dt=dt, adaptive=false, save_everystep=false)
    return sol.u[end], nsteps
end

# The state vector of `fs` filled by `f` at its owned points, its
# appendix included.
function filled_state(f, fs)
    fill_by_coordinates!(f, fs)
    u = statevector(fs)
    gather!(u, fs)
    return u
end

"""
    ball_errors([T]; N, roots=(1, 1), Lc=1, R1=3, k=(1.1, -0.7, 0.5), φ=0.3,
                tend=1, cfl=1/4, dissipation=1/2, glue=4, near=2)

Evolve the plane wave ([`PlaneWave`](@ref)) on the seven-patch ball of
[`ball_problem`](@ref) to `tend`, with the wave as Dirichlet data on the
outer sphere. Return the `√g`-weighted L2 error of `u` and `v` together,
each physical point counted once (TreeAMR's norm), the L∞ error over
every owned point, the L∞ error within `near` spacings of a glued face,
the cube's physical spacing `h`, the step count and the number of
blocks.
"""
function ball_errors(::Type{T}=Float64; N, roots=(1, 1), Lc=1, R1=3,
                     k=(1.1, -0.7, 0.5), φ=0.3, tend=1, cfl=T(1//4),
                     dissipation=T(1//2), glue=4, near=2) where {T}
    wave = PlaneWave(T.(k), T(φ))
    p, layout = ball_problem(T; N=N, roots=roots, Lc=Lc, R1=R1, glue=glue,
                             dissipation=dissipation,
                             boundary=l -> PlaneWaveHook(l, wave))
    fs = p.fs
    u0 = filled_state(ball_exact(layout, wave, zero(T)), fs)
    u, nsteps = patch_evolve(p, u0, zero(T), T(tend); cfl=cfl)
    err = u .- filled_state(ball_exact(layout, wave, T(tend)), fs)

    w = filled_state(WithPatch(AllVariables(
        (ξ, q) -> (s = sqrt(volume_element(layout, ξ, q)); (s, s)))), fs)
    l2 = volume_weighted_norm(fs, w .* err) / volume_weighted_norm(fs, w)
    linf = volume_weighted_norm(fs, err; p=Inf)
    linf_near = mapreduce(max, owned_points(fs, err); init=zero(T)) do (q, I, vals)
        M = patch_points(layout, q, N)
        glued(d, hi) = !(hi && q >= 2 && d == radial_dimension(layout, q))
        isnear = any(d -> (I[d] <= near && glued(d, false)) ||
                          (I[d] >= M[d] - near && glued(d, true)), 1:3)
        isnear ? maximum(abs, vals) : zero(T)
    end
    h = 2 * T(Lc) / (roots[1] * N)
    return (l2=l2, linf=linf, linf_near=linf_near, h=h, nsteps=nsteps,
            nblocks=nleaves(fs.forest))
end

# The number of cells of patch `q` along each dimension at `N` per block.
patch_points(layout, q, N) = layout_patches(layout)[q].roots .* N

"""
    owned_points(fs, u) -> Vector

Every point the state `u` of the field set `fs` holds, its appendix
included, as `(patch, I, values)`: `I` the point's integer position in its
patch, in spacings of its block from the patch's low corner, and `values`
its variables. A point on a glued face appears once for each patch
holding it. On the CPU and serial: for measurements, not for a run's
loop.
"""
function owned_points(fs::FieldSet{T,D}, u) where {T,D}
    forest = fs.forest
    a = statearray(u, fs; appendix=true)
    pts = Tuple{Int,NTuple{D,Int},NTuple{2,T}}[]
    for b in 1:nblocks(fs)
        key = blockkey(fs, b)
        q = patch(forest, key)
        o = block_origin(T, forest, key)
        h = spacing(T, forest, key)
        for i in CartesianIndices(ntuple(_ -> forest.N + 1, D))
            isowned(a, Tuple(i)..., 1, b) || continue
            I = ntuple(d -> round(Int, o[d] / h) + i[d] - 1, D)
            push!(pts, (q, I, (a[Tuple(i)..., 1, b], a[Tuple(i)..., 2, b])))
        end
    end
    return pts
end

"""
    patch_quadrature(f, p::PatchWaveProblem)

`Σ w f(work, metric, c, b, h)` over every point of every patch's closed
box, once per patch holding it: the trapezoidal rule of each patch in its
logical coordinates, weight `h^D` halved along each dimension where the
point is on a face of its patch. A point on a glued face is so split
between its patches, and the outer sphere's plane, which the hook fills,
is included. `f` reads the filled stored arrays of the state and the
metric at stored index `c` of block `b`. The quadrature of `√g` is the
physical volume. Host and serial, after a fill: for measurements.
"""
function patch_quadrature(f, p::PatchWaveProblem{T,D}) where {T,D}
    fs = p.fs
    forest = fs.forest
    N = forest.N
    work = Array(fs.work)
    metric = Array(p.metric.work)
    total = zero(T)
    for b in 1:nblocks(fs)
        key = blockkey(fs, b)
        q = patch(forest, key)
        o = block_origin(T, forest, key)
        h = spacing(T, forest, key)
        n = layout_patches(p.layout)[q].roots
        lo = ntuple(d -> round(Int, o[d] / h), D)
        M = ntuple(d -> round(Int, n[d] / h), D)
        top = ntuple(d -> lo[d] + N == M[d], D)
        for i in CartesianIndices(ntuple(d -> N + top[d], D))
            w = h^D
            for d in 1:D
                I = lo[d] + i[d] - 1
                (I == 0 || I == M[d]) && (w /= 2)
            end
            c = ntuple(d -> i[d] + fs.G[d], D)
            total += w * f(work, metric, c, b, h)
        end
    end
    return total
end

"""
    patch_energy(p::PatchWaveProblem, u, t)

The energy `½ ∫ (v² + g^{ab} ∂_a u ∂_b u) √g dξ` of the state `u` at time
`t`, by [`patch_quadrature`](@ref), with centered first differences:
`½ Σ w (√g v² + A^{ab} D_a u D_b u)`. Fills `p.fs` from `u`, ghosts and
hook included.
"""
function patch_energy(p::PatchWaveProblem{T,D}, u, t) where {T,D}
    scatter!(p.fs, u)
    patch_fill!(p, t)
    return patch_quadrature(p) do work, metric, c, b, h
        Du = ntuple(Val(D)) do a
            (work[Base.setindex(c, c[a] + 1, a)..., 1, b] -
             work[Base.setindex(c, c[a] - 1, a)..., 1, b]) / (2h)
        end
        e = metric[c..., 1, b] * work[c..., 2, b]^2
        for a in 1:D, e2 in 1:D
            e += metric[c..., metric_index(min(a, e2), max(a, e2), Val(D)), b] *
                 Du[a] * Du[e2]
        end
        e / 2
    end
end

"""
    patch_volume(p::PatchWaveProblem)

The physical volume of `p`'s patches, the [`patch_quadrature`](@ref) of
`√g`: the scheme's own quadrature, which splits each shared face point
between its patches.
"""
patch_volume(p::PatchWaveProblem) =
    patch_quadrature((work, metric, c, b, h) -> metric[c..., 1, b], p)

"""
    ball_energies([T]; N, tend, every=1, roots=(1, 1), dissipation=1/2,
                  cfl=1/4, seed=1)

The energy of random data (uniform in `[-1, 1]` at every point the state
holds, from a seeded `Xoshiro`) on the closed seven-patch ball of
[`ball_problem`](@ref), `u = v = 0` beyond the outer sphere, every
`every` time units to `tend`: the times and the energies.
"""
function ball_energies(::Type{T}=Float64; N, tend, every=1, roots=(1, 1),
                       dissipation=T(1//2), cfl=T(1//4), seed=1, symmetry=nothing,
                       refine=nothing) where {T}
    p, _ = ball_problem(T; N=N, roots=roots, dissipation=dissipation,
                        boundary=_ -> zero_hook, symmetry=symmetry, refine=refine)
    u = statevector(p.fs)
    u .= 2 .* rand(Xoshiro(seed), T, length(u)) .- 1
    ts, Es = T[0], T[patch_energy(p, u, zero(T))]
    t = zero(T)
    while t < tend
        u, _ = patch_evolve(p, u, t, t + every; cfl=cfl)
        t += every
        push!(ts, t)
        push!(Es, patch_energy(p, u, t))
    end
    return ts, Es
end

# The image of point `I` of patch `q` of the seven-patch ball under the
# point inversion `x → -x` (`:inversion`) or the mirror `x₁ → -x₁`
# (`:mirror`), at `M` cells per patch along each dimension. The cube's
# indices along a reversed axis turn round; a shell over the cube's face
# `(k, s)` goes to the shell over `(k, -s)` when the symmetry reverses
# `x_k`, with its radius kept, and its tangential axes, each along an
# `x_j`, turn round where `x_j` is reversed.
function ball_image(layout, symmetry, q, I, N)
    flips = symmetry === :inversion ? (true, true, true) : (true, false, false)
    M = patch_points(layout, q, N)
    if q == 1
        return q, ntuple(d -> flips[d] ? M[d] - I[d] : I[d], 3)
    end
    k = radial_dimension(layout, q)
    q′ = flips[k] ? (iseven(q) ? q + 1 : q - 1) : q
    return q′, ntuple(d -> d != k && flips[d] ? M[d] - I[d] : I[d], 3)
end

"""
    ball_asymmetry([T]; N, symmetry, tend, roots=(1, 1), dissipation=1/2,
                   cfl=1/4, seed=1)

Evolve random data `A` and its image `B` under `symmetry`, `:inversion`
(`x → -x`) or `:mirror` (`x₁ → -x₁`), on the closed seven-patch ball to
`tend`, and return the largest difference between `B` and the image of
`A` at the end, over every point the state holds, and the largest value
of `A` there for scale. Each copy of a glued face point is compared with
the copy its image is, on the image's patch.
"""
function ball_asymmetry(::Type{T}=Float64; N, symmetry, tend, roots=(1, 1),
                        dissipation=T(1//2), cfl=T(1//4), seed=1) where {T}
    symmetry in (:inversion, :mirror) || throw(ArgumentError(
        "the seven-patch ball's symmetries measured here are :inversion and " *
        ":mirror, got $(repr(symmetry))"))
    p, layout = ball_problem(T; N=N, roots=roots, dissipation=dissipation,
                             boundary=_ -> zero_hook)
    fs = p.fs
    uA = statevector(fs)
    uA .= 2 .* rand(Xoshiro(seed), T, length(uA)) .- 1
    A0 = Dict((q, I) => v for (q, I, v) in owned_points(fs, uA))
    uB = statevector(fs)
    aB = statearray(uB, fs; appendix=true)
    for b in 1:nblocks(fs)
        key = blockkey(fs, b)
        q = patch(fs.forest, key)
        o = block_origin(T, fs.forest, key)
        h = spacing(T, fs.forest, key)
        for i in CartesianIndices(ntuple(_ -> N + 1, 3))
            isowned(aB, Tuple(i)..., 1, b) || continue
            I = ntuple(d -> round(Int, o[d] / h) + i[d] - 1, 3)
            v = A0[ball_image(layout, symmetry, q, I, N)]
            aB[Tuple(i)..., 1, b] = v[1]
            aB[Tuple(i)..., 2, b] = v[2]
        end
    end
    uA, _ = patch_evolve(p, uA, zero(T), T(tend); cfl=cfl)
    uB, _ = patch_evolve(p, uB, zero(T), T(tend); cfl=cfl)
    A = Dict((q, I) => v for (q, I, v) in owned_points(fs, uA))
    worst, scale = zero(T), zero(T)
    for (q, I, v) in owned_points(fs, uB)
        w = A[ball_image(layout, symmetry, q, I, N)]
        worst = max(worst, abs(v[1] - w[1]), abs(v[2] - w[2]))
        scale = max(scale, abs(w[1]), abs(w[2]))
    end
    return (difference=worst, scale=scale)
end

"""
    ball_symmetric_difference([T]; N, symmetry, tend, roots=(2, 1), refine=nothing,
                              k=(1.1, -0.7, 0.5), φ=0.3, dissipation=1/2, cfl=1/4)

The run of the seven-patch ball reduced by `symmetry` (TreeMultiPatch's
`SymmetricBall`, TreeAMR's M21) against the full ball's, from the sum of
the plane wave's images under the symmetry's group ([`SymmetricWave`](@ref)),
an exact solution with that symmetry, with its own Dirichlet data, to
`tend`: the largest difference over every point the reduced ball holds
and the full one's at the same position, and the largest value. Both
balls refined `refine` about the origin when given. The step count of
both is the same, the time step being the cube's.
"""
function ball_symmetric_difference(::Type{T}=Float64; N, symmetry, tend, roots=(2, 1),
                                   refine=nothing, k=(1.1, -0.7, 0.5), φ=0.3,
                                   dissipation=T(1//2), cfl=T(1//4)) where {T}
    reduced = SevenPatchBall(one(T), T(3), roots; symmetry=symmetry)
    wave = SymmetricWave(PlaneWave(T.(k), T(φ)), reduced.mirrors, reduced.turn)
    # Both take the full ball's steps: the reduced ball's own step could
    # differ, its fastest point being elsewhere, and then so would the runs.
    run(sym, steps) = begin
        p, layout = ball_problem(T; N=N, roots=roots, dissipation=dissipation,
                                 boundary=l -> PlaneWaveHook(l, wave), symmetry=sym,
                                 refine=refine)
        u0 = filled_state(ball_exact(layout, wave, zero(T)), p.fs)
        steps = something(steps, max(1, ceilint(T(tend) / patch_timestep(p; cfl=cfl))))
        u, _ = patch_evolve(p, u0, zero(T), T(tend); cfl=cfl, nsteps=steps)
        p, layout, u, steps
    end
    pf, lf, uf, nsteps = run(nothing, nothing)
    pr, lr, ur, _ = run(symmetry, nsteps)
    # Every point keyed by its full patch and its logical position there,
    # exact for the dyadic positions of a block's points.
    function positions(fs, u, full_of)
        forest = fs.forest
        a = statearray(u, fs; appendix=true)
        out = Dict{Tuple{Int,NTuple{3,T}},NTuple{2,T}}()
        for b in 1:nblocks(fs)
            key = blockkey(fs, b)
            q = patch(forest, key)
            P, off = full_of(q)
            o, h = block_origin(T, forest, key), spacing(T, forest, key)
            for i in CartesianIndices(ntuple(_ -> forest.N + 1, 3))
                isowned(a, Tuple(i)..., 1, b) || continue
                ξ = ntuple(d -> o[d] + (i[d] - 1) * h + off[d], 3)
                out[(P, ξ)] = (a[Tuple(i)..., 1, b], a[Tuple(i)..., 2, b])
            end
        end
        return out
    end
    red = positions(pr.fs, ur, q -> (Int(lr.patch[q]), T.(lr.offset[q])))
    full = positions(pf.fs, uf, q -> (q, ntuple(_ -> zero(T), 3)))
    difference = maximum(((key, v),) -> maximum(abs, v .- full[key]), red)
    scale = maximum(v -> maximum(abs, v), values(full))
    return (difference=difference, scale=scale, npoints=length(red))
end

