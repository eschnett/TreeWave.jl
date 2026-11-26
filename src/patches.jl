# The wave equation on a forest of patches: the Laplacian in a patch's
# logical coordinates `ξ`, through the metric of TreeMultiPatch's maps,
#
#     ∇²u = (1/√g) ∂_a (A^{ab} ∂_b u),     A^{ab} = S[a] ⋅ S[b] / √g = √g g^{ab},
#
# with `√g` the volume element and `S[a] = √g ∇ξ^a` the area vectors
# (`patch_frame`). TreeAMR's coordinates stay logical, so the mesh sees a
# forest of unit-cube roots and nothing of the map; the metric enters here,
# as one more field set the kernel reads. See "Patches" in `CODE.md`.
#
# Vertex-centred only. The cell-centred comparison the flat box keeps has
# no counterpart on patches: a glued face is a vertex plane, which is what
# TreeAMR's glue and TreeMultiPatch's alignment are written for.

# Where `A^{ab}`, `a ≤ b`, sits among the metric set's variables: after
# `√g`, row by row, `(1,1), (1,2), …, (1,D), (2,2), …`.
@inline metric_index(a, b, ::Val{D}) where {D} =
    2 + (a - 1) * (D + 1) - ((a - 1) * a) ÷ 2 + (b - a)

# `√g`, the `D(D+1)/2` coefficients `A^{ab}`, and the largest logical
# speed of light.
metric_nvars(D) = 2 + D * (D + 1) ÷ 2

# The metric's values at logical `ξ` of patch `p`, in `metric_index`'s
# order, as one tuple.
@inline function metric_values(layout, ξ, p, ::Val{D}) where {D}
    J, S = patch_frame(layout, ξ, p)
    A = ntuple(Val(D * (D + 1) ÷ 2)) do k
        # The `k`-th pair `a ≤ b` of `metric_index`.
        a, r = 1, k
        while r > D - a + 1
            r -= D - a + 1
            a += 1
        end
        b = a + r - 1
        s = S[a][1] * S[b][1]
        for i in 2:D
            s += S[a][i] * S[b][i]
        end
        s / J
    end
    return (J, A..., logical_speed(layout, ξ, p))
end

# Every **stored** point of every block, ghosts included, from `f(ξ, p) ->
# vals`, a tuple of one value per variable. `fill_by_coordinates!` fills
# the owned points only, since the ghosts are the exchange's; but a
# coefficient is known everywhere the map is, and the stencil reads it at
# the ghosts too. Vertex-centred: stored index `idx` is owned point
# `idx - G`, at `origin + (idx - G - 1) h`.
@kernel function stored_fill_kernel!(work, f, @Const(origins), @Const(spacings),
                                     @Const(patches), ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                 # stored index, block
    b = I[D + 1]
    idx = ntuple(d -> I[d], Val(D))
    o = origins[b]
    h = spacings[b]
    ξ = ntuple(d -> o[d] + (idx[d] - G[d] - 1) * h, Val(D))
    vals = f(ξ, patches[b])
    for v in eachindex(vals)
        work[idx..., v, b] = vals[v]
    end
end

"""
    fill_stored!(f, fs, patches)

Set every stored point of every block of the vertex-centred `fs`, ghosts
and the shared high plane included, from `f(ξ, p) -> vals`, where `ξ` is
the point's logical position in patch `p` and `vals` a tuple of
`fs.nvars` values. `patches` is [`block_patches`](@ref) on the field
set's backend. For what is known in closed form wherever the map is
defined, the metric (see [`PatchWaveProblem`](@ref)) and a test's exact
data: the map is defined past each patch's box, so a ghost across a glue
gets the value its own patch's map names there.
"""
function fill_stored!(f, fs::FieldSet{T,D}, patches) where {T,D}
    all(==(1), staggers(fs)) || throw(ArgumentError(
        "fill_stored! places points as vertices, and this field set is not " *
        "vertex-centred"))
    backend = get_backend(fs.work)
    origins = to_backend(backend, block_origins(fs.forest, T))
    spacings = to_backend(backend, block_spacings(fs.forest, T))
    map_blocks!(stored_fill_kernel!, fs, fs.work, f, origins, spacings, patches,
                Val(D), Val(fs.G); stored=true)
    synchronize(backend)
    return fs
end

# The right-hand side on patches. The diagonal terms are the flat box's
# three-point stencil with `A^{aa}` averaged to the half points; the cross
# terms, `a ≠ b`, are centered differences of centered differences, which
# read the edge ghosts `c ± e_a ± e_b`. Both are second order; on the
# annulus the cross terms vanish. With `ε ≠ 0` both variables get
# Kreiss–Oliger dissipation of the scheme's order, `-(ε/16h) Σ_a δ_a⁴ w`,
# which reads two points out and needs `G ≥ 2`.
#
# On a set interpolated across a glue the launch covers the closed range
# and `du` is TreeAMR's `AppendixArray`, which keeps the high glued face
# planes a block owns and drops the rest; `isowned` skips those. On any
# other set `du` is the cores' array and the range the owned one.
# `@inbounds` as in `wave_rhs_kernel!`: the stencil reaches `1 + (ε ≠ 0)`
# points past `c`, at most `N + 1 + G` along a dimension, which `G` puts
# within the stored array.
@kernel function patch_wave_rhs_kernel!(du, @Const(work), @Const(metric),
                                        @Const(spacings), ε, ::Val{D},
                                        ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                 # (i1..iD, block)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))          # state-layout index
    c = ntuple(d -> I[d] + G[d], Val(D))       # working-array index

    if isowned(du, inner..., 1, b)
        @inbounds u0 = work[c..., 1, b]
        flux = zero(eltype(du))
        @inbounds for a in 1:D
            up = Base.setindex(c, c[a] + 1, a)
            um = Base.setindex(c, c[a] - 1, a)
            k = metric_index(a, a, Val(D))
            A0 = metric[c..., k, b]
            Ap = (A0 + metric[up..., k, b]) / 2
            Am = (A0 + metric[um..., k, b]) / 2
            flux += Ap * (work[up..., 1, b] - u0) - Am * (u0 - work[um..., 1, b])
            for e in 1:D
                e == a && continue
                k = metric_index(min(a, e), max(a, e), Val(D))
                δp = work[Base.setindex(up, up[e] + 1, e)..., 1, b] -
                     work[Base.setindex(up, up[e] - 1, e)..., 1, b]
                δm = work[Base.setindex(um, um[e] + 1, e)..., 1, b] -
                     work[Base.setindex(um, um[e] - 1, e)..., 1, b]
                flux += (metric[up..., k, b] * δp - metric[um..., k, b] * δm) / 4
            end
        end
        @inbounds h = spacings[b]
        @inbounds J = metric[c..., 1, b]
        @inbounds v0 = work[c..., 2, b]
        du_u = v0
        du_v = flux / (J * h * h)
        if !iszero(ε)
            σ = ε / (16 * h)
            @inbounds for a in 1:D
                c1 = Base.setindex(c, c[a] + 1, a)
                c2 = Base.setindex(c, c[a] + 2, a)
                m1 = Base.setindex(c, c[a] - 1, a)
                m2 = Base.setindex(c, c[a] - 2, a)
                du_u -= σ * (work[c2..., 1, b] - 4 * work[c1..., 1, b] + 6 * u0 -
                             4 * work[m1..., 1, b] + work[m2..., 1, b])
                du_v -= σ * (work[c2..., 2, b] - 4 * work[c1..., 2, b] + 6 * v0 -
                             4 * work[m1..., 2, b] + work[m2..., 2, b])
            end
        end
        @inbounds du[inner..., 1, b] = du_u
        @inbounds du[inner..., 2, b] = du_v
    end
end

"""
    PatchWaveProblem(fs, schedule, layout; dissipation, boundary=nothing)

Everything the right-hand side on a forest of patches needs, built once:
the state's vertex-centred field set `fs` (`u` and `v`) and its ghost
schedule, the patch system `layout` (a TreeMultiPatch layout, in the
field set's number type, `PolarAnnulus{T}(annulus)`), and the metric it
names, `√g`, `A^{ab} = √g g^{ab}` and the largest logical speed of light,
as a field set of its own on `fs`'s layout. The metric is filled at
every stored point from the maps (`fill_stored!`), ghosts included, since
the stencil reads it there and the maps are defined past each box; it is
not exchanged.

`dissipation` is the Kreiss–Oliger coefficient `ε`, with no default:
`0` for none, as on the annulus, while the seven-patch ball's corners,
where three patches meet, need `ε = 1/2` (TreeAMR's M20). A nonzero `ε`
needs `G ≥ 2`. `boundary` is `nothing` for a forest without outer faces,
or a function of the time returning the `boundary` that `fill_ghosts!`
takes, a `CellBoundary`, for the hook that fills them.

`fs` may be interpolated across a glue (`glue = :interpolate`), as on a
ball: the right-hand side then covers the appendix, the glued high face
planes each patch owns.

[`wave_rhs!`](@ref) takes it as it takes a [`WaveProblem`](@ref), and
[`patch_timestep`](@ref) gives its time step. The layout's maps are
evaluated once, here; a regrid would need a new problem.
"""
struct PatchWaveProblem{T,D,G,F,S,M,V,L,B}
    fs::F
    schedule::S
    metric::M
    spacings::V
    layout::L
    ε::T
    boundary::B
    valD::Val{D}
    valG::Val{G}
end

function PatchWaveProblem(fs::FieldSet{T,D}, schedule, layout; dissipation,
                          boundary=nothing) where {T,D}
    all(==(1), staggers(fs)) || throw(ArgumentError(
        "the wave equation on patches is vertex-centred only: a glued face is a " *
        "vertex plane, and the cell-centred comparison of the flat box has no " *
        "counterpart there"))
    all(>=(1), fs.G) || throw(ArgumentError(
        "the patch Laplacian reads one point past each owned one, and its cross " *
        "terms the edge ghosts, so it needs G ≥ 1, got $(fs.G)"))
    dissipation >= 0 || throw(ArgumentError(
        "the dissipation coefficient must not be negative, which would amplify " *
        "the shortest waves, got $dissipation"))
    iszero(dissipation) || all(>=(2), fs.G) || throw(ArgumentError(
        "the dissipation's fourth differences read two points past each owned " *
        "one, so a nonzero dissipation needs G ≥ 2, got $(fs.G)"))
    backend = get_backend(fs.work)
    # The metric is never exchanged, so its parity is only what the
    # forest's reflecting faces require a set to say; `u`'s does.
    parity = fs.parity === nothing ? nothing : fill(fs.parity[1], metric_nvars(D))
    # Over a forest with a turn (TreeAMR's M21) a set declares how it turns;
    # the metric is never turned either, so the identity.
    nm = metric_nvars(D)
    turns = (fs.rotation === nothing ? (;) : (rotation=Tuple(1:nm),))
    fs.halfturn === nothing || (turns = merge(turns, (halfturn=Tuple(1:nm),)))
    metric = FieldSet(fs.forest, nm; G=fs.G, centering=fs.centering,
                      parity=parity, backend=backend, turns...)
    patches = to_backend(backend, block_patches(fs.forest))
    fill_stored!((ξ, p) -> metric_values(layout, ξ, p, Val(D)), metric, patches)
    spacings = to_backend(backend, block_spacings(fs.forest, T))
    G = fs.G
    return PatchWaveProblem{T,D,G,typeof(fs),typeof(schedule),typeof(metric),
                            typeof(spacings),typeof(layout),typeof(boundary)}(
        fs, schedule, metric, spacings, layout, T(dissipation), boundary, Val(D),
        Val(G))
end

# The right-hand side from ghosts already filled, which a test of the
# operator sets itself. `appendix = true` is ignored on a set without
# one, where `statearray` is the cores' array.
function patch_rhs!(du, p::PatchWaveProblem)
    map_blocks!(patch_wave_rhs_kernel!, p.fs, statearray(du, p.fs; appendix=true),
                p.fs.work, p.metric.work, p.spacings, p.ε, p.valD, p.valG;
                appendix=true)
    return nothing
end

# The ghosts of `p.fs`, with the boundary hook at time `t` if it has one.
patch_fill!(p::PatchWaveProblem, t) =
    p.boundary === nothing ? fill_ghosts!(p.fs, p.schedule) :
    fill_ghosts!(p.fs, p.schedule; boundary=p.boundary(t))

function wave_rhs!(du, u, p::PatchWaveProblem, t)
    scatter!(p.fs, u)
    patch_fill!(p, t)
    patch_rhs!(du, p)
    return nothing
end

"""
    patch_timestep(p::PatchWaveProblem; cfl)

The time step `cfl` times the smallest logical spacing over the largest
logical speed of light, block by block: the flat box's `cfl ⋅ h` in the
coordinates the stencil is written in, where the speed of light is
`√λ_max(g^{ab})` rather than one.
"""
function patch_timestep(p::PatchWaveProblem{T,D}; cfl) where {T,D}
    forest = p.fs.forest
    rate = mesh_mapreduce(identity, max, zero(T), p.metric; vars=metric_nvars(D),
                          weight=key -> inv(spacing(T, forest, key)))
    return T(cfl) / rate
end
