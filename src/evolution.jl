# The evolution system: the right-hand side of the wave equation in
# 2nd-order form, and the container the integrator carries for it.
#
#     ∂ₜu = v
#     ∂ₜv = ∇²u                    (wave speed c = 1)
#
# Variable 1 is `u`, variable 2 is `v`. The Laplacian is the standard
# 2nd-order centered 3-point stencil per dimension, so the RHS needs only
# a one-cell stencil — but see `WaveProblem` for why `G = 2` is
# nonetheless the useful choice on a refined mesh.

@kernel function wave_rhs_kernel!(du, @Const(work), @Const(spacings),
                                  ::Val{D}, ::Val{G}) where {D,G}
    I = @index(Global, NTuple)                 # (i1..iD, block)
    b = I[D + 1]
    inner = ntuple(d -> I[d], Val(D))          # state-layout index
    c = ntuple(d -> I[d] + G, Val(D))          # working-array index

    u0 = work[c..., 1, b]
    laplacian = zero(eltype(du))
    for d in 1:D
        up = Base.setindex(c, c[d] + 1, d)
        um = Base.setindex(c, c[d] - 1, d)
        laplacian += work[up..., 1, b] - 2 * u0 + work[um..., 1, b]
    end
    h = spacings[b]

    du[inner..., 1, b] = work[c..., 2, b]
    du[inner..., 2, b] = laplacian / (h * h)
end

"""
Everything the right-hand side needs, built once. The application writes
`f!` itself and calls scatter -> fill_ghosts -> map_blocks explicitly.
"""
struct WaveProblem{T,D,G,F,S}
    fs::F
    schedule::S
    spacings::Vector{T}
    valD::Val{D}
    valG::Val{G}
end

# D and G are carried as Val parameters so the kernel specializes on
# them once, rather than rebuilding them at every RHS evaluation.
function WaveProblem(fs::FieldSet{T,D}, schedule) where {T,D}
    G = fs.forest.G
    return WaveProblem{T,D,G,typeof(fs),typeof(schedule)}(
        fs, schedule, block_spacings(fs.forest, T), Val(D), Val(G))
end

function wave_rhs!(du, u, p, t)
    scatter!(p.fs, u)
    fill_ghosts!(p.fs, p.schedule)
    map_blocks!(wave_rhs_kernel!, p.fs, statearray(du, p.fs), p.fs.work,
                p.spacings, p.valD, p.valG)
    return nothing
end

"""Least-squares convergence rate of `errs` against spacings `hs`."""
function convergence_rate(hs, errs)
    x = log.(hs)
    y = log.(errs)
    n = length(x)
    x̄, ȳ = sum(x) / n, sum(y) / n
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end
