# The viewers' command line, kept out of the scripts so that what is left
# there is the run.
#
# Shared by `visualize.jl` and `visualize2d.jl`. Nothing in here changes
# how a run is computed: it says which run, in which type, on which layout
# and device, and where its figure goes, and it turns `--ops=` into the
# two things a run needs from it, the operators and the ghost width.
# `--case=` and `--n=` belong to the 1D viewer and `--frozen-scales` to the
# 2D one; each script says which it takes.

# `--type=` selects the floating-point type the run is computed in. Only
# the two hardware types are offered: a MultiFloat is a fine thing to *run*
# (see "Precision" in CODE.md) but Makie cannot plot one, so a viewer flag
# for it would be an invitation to a confusing failure.
const FLOATTYPES = Dict("f32" => Float32, "f64" => Float64)

# `--centering=` selects where the values sit. Vertex is the default, as
# it is throughout the package; `cell` renders the comparison.
const CENTERINGS = Dict("vertex" => vertexcentered, "cell" => cellcentered)

"""
    viewer_options(args; D, cases=nothing, order=false, frozen_scales=false)
        -> NamedTuple

The viewer's command line `args` as the inputs of a run and the place its
figure goes. `cases` lists the cases a `--case=` may name, and leaves the
flag out when `nothing`; `order` admits `--n=`, the pulse's super-Gaussian
order, and `frozen_scales` admits `--frozen-scales`. The blast wave takes
no `--n=`: its exact solution exists only for the ordinary Gaussian.

The returned `G` and `ops` are what `--ops=` means for a run: order-`p`
operators, and the ghost width they need, which is not a free choice and
is not the same on both layouts — order-`p` prolongation reaches `p/2`
planes past a cell-centred interface and `p/2 - 1` planes past a vertex
dimension's shared plane, and the Laplacian needs one either way. See
"Operator order" in `CODE.md`.

`suffix` keeps the default spelling's filename for the default run, so a
`Float32` or cell-centred render never overwrites the figure CI checks.
"""
function viewer_options(args; D, cases=nothing, order=false,
                        frozen_scales=false)
    case = "both"
    out = joinpath(@__DIR__, "output")
    n = 1
    ops_order = 4
    T, typetag = Float64, ""
    makecentering, centeringtag = vertexcentered, ""
    backend = "cpu"
    refresh_scales = true
    # Sixel is for a human looking at a terminal; a pipe gets the paths.
    inline = stdout isa Base.TTY
    for a in args
        flag, value = occursin('=', a) ? split(a, '='; limit=2) : (a, "")
        if flag == "--case" && cases !== nothing
            value in (cases..., "both") ||
                error("--case must be $(join(cases, ", ")) or both; got $value")
            case = value
        elseif flag == "--out"
            out = value
        elseif flag == "--n" && order
            n = parse(Int, value)
        elseif flag == "--ops"
            ops_order = parse(Int, value)
        elseif flag == "--type"
            haskey(FLOATTYPES, value) ||
                error("--type must be f32 or f64; got $value")
            T = FLOATTYPES[value]
            typetag = value == "f64" ? "" : "_$value"
        elseif flag == "--centering"
            haskey(CENTERINGS, value) ||
                error("--centering must be vertex or cell; got $value")
            makecentering = CENTERINGS[value]
            centeringtag = value == "vertex" ? "" : "_$value"
        elseif flag == "--backend"
            backend = value
        elseif a == "--frozen-scales" && frozen_scales
            refresh_scales = false
        elseif a == "--display"
            inline = true
        elseif a == "--no-display"
            inline = false
        else
            error("unknown argument $a; expected " *
                  join([cases === nothing ? [] : ["--case="];
                        order ? ["--n="] : [];
                        ["--out=", "--ops=", "--type=", "--centering=",
                         "--backend="];
                        frozen_scales ? ["--frozen-scales"] : [];
                        ["--display", "--no-display"]], ", "))
        end
    end

    centering = makecentering(D)
    vertex = all(==(:vertex), centering)
    G = max(1, ops_order ÷ 2 - (vertex ? 1 : 0))
    ops = Operators(prolongation=ops_order, restriction=ops_order)
    return (; T, centering, G, ops, ops_order, n, backend, refresh_scales,
            cases=cases === nothing ? () : case == "both" ? cases : (case,),
            out, inline, suffix=typetag * centeringtag)
end

centeringname(centering) = all(==(:vertex), centering) ? "vertex-centred" :
                           "cell-centred"
