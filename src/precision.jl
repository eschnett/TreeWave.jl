# The operations this package performs that a *software* floating-point type
# does not provide, written so that they need only what every type here has.
#
# The package is generic in its element type `T`, and TreeAMR's mesh is too
# (see "Precision" in `CODE.md`). What is not generic is `Base`: MultiFloats.jl
# implements `floor` and `ceil` returning a float, but no conversion to
# `Integer` and no `rem`, so `ceil(Int, x)` and `mod(x, y)` -- both of which
# close through those -- are `MethodError`s at `Float64x2` while working at
# `Float32` and `Float64`. Neither is a physics decision, so neither belongs
# spelled out at the call site.
#
# Nothing here is exported. They are spellings, not concepts.

"""
    wrap(x, L)

`x` reduced into `[0, L)` for positive `L` — what `mod(x, L)` means on a
periodic box.

Spelled `x - L·floor(x/L)` rather than `mod`, because `Base.mod` on floats
goes through `rem`, which MultiFloats.jl does not define. The two agree
wherever `L > 0`, which is the only case a box produces; at `L = 1`, which is
every run here, both are exact and the results are bit-identical.
"""
wrap(x, L) = x - L * floor(x / L)

"""
    ceilint(x)
    floorint(x)

`ceil(Int, x)` and `floor(Int, x)`, for a type that may not define
`Int(::AbstractFloat)`.

Both `ceil(Int, x)` and `floor(Int, x)` close through a conversion to
`Integer` that MultiFloats.jl does not provide — and neither does a detour
through `Float64`, since `Float64(::Float32x2)` is not defined either. What
*every* `AbstractFloat` in Julia converts to is `BigFloat`, so that is the
fallback, taken only once the value is already an exact integer and therefore
only ever exact. A hardware float never reaches it.

The fallback allocates, which is why these are confined to what they are used
for: a count of chunks, of substeps, of buffer cells — host-side control flow,
evaluated a handful of times per run — and the blast wave's radial-table
subscript, which is a lookup into a `Float64` quadrature and not part of the
evolution at all.
"""
ceilint(x) = _toint(ceil(x))
floorint(x) = _toint(floor(x))

# `y` is an exact integer value by construction, so both branches are exact.
_toint(y::Base.IEEEFloat) = Int(y)
_toint(y) = Int(BigFloat(y))

"""
    tofloat64(x)

`x` as a `Float64` — the bridge to the blast wave's reference quadrature,
which is a `Float64` table whatever the run's type is (see
[`blast_reference`](@ref)).

`Float64(x)` is not universal either: MultiFloats.jl defines a conversion
only to its own *limb* type, so `Float64(::Float32x2)` is a `MethodError`
while `Float32(::Float32x2)` is not. `BigFloat` is again the common currency.
Host-side, and a handful of times per run — once per reference and once per
error measurement — never per cell.
"""
tofloat64(x::Base.IEEEFloat) = Float64(x)
tofloat64(x) = Float64(BigFloat(x))
