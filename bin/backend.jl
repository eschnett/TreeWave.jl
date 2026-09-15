# `--backend=` for the scripts in this directory.
#
# Shared by all three, because all three take the same flag and there is
# exactly one right way to resolve it. The device package is loaded only
# when asked for: neither CUDA nor Metal is a dependency of TreeWave or of
# TreeAMR, and the two environments here -- `bin/Project.toml` for the
# viewers, the package environment for the benchmark -- should not grow one
# just so that `--backend=cpu` keeps working. So the load is a `Core.eval`
# into `Main` rather than a `using` at the top of a script, and a missing
# package is reported as the one command that fixes it.

using KernelAbstractions: Backend, CPU, supports_float64

const BACKENDNAMES = Dict("cuda" => (:CUDA, :CUDABackend),
                          "metal" => (:Metal, :MetalBackend))

"""
    resolvebackend(name) -> Backend

The KernelAbstractions backend `name` asks for: `"cpu"`, `"cuda"` or
`"metal"`.

`Core.eval` rather than `@eval` at the call site because the constructor
has to be called in the world the `using` created; evaluating both in
`Main` in order is the whole of what that needs.
"""
function resolvebackend(name::AbstractString)
    name == "cpu" && return CPU()
    haskey(BACKENDNAMES, name) ||
        error("unknown --backend=$name; expected cpu, cuda or metal")
    pkg, ctor = BACKENDNAMES[name]
    Base.identify_package(String(pkg)) === nothing && error(
        "--backend=$name needs $pkg in this environment, which is deliberately \
         not a dependency of TreeWave. Run the script against a project that \
         has it:\n\n    julia --project=/tmp/twgpu -e 'using Pkg; \
         Pkg.develop(path=\".\"); Pkg.add(\"$pkg\")'\n")
    Core.eval(Main, :(using $pkg))
    Core.eval(Main, :($pkg.functional())) || error(
        "$pkg loaded but reports no functional device, so --backend=$name has \
         nothing to run on")
    return Core.eval(Main, :($ctor()))
end

"""
    withbackend(f, name, T) -> f(backend)

Resolve `--backend=name`, check `--type=T` against it, and run `f` on the
resulting backend -- in the **latest** world, which is the whole reason
this is a function taking a callback rather than two lines in `main`.

Loading a device package adds methods, and a script's `main` is already
running by then, so **every** dispatch on the new backend type resolves
in the world age `main` was called in and falls through to whatever
generic method KernelAbstractions defines. Both halves of that were
measured here:

- `allocate` falls through to the generic method, which throws a
  `MethodError` naming a method the same message lists as a candidate --
  which is the tell;
- `supports_float64` falls through to the generic `true`, so
  [`checkprecision`](@ref) below silently passed a `Float64` run to a
  device that has no fp64, and the objection arrived from the mesh
  instead.

Hence the `invokelatest` covering the check *and* the work, rather than
only the work.
"""
function withbackend(f, name::AbstractString, ::Type{T}) where {T}
    backend = resolvebackend(name)
    return Base.invokelatest() do
        checkprecision(T, backend)
        f(backend)
    end
end

"""
    checkprecision(T, backend)

Refuse `Float64` on a backend without hardware fp64 *here*, in terms of
the flags that caused it.

TreeAMR raises the same objection, with a better explanation, when the
field set is built -- but it can only talk about a field set, and what
the caller actually did was combine two command-line flags that do not
go together. This says so before any work starts.
"""
function checkprecision(::Type{T}, backend::Backend) where {T}
    T === Float64 && !supports_float64(backend) && error(
        "$(nameof(typeof(backend))) has no hardware Float64, so --type=f64 \
         cannot run on it; pass --type=f32")
    return nothing
end
