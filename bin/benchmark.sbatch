#!/usr/bin/env bash
#SBATCH --partition=amddebugq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --job-name=treewave-threads
#
# The thread-scaling scan behind CODE.md's multi-threading table: run
# bin/benchmark.jl once per thread count and print the rows. It is a
# plain bash script as well as a batch job -- the #SBATCH lines are
# comments anywhere else -- so the same file serves a laptop and a node:
#
#     TREEWAVE_THREADS="1 2 4 8" bin/benchmark.sbatch                  # here
#     TREEWAVE_NUMA="default interleave" sbatch bin/benchmark.sbatch   # Symmetry
#
# Both page placements are swept in *one* job, on one node, one after the
# other. Two jobs would be quicker but would compare two machines: the
# nodes are nominally identical and measurably are not -- the same
# single-threaded scan came out 20% apart on two of them -- so a 2x
# attributed to placement could be half node. Everything else,
# JULIA_EXCLUSIVE included, is held fixed across the sweep.
#
# Instantiate and precompile *before* submitting, on the node type the
# job will run on -- Julia's cache is keyed by CPU target, so doing it on
# a login node of a different make buys nothing. Measured: 324 s of a
# job's own wall clock when skipped. One interactive allocation, once:
#
#     srun -p amddebugq -c 64 --pty bash -lc \
#         "module load julia/1.11.2 && \
#          julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'"
#
# Do not submit both placements before that has finished: two jobs
# precompiling the same depot at once serialize on its lock files.
set -euo pipefail

# SLURM copies the batch script to a spool directory and runs it there,
# so `dirname $0` is not the repository when this is a job -- submit it
# from the package root and SLURM_SUBMIT_DIR is.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[ -f Project.toml ] || {
    echo "run this from the package root (no Project.toml in $PWD)" >&2
    exit 1
}

# Stop at 64: the AMD nodes are 64 physical cores with no SMT, so a
# higher count is oversubscription rather than more parallelism.
threads=${TREEWAVE_THREADS:-"1 2 4 8 16 32 64"}
args=${TREEWAVE_ARGS:-"--dim=2 --n=128 --roots=32 --reps=3"}
placements=${TREEWAVE_NUMA:-default}

# Nothing here calls BLAS, but Julia's default would start a pool per
# process anyway, and on a no-SMT node that pool competes for the cores
# the kernels are on.
export OPENBLAS_NUM_THREADS=1

launch=(julia)
if [ -n "${SLURM_JOB_ID:-}" ]; then
    [ -n "${MODULESHOME:-}" ] || source /etc/profile.d/modules.sh
    module load julia/1.11.2
    # SLURM's own affinity is switched off, and the threads are then left
    # to the OS. `JULIA_EXCLUSIVE=1` would pin them one per core, but
    # cannot be used for the whole sweep: Julia adds an interactive thread
    # on top of `-t N`, so at the full core count it asks for one more
    # thread than there are cores and exits -- and `-t N,0`, which would
    # be the fix, is rejected by 1.11 ("n and m must be integers >= 1").
    # A sweep whose last point runs under different conditions than the
    # rest is worse than an unpinned one.
    launch=(srun --cpu-bind=none julia)
fi

# Page placement is the single biggest effect on a NUMA node, and it is
# a process-level policy no library can set for itself.
for numa in $placements; do
    prefix=()
    [ "$numa" = interleave ] && prefix=(numactl --interleave=all)

    echo "# numa=$numa args=$args $(hostname)"
    for t in $threads; do
        "${prefix[@]}" "${launch[@]}" -t "$t" --project=. bin/benchmark.jl $args
    done
done
