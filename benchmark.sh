#!/usr/bin/env bash
# HTCondor job script (see benchmark.sub): runs all benchmarks of the paper.
# HTCondor copies this script to a scratch directory, so the repository path is passed
# from benchmark.sub (submit the job from the repository directory).
cd "${REPO_DIR:?REPO_DIR not set; submit benchmark.sub from the repository directory}"

# Intel oneMKL installation on the WIAS cluster (adapt or remove on other machines)
export MKLROOT=/opt/intel/oneapi/mkl/2025.3
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

# Thread settings are stored in the result files (environment_info in
# src/benchmark.jl). For reproducible runs, set them explicitly,
# e.g. to the number of CPUs requested in benchmark.sub:
# export OMP_NUM_THREADS=12
# export MKL_NUM_THREADS=12

# Optional argument: maximal refinement for a test run (see scripts/run_benchmarks.jl)
julia --project scripts/run_benchmarks.jl "$@"
