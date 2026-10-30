#!/usr/bin/env bash
# HTCondor job script (see benchmark.sub): runs all benchmarks of the paper.
# HTCondor copies this script to a scratch directory, so the repository path is passed
# from benchmark.sub (submit the job from the repository directory).
cd "${REPO_DIR:?REPO_DIR not set; submit benchmark.sub from the repository directory}"

# Julia brings all libraries via the Manifest; library paths of the system (e.g. Intel
# oneAPI with its own MKL, OpenMP and zstd) would mix in other library versions.
unset LD_LIBRARY_PATH

# PyAMG runs in a Python environment that CondaPkg.jl installs with pixi. In HTCondor jobs,
# pixi cannot detect the glibc version of the node, so it is passed explicitly.
export CONDA_OVERRIDE_GLIBC="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"

# All solvers run single-threaded (also set in scripts/run_benchmarks.jl).
# The thread settings are stored in the result files (environment_info in src/benchmark.jl).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Optional argument: maximal refinement for a test run (see scripts/run_benchmarks.jl)
julia --project --threads=1 scripts/run_benchmarks.jl "$@"
