#!/usr/bin/env bash
# HTCondor job script (see benchmark.sub): runs all benchmarks of the paper.
cd "$(dirname "$0")"

# Intel oneMKL installation on the WIAS cluster (adapt or remove on other machines)
export MKLROOT=/opt/intel/oneapi/mkl/2025.3
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

# Thread settings are stored in the result files (environment_info in
# src/benchmark.jl). For reproducible runs, set them explicitly,
# e.g. to the number of CPUs requested in benchmark.sub:
# export OMP_NUM_THREADS=12
# export MKL_NUM_THREADS=12

julia --project scripts/run_benchmarks.jl
