#!/usr/bin/env bash
# HTCondor job script (see benchmark.sub): runs the model-problem benchmarks of the paper.
cd "$(dirname "$0")"

# Intel oneMKL installation on the WIAS cluster (adapt or remove on other machines)
export MKLROOT=/opt/intel/oneapi/mkl/2025.3
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

# Thread settings are stored in the result files (environment_info in
# scripts/run_model_problem_benchmark.jl). For reproducible runs, set them explicitly,
# e.g. to the number of CPUs requested in benchmark.sub:
# export OMP_NUM_THREADS=12
# export MKL_NUM_THREADS=12

# Repeated runtime measurements with BenchmarkTools.jl (runtimes and iteration counts).
# Use benchmark_strategy = :elapsed for single runs with setup/solve times and residuals.
julia --project -e 'include("scripts/run_model_problem_benchmark.jl"); ModelProblemBenchmarkRun.several_benchmarks(benchmark_strategy = :belapsed)'
