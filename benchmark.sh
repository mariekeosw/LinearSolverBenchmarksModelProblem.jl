#!/usr/bin/env bash
cd /Home/guests/osewold/Code/LinearSolverBenchmarks.jl
export MKLROOT=/opt/intel/oneapi/mkl/2025.3
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
julia --project -e 'include("scripts/run_model_problem_benchmark.jl"); ModelProblemBenchmarkRun.benchmark("sym_mp_direct_all.jld2", refinement_max = 14, symmetric = true, solver_config=:direct_new)'