#=
Runs all benchmarks of the paper and writes one .jld2 file per benchmark to results/.

    julia --project scripts/run_benchmarks.jl

The largest problem sizes take several days. For a quick test, pass a maximal refinement;
the results are then written to results_test/:

    julia --project scripts/run_benchmarks.jl 6

To run a single benchmark, use `benchmark` directly (see README).

All solvers run single-threaded, so that they are compared under the same conditions.
=#

# Must be set before AMGCL (OpenMP) and MKLPardiso are loaded
ENV["OMP_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"

using LinearAlgebra: BLAS
using Pardiso: Pardiso
using LinearSolverBenchmarksModelProblem

BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("Start Julia with one thread (julia --threads=1)")

# MKLPardiso has to use MKL_jll from the Manifest. Pardiso.jl switches to a system MKL if
# MKLROOT was set when it was built; then rebuild it with `unset MKLROOT` (see README).
Pardiso.LOCAL_MKL_FOUND && error("Pardiso.jl was built with a system MKL (MKLROOT)")

# (file name, solver configurations, symmetric, maximal refinement, dimensions)
const BENCHMARKS = [
    # Symmetric problem
    ("sym_direct", solver_configs_direct, true, 15, [1, 2, 3]),
    ("sym_direct_pardiso", solver_configs_direct_pardiso, true, 20, [1, 2, 3]),
    ("sym_cg", sym_solver_configs_cg, true, 22, [1, 2, 3]),
    ("sym_cg_ilu", sym_solver_configs_cg_ilu, true, 22, [1, 2, 3]),
    ("sym_cg_amgcl", sym_solver_configs_cg_amgcl, true, 22, [1, 2, 3]),
    ("sym_cg_amg_1d", sym_solver_configs_cg_amg, true, 20, [1]),
    ("sym_cg_amg_23d", sym_solver_configs_cg_amg, true, 22, [2, 3]),
    ("sym_cg_pyamg", sym_solver_configs_cg_pyamg, true, 22, [1, 2, 3]),
    # Nonsymmetric problem
    ("nsym_direct", solver_configs_direct, false, 15, [1, 2, 3]),
    ("nsym_direct_pardiso", solver_configs_direct_pardiso, false, 20, [1, 2, 3]),
    ("nsym_krylov", nsym_solver_configs_krylov, false, 22, [1, 2, 3]),
    ("nsym_krylov_ilu", nsym_solver_configs_krylov_ilu, false, 22, [1, 2, 3]),
    ("nsym_krylov_amgcl_spai0", nsym_solver_configs_krylov_amgcl_spai0, false, 22, [1, 2, 3]),
    ("nsym_krylov_amgcl_ilu0_1d", nsym_solver_configs_krylov_amgcl_ilu0, false, 20, [1]),
    ("nsym_krylov_amgcl_ilu0_23d", nsym_solver_configs_krylov_amgcl_ilu0, false, 22, [2, 3]),
    ("nsym_krylov_amg_1d", nsym_solver_configs_krylov_amg, false, 20, [1]),
    ("nsym_krylov_amg_23d", nsym_solver_configs_krylov_amg, false, 22, [2, 3]),
    ("nsym_krylov_pyamg", nsym_solver_configs_krylov_pyamg, false, 22, [1, 2, 3]),
]

const MAX_REFINEMENT = isempty(ARGS) ? nothing : parse(Int, ARGS[1])
const OUTDIR = isnothing(MAX_REFINEMENT) ? "results" : "results_test"
mkpath(OUTDIR)

for (name, configs, symmetric, refinement_max, dimensions) in BENCHMARKS
    isnothing(MAX_REFINEMENT) || (refinement_max = min(refinement_max, MAX_REFINEMENT))

    # Small warm-up run so that compilation time is not included in the measurements
    benchmark(tempname() * ".jld2", configs; symmetric, dimensions, refinement_max = 0)

    t = @elapsed benchmark(joinpath(OUTDIR, name * ".jld2"), configs; symmetric, dimensions, refinement_max)
    @info "$name took $(round(t; digits = 1)) seconds"
end
