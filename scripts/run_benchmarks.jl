#=
Runs the benchmarks of the paper and writes one .jld2 file per benchmark.

    julia --project scripts/run_benchmarks.jl [job] [max_refinement] [unscaled|scaled]

`job` selects the benchmarks: "all" (default), a job (last column of BENCHMARKS:
"sym_pardiso", "sym_direct", "sym" (CG, ILU), "sym_amg", "sym_pyamg" and the same for "nsym") or a single benchmark by
its file name, e.g. "sym_cg_ilu". benchmark.sub runs the jobs in parallel.

The matrix variant (see src/matrices.jl) is "unscaled" (default, L + δI, results in results/)
or "scaled" (L + δh²I, results in results_scaled/). For the scaled matrix, 1D is restricted to
refinement MAX_REFINEMENT_1D_SCALED, since all solvers lose accuracy for larger 1D problems.

For a quick test, pass a maximal refinement; the results are then written to results_test/
(or results_test_scaled/), which are overwritten:

    julia --project scripts/run_benchmarks.jl all 6 scaled

Results of a full run are never overwritten; move or delete existing files first.
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

# (file name, solver configurations, symmetric, maximal refinement, dimensions, job)
const BENCHMARKS = [
    # Symmetric problem
    ("sym_direct", solver_configs_direct, true, 15, [1, 2, 3], "sym_direct"),
    # MKLPardiso: in 3D, refinement 19 needs more than 250 GB (results end at refinement 18)
    ("sym_direct_pardiso", solver_configs_direct_pardiso, true, 19, [1, 2, 3], "sym_pardiso"),
    ("sym_cg", sym_solver_configs_cg, true, 22, [1, 2, 3], "sym"),
    ("sym_cg_ilu", sym_solver_configs_cg_ilu, true, 22, [1, 2, 3], "sym"),
    ("sym_cg_amgcl", sym_solver_configs_cg_amgcl, true, 22, [1, 2, 3], "sym_amg"),
    ("sym_cg_amg_1d", sym_solver_configs_cg_amg, true, 20, [1], "sym_amg"),
    ("sym_cg_amg_23d", sym_solver_configs_cg_amg, true, 22, [2, 3], "sym_amg"),
    ("sym_cg_pyamg", sym_solver_configs_cg_pyamg, true, 22, [1, 2, 3], "sym_pyamg"),
    # Nonsymmetric problem
    ("nsym_direct", solver_configs_direct, false, 15, [1, 2, 3], "nsym_direct"),
    ("nsym_direct_pardiso", solver_configs_direct_pardiso, false, 19, [1, 2, 3], "nsym_pardiso"),
    ("nsym_krylov", nsym_solver_configs_krylov, false, 22, [1, 2, 3], "nsym"),
    ("nsym_krylov_ilu", nsym_solver_configs_krylov_ilu, false, 22, [1, 2, 3], "nsym"),
    ("nsym_krylov_amgcl_spai0", nsym_solver_configs_krylov_amgcl_spai0, false, 22, [1, 2, 3], "nsym_amg"),
    ("nsym_krylov_amgcl_ilu0_1d", nsym_solver_configs_krylov_amgcl_ilu0, false, 20, [1], "nsym_amg"),
    ("nsym_krylov_amgcl_ilu0_23d", nsym_solver_configs_krylov_amgcl_ilu0, false, 22, [2, 3], "nsym_amg"),
    ("nsym_krylov_amg_1d", nsym_solver_configs_krylov_amg, false, 20, [1], "nsym_amg"),
    ("nsym_krylov_amg_23d", nsym_solver_configs_krylov_amg, false, 22, [2, 3], "nsym_amg"),
    ("nsym_krylov_pyamg", nsym_solver_configs_krylov_pyamg, false, 22, [1, 2, 3], "nsym_pyamg"),
]

# Larger refinements of a solver configuration are skipped after a solve exceeds this time (s)
const TIME_LIMIT = 600.0
# Scaled matrix in 1D: κ ≈ 400 n², so the relative residual of any solver is limited to about
# ε_mach * 400 n² (≈ 1e-4 at refinement 12, N ≈ 4e4)
const MAX_REFINEMENT_1D_SCALED = 12

# Arguments in any order: job, maximal refinement (integer), matrix variant
const SCALED = "scaled" in ARGS
const ARGS_INT = filter(a -> !isnothing(tryparse(Int, a)), ARGS)
const ARGS_JOB = filter(a -> isnothing(tryparse(Int, a)) && !(a in ("scaled", "unscaled")), ARGS)
const JOB = isempty(ARGS_JOB) ? "all" : only(ARGS_JOB)
const MAX_REFINEMENT = isempty(ARGS_INT) ? nothing : parse(Int, only(ARGS_INT))
const OUTDIR = (isnothing(MAX_REFINEMENT) ? "results" : "results_test") * (SCALED ? "_scaled" : "")
JOB == "all" || JOB in last.(BENCHMARKS) || JOB in first.(BENCHMARKS) || error("Unknown job or benchmark $JOB")
@info "Job $JOB, $(SCALED ? "scaled" : "unscaled") matrix, results in $OUTDIR"
mkpath(OUTDIR)

const SELECTED = filter(b -> JOB == "all" || JOB == b[6] || JOB == b[1], BENCHMARKS)

# Do not overwrite results of a full run (test runs are overwritten)
if isnothing(MAX_REFINEMENT)
    existing = filter(isfile, [joinpath(OUTDIR, b[1] * ".jld2") for b in SELECTED])
    isempty(existing) || error("Result files exist, move or delete them first: $existing")
end

for (name, configs, symmetric, refinement_max, dimensions, job) in SELECTED
    isnothing(MAX_REFINEMENT) || (refinement_max = min(refinement_max, MAX_REFINEMENT))

    # Small warm-up run so that compilation time is not included in the measurements
    benchmark(tempname() * ".jld2", configs; symmetric, dimensions, refinement_max = 0, scaled = SCALED)

    refinement_max_1d = SCALED ? min(refinement_max, MAX_REFINEMENT_1D_SCALED) : refinement_max
    t = @elapsed benchmark(joinpath(OUTDIR, name * ".jld2"), configs; symmetric, dimensions, refinement_max, refinement_max_1d, scaled = SCALED, time_limit = TIME_LIMIT)
    @info "$name took $(round(t; digits = 1)) seconds"
end
