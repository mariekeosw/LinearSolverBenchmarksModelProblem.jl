# LinearSolverBenchmarksModelProblem.jl

Benchmarks of sparse direct solvers and preconditioned Krylov methods in Julia for symmetric and nonsymmetric elliptic model problems in one, two and three spatial dimensions.

This repository contains the code, the benchmark results and the post-processing scripts for the JuliaCon proceedings paper *Benchmarking Iterative vs. Direct Linear Solvers for PDEs in Julia* (M. Osewold, J. Fuhrmann, P. Jaap). It is the model-problem part of [`LinearSolverBenchmarks.jl`](https://github.com/mariekeosw/LinearSolverBenchmarks.jl), which additionally contains MOSFET drift-diffusion simulations.

## Model problem and solvers

The matrices have the structure of a cell-centered finite volume discretization of $-\Delta u + \delta u = f$ on $(0,1)^d$ with homogeneous Neumann boundary conditions, $\delta = 10^{-2}$, $n$ cells per direction and mesh size $h = 1/n$ (`src/matrices.jl`). Two variants are available:

- **unscaled** (default, results in `results/`, figures in `figures/`): $A_\mathrm{sym} = L_{d,n} + \delta I$ (`fdmatrix_sym`) and $A_\mathrm{nsym} = L_{d,n} D + \delta I$ with a random diagonal $D$, $d_i \in \{0.8, 0.801, \dots, 1.2\}$, fixed seed (`fdmatrix_nsym`), as in [`benchmarks/sparselu.jl`](https://github.com/SciML/LinearSolve.jl/blob/main/benchmarks/sparselu.jl) of LinearSolve.jl. The matrices are strictly diagonally dominant by the fixed margin $\delta$, and the condition number of $A_\mathrm{sym}$ is bounded by $(4d + \delta)/\delta$ independently of $N$.
- **scaled** (`scaled = true`, results in `results_scaled/`): $L_{d,n} + \delta h^2 I$ and $L_{d,n} D + \delta h^2 I$. Up to the factor $h^{d-2}$, $A_\mathrm{sym}$ is the finite volume matrix, and its condition number grows like $4d/(\delta h^2)$. In 1D, every solver can then only reach a relative residual of about $\varepsilon_\mathrm{mach} \cdot 400\, n^2$, so 1D is restricted to refinement 12 ($N \approx 4 \cdot 10^4$).

All solvers are accessed through [`LinearSolve.jl`](https://github.com/SciML/LinearSolve.jl) (`src/solvers.jl`):

| Category | Methods and implementations |
| --- | --- |
| Direct solvers | UMFPACK and KLU from SuiteSparse, MKLPardiso through [`Pardiso.jl`](https://github.com/JuliaSparse/Pardiso.jl), [`Sparspak.jl`](https://github.com/PetrKryslUCSD/Sparspak.jl) |
| Krylov methods | CG (symmetric), BiCGSTAB and GMRES (nonsymmetric) from [`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl) |
| Incomplete factorizations | ILU(0) from [`ILUZero.jl`](https://github.com/mcovalt/ILUZero.jl), threshold ILU (drop tolerances 0.01 and 0.1) from [`IncompleteLU.jl`](https://github.com/haampie/IncompleteLU.jl) |
| Algebraic multigrid | Ruge–Stüben and smoothed aggregation from [`AMGCLWrap.jl`](https://github.com/j-fu/AMGCLWrap.jl) (with SPAI(0) or ILU(0) relaxation), [`AlgebraicMultigrid.jl`](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) and [`LinearSolvePyAMG.jl`](https://github.com/SciML/LinearSolve.jl/tree/main/lib/LinearSolvePyAMG) |

All solvers run single-threaded (`scripts/run_benchmarks.jl` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and the BLAS threads to 1). The linear solver runtime (setup + solve) is measured with [`BenchmarkTools.jl`](https://github.com/JuliaCI/BenchmarkTools.jl) (minimum over at most 10 samples within 0.5 s; all solvers are compiled in a small warm-up run beforehand, so no extra warm-up evaluation is needed). The number of Krylov iterations, the return code, the relative residual and the mean of the solution are taken from the solution of the last timed sample (`src/benchmark.jl`). All Krylov methods stop when $\lVert b - Au_k \rVert_2 \le 10^{-8} \lVert b \rVert_2$, evaluated in a callback with the residual that Krylov.jl updates in every iteration; Krylov.jl's own criterion, which uses the preconditioned norm for preconditioned CG, is switched off (`euclidean_stopping` in `src/benchmark.jl`). The maximal number of Krylov iterations is $10^5$. If a solve takes longer than 10 minutes, the larger problem sizes of this solver configuration and dimension are skipped (`TIME_LIMIT` in `scripts/run_benchmarks.jl`).

## Structure

```
src/
  LinearSolverBenchmarksModelProblem.jl   package module
  matrices.jl                             model matrices
  solvers.jl                              solver configurations
  benchmark.jl                            run_single, benchmark
scripts/run_benchmarks.jl                 all benchmarks of the paper
postprocess/figures.jl                    plot_runtimes, plot_iterations
postprocess/tables.jl                     LaTeX tables
results/                                  benchmark results (.jld2), created by scripts/run_benchmarks.jl
results_scaled/                           benchmark results for the scaled matrix L + δh²I
results_archive/                          earlier results with Krylov.jl's default stopping criterion
figures/                                  figures
benchmark.sh, benchmark.sub               job script and HTCondor submit file
```

## Usage

The results in the paper were computed with Julia 1.13.3 and the package versions recorded in `Manifest.toml`.

```bash
git clone https://github.com/mariekeosw/LinearSolverBenchmarksModelProblem.jl
cd LinearSolverBenchmarksModelProblem.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

MKLPardiso uses the MKL recorded in the Manifest (`MKL_jll`). Pardiso.jl switches to a system MKL if the environment variable `MKLROOT` is set when it is built, e.g. on machines with Intel oneAPI. In that case, rebuild it once without `MKLROOT`:

```bash
unset MKLROOT LD_LIBRARY_PATH
julia --project -e 'using Pkg; Pkg.build("Pardiso")'
```

### Single solve

```julia
using LinearSolverBenchmarksModelProblem
run_single(sym_solver_configs_cg_ilu[3].alg; symmetric = true, dimension = 2, refinement = 5) # CG + ILU(0)
```

returns `(problem_size, runtime, niters, retcode, residual, testval)`. The target number of unknowns is `10 * 2^refinement`.

### Benchmark

```julia
benchmark("sym_cg_ilu_example.jld2", sym_solver_configs_cg_ilu; symmetric = true, dimensions = [1, 2, 3], refinement_max = 10)
```

The solver configurations are defined in `src/solvers.jl`. `julia --project scripts/run_benchmarks.jl` runs all benchmarks of the paper and writes the results to `results/`; the result files are updated after every solve. `julia --project scripts/run_benchmarks.jl sym_amg` runs only one job (`sym_pardiso`, `sym_direct`, `sym` (CG, ILU), `sym_amg`, `sym_pyamg` and the same for `nsym`) or a single benchmark by its file name (e.g. `sym_cg_ilu`). `julia --project scripts/run_benchmarks.jl all 6` runs all benchmarks up to refinement 6 and writes to `results_test/`, as a quick test. The argument `scaled` selects the scaled matrix (`results_scaled/`, `results_test_scaled/`), e.g. `julia --project scripts/run_benchmarks.jl sym_amg scaled`. Existing result files of a full run are not overwritten.

On the HTCondor cluster, submit from the repository directory: `condor_submit benchmark.sub` runs the jobs selected in its queue block in parallel on the same node (full run), `condor_submit benchmark.sub -append "max_refinement = 6"` runs a test, and `-append "variant = scaled"` selects the scaled matrix.

A result file contains `labels`, `dimensions`, `refinement`, `problem_sizes`, `tlinsolves` (runtimes), `niters`, `retcodes`, `residuals`, `tests`, `scaled` (matrix scaling) and `environment` (Julia version, CPU and thread settings). Results are dictionaries with keys `(dimension, label)`.

### Figures

```julia
using Plots
include("postprocess/figures.jl")
```

`Figures.plot_runtimes` and `Figures.plot_iterations` take any number of result files and the keyword arguments `algorithms` (selection and order, which defines the colors), `active_dimensions`, `references` (reference lines `factor * N^exponent` as `(exponent, factor)` pairs), `ylims`, `xlims`, `save_figure` and `title_figure`. With the same `algorithms` list, both figures use the same colors.

Figures of the paper (file names as used in the paper):

```julia
r(name) = joinpath("results", name * ".jld2")

Figures.plot_runtimes(r("sym_direct"); Plotter = Plots,
    algorithms = ["UMFPACK", "KLU", "MKLPardiso", "Sparspak"],
    save_figure = true, title_figure = "sym_mp_direct.pdf")

Figures.plot_runtimes(r("sym_direct_pardiso"), r("sym_cg_ilu"); Plotter = Plots,
    algorithms = ["MKLPardiso", "CG-ILU-drop-0.01", "CG-ILU-drop-0.1", "CG-ILU0"],
    save_figure = true, title_figure = "sym_mp_cg_ilu.pdf")

Figures.plot_runtimes(r("sym_direct_pardiso"), r("sym_cg_amgcl"); Plotter = Plots,
    algorithms = ["MKLPardiso", "CG-AMGCL-RS-SPAI0", "CG-AMGCL-SA-SPAI0", "CG-AMGCL-RS-ILU0", "CG-AMGCL-SA-ILU0"],
    save_figure = true, title_figure = "sym_mp_cg_amgcl.pdf")

Figures.plot_runtimes(r("sym_direct_pardiso"), r("sym_cg_amg_1d"), r("sym_cg_amg_23d"); Plotter = Plots,
           algorithms = ["MKLPardiso", "CG-AMG-RS", "CG-AMG-SA"],
           save_figure = true, title_figure = "sym_mp_cg_amg.pdf")

Figures.plot_runtimes(r("sym_direct_pardiso"), r("sym_cg_pyamg"); Plotter = Plots,
           algorithms = ["MKLPardiso", "CG-PyAMG-RS", "CG-PyAMG-SA"],
           save_figure = true, title_figure = "sym_mp_cg_pyamg.pdf")
```

The iteration figures use the same calls with `Figures.plot_iterations`.
```julia
Figures.plot_iterations(r("sym_cg"), r("sym_cg_ilu"); Plotter = Plots,
    algorithms = ["CG", "CG-ILU-drop-0.01", "CG-ILU-drop-0.1", "CG-ILU0"],
    save_figure = true, title_figure = "sym_mp_cg_ilu_iters.pdf")

Figures.plot_iterations(r("sym_cg"), r("sym_cg_amgcl"); Plotter = Plots,
           algorithms = ["CG", "CG-AMGCL-RS-SPAI0", "CG-AMGCL-SA-SPAI0", "CG-AMGCL-RS-ILU0", "CG-AMGCL-SA-ILU0"],
           save_figure = true, title_figure = "sym_mp_cg_amgcl_iters.pdf")

Figures.plot_iterations(r("sym_cg"), r("sym_cg_amg_1d"), r("sym_cg_amg_23d"); Plotter = Plots,
           algorithms = ["CG", "CG-AMG-RS", "CG-AMG-SA"],
           save_figure = true, title_figure = "sym_mp_cg_amg_iters.pdf")

Figures.plot_iterations(r("sym_cg"), r("sym_cg_pyamg"); Plotter = Plots,
           algorithms = ["CG", "CG-PyAMG-RS", "CG-PyAMG-SA"],
           save_figure = true, title_figure = "sym_mp_cg_pyamg_iters.pdf")
```








### Tables

```julia
include("postprocess/tables.jl")
Tables.export_table("results/sym_direct.jld2", "sym_direct.tex")
```

### Tests

```julia
julia> ]
pkg> test
```

## Citation and license

Please cite this software as described in `CITATION.cff`. MIT license, see `LICENSE`.
