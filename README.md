# LinearSolverBenchmarksModelProblem.jl

Benchmarks of sparse direct solvers and preconditioned Krylov methods in Julia for symmetric and nonsymmetric elliptic model problems in one, two and three spatial dimensions.

This repository contains the code, the benchmark results and the post-processing scripts for the JuliaCon proceedings paper *Benchmarking Iterative vs. Direct Linear Solvers for PDEs in Julia* (M. Osewold, J. Fuhrmann, P. Jaap). It is the model-problem part of [`LinearSolverBenchmarks.jl`](https://github.com/mariekeosw/LinearSolverBenchmarks.jl), which additionally contains MOSFET drift-diffusion simulations.

## Overview

The model matrices have the nearest-neighbor structure of a low-order discretization of $-\Delta u + \delta u = f$ on $(0,1)^d$ with homogeneous Neumann boundary conditions and $\delta = 10^{-2}$:

- symmetric: $A_\mathrm{sym} = L_{d,n} + \delta I$,
- nonsymmetric: $A_\mathrm{nsym} = L_{d,n} D + \delta I$ with a random diagonal $D$, $d_i \in \{0.8, 0.801, \dots, 1.2\}$ (fixed seed).

All solvers are accessed through [`LinearSolve.jl`](https://github.com/SciML/LinearSolve.jl):

| Category | Methods and implementations |
| --- | --- |
| Direct solvers | UMFPACK and KLU from SuiteSparse, MKLPardiso through [`Pardiso.jl`](https://github.com/JuliaSparse/Pardiso.jl), [`Sparspak.jl`](https://github.com/PetrKryslUCSD/Sparspak.jl) |
| Krylov methods | CG (symmetric), BiCGSTAB, CGS and GMRES (nonsymmetric) from [`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl) |
| Incomplete factorizations | ILU(0) from [`ILUZero.jl`](https://github.com/mcovalt/ILUZero.jl), threshold ILU (drop tolerances 0.01 and 0.1) from [`IncompleteLU.jl`](https://github.com/haampie/IncompleteLU.jl) |
| Algebraic multigrid | Ruge–Stüben and smoothed aggregation from [`AMGCLWrap.jl`](https://github.com/j-fu/AMGCLWrap.jl) (with SPAI(0) or ILU(0) relaxation), [`AlgebraicMultigrid.jl`](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) and [`LinearSolvePyAMG.jl`](https://github.com/SciML/LinearSolve.jl/tree/main/lib/LinearSolvePyAMG) |

For each solver configuration, dimension and problem size, the benchmark records the linear solver runtime (setup + solve), the number of Krylov iterations and the return code. Runs with `benchmark_strategy = :elapsed` additionally record the setup and solve times separately, the relative residual and the mean of the solution as a consistency check.

**Note on CGS:** only unpreconditioned CGS is benchmarked. In `LinearSolve.jl` v5.18.2, `KrylovJL(KrylovAlg = Krylov.cgs!, precs = ...)` does not pass the preconditioner to `Krylov.cgs!`. The preconditioned CGS entries in the existing result files in `jld2_files_server/nsym_model_problem/` were therefore computed without preconditioning.

## Directory structure

- **`src/`**: package code; `src/model_problem/benchmark.jl` defines the solver configurations.
- **`scripts/run_model_problem_benchmark.jl`**: matrix construction, single solves (`main`), benchmark loops (`benchmark`, `several_benchmarks`) and storage of the results.
- **`post_process/figures/model_problem/`**: `fig1_runtimes.jl` (runtimes) and `fig2_iterations.jl` (iteration counts).
- **`post_process/tables/model_problem/`**: detailed LaTeX tables for `@elapsed` and `@belapsed` results.
- **`jld2_files_server/`**: benchmark results used in the paper, for the symmetric and nonsymmetric problem.
- **`benchmark.sh`, `benchmark.sub`**: job script and HTCondor submit file for the cluster runs.

## Usage

You need Julia 1.10 or later. The Manifest records the package versions used for the results in the paper (Julia 1.12).

### 1. Install

```bash
git clone https://github.com/mariekeosw/LinearSolverBenchmarksModelProblem.jl
cd LinearSolverBenchmarksModelProblem.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### 2. Solve a single system

```julia
include("scripts/run_model_problem_benchmark.jl")

problem_size, testval, residual, tlinsolve, tlinsolve_setup, tlinsolve_solve, niters, retcode =
    ModelProblemBenchmarkRun.main(
        symmetric = true,
        dimension = 2,
        refinement = 5,
        strategy = ModelProblemBenchmarkRun.ModelProblem.sym_solver_configs_cg_ilu[3].alg, # CG + ILU(0)
        benchmark_strategy = :elapsed,
    )
```

The target number of unknowns is `10 * 2^refinement`.

### 3. Run a benchmark

```julia
ModelProblemBenchmarkRun.benchmark(
    "sym_mp_cg_ilu_example.jld2";
    dimensions = [1, 2, 3],
    refinement_max = 10,
    symmetric = true,
    solver_config = :sym_cg_ilu,
    benchmark_strategy = :belapsed,
)
```

The available solver groups (`solver_config`) are listed in `benchmark()` in `scripts/run_model_problem_benchmark.jl` and defined in `src/model_problem/benchmark.jl`. `ModelProblemBenchmarkRun.several_benchmarks(benchmark_strategy = :belapsed)` runs all benchmarks of the paper (this takes several days for the largest problem sizes).

Each `.jld2` file contains `labels`, `dimensions`, `refinement`, `problem_sizes`, `tlinsolves`, `tlinsolve_setups`, `tlinsolve_solves`, `walltimes`, `residuals`, `tests`, `niters`, `retcodes` and `environment` (Julia version, CPU and thread settings). Results are dictionaries with keys `(dimension, label)`. Files in `jld2_files_server/` were created before iteration counts were recorded and do not contain `niters`, `retcodes` and `environment`.

### 4. Figures

The figure scripts need [`Plots.jl`](https://github.com/JuliaPlots/Plots.jl) in your global environment and are loaded with [`Revise.jl`](https://github.com/timholy/Revise.jl):

```julia
using Revise, Plots
includet("post_process/figures/model_problem/fig1_runtimes.jl")
includet("post_process/figures/model_problem/fig2_iterations.jl")
```

Both functions accept any number of `.jld2` files and the keyword arguments `algorithms` (selection and order, which defines the colors), `active_dimensions`, `references` (reference lines `factor * N^exponent` as `(exponent, factor)` pairs), `ylims`, `xlims`, `save_figure` and `title_figure`. Using the same `algorithms` list in both functions gives the same colors in the runtime and the iteration figure.

Figures of the paper:

```julia
sym = "jld2_files_server/sym_model_problem/"
nsym = "jld2_files_server/nsym_model_problem/"

# Sparse direct solvers
Fig1Runtimes.plot_runtimes(sym * "sym_mp_direct_15_belapsed.jld2"; Plotter = Plots,
    algorithms = ["UMFPACK", "KLU", "MKLPardiso", "Sparspak"],
    save_figure = true, title_figure = "sym_mp_direct.pdf")

# MKLPardiso and CG with ILU preconditioning
Fig1Runtimes.plot_runtimes(sym * "sym_mp_direct_pardiso_20_belapsed.jld2", sym * "sym_mp_cg_ilu_22_belapsed.jld2"; Plotter = Plots,
    algorithms = ["MKLPardiso", "CG-ILU-drop-0.01", "CG-ILU-drop-0.1", "CG-ILU0"],
    save_figure = true, title_figure = "sym_mp_cg_ilu.pdf")

# MKLPardiso and CG with AMGCL preconditioning
Fig1Runtimes.plot_runtimes(sym * "sym_mp_direct_pardiso_20_belapsed.jld2", sym * "sym_mp_cg_amgcl_22_belapsed.jld2"; Plotter = Plots,
    algorithms = ["MKLPardiso", "CG-AMGCL-RS-SPAI0", "CG-AMGCL-SA-SPAI0", "CG-AMGCL-RS-ILU0", "CG-AMGCL-SA-ILU0"],
    save_figure = true, title_figure = "sym_mp_cg_amgcl.pdf")

# CG, CG + ILU(0), CG + AMGCL in 2D and 3D
cg_files = (sym * "sym_mp_cg_22_belapsed.jld2", sym * "sym_mp_cg_ilu_22_belapsed.jld2", sym * "sym_mp_cg_amgcl_22_belapsed.jld2")
Fig1Runtimes.plot_runtimes(cg_files...; Plotter = Plots,
    algorithms = ["CG", "CG-ILU0", "CG-AMGCL-SA-SPAI0"], active_dimensions = 2,
    references = [(1, 1.0e-6), (3 / 2, 1.0e-8)],
    save_figure = true, title_figure = "sym_mp_cg_ilu_amgcl_2d.pdf")
Fig1Runtimes.plot_runtimes(cg_files...; Plotter = Plots,
    algorithms = ["CG", "CG-ILU0", "CG-AMGCL-SA-SPAI0"], active_dimensions = 3,
    references = [(1, 1.0e-6), (4 / 3, 1.0e-7)],
    save_figure = true, title_figure = "sym_mp_cg_ilu_amgcl_3d.pdf")

# Nonsymmetric problem in 3D: BiCGSTAB and GMRES without preconditioning, with ILU(0) and with AMGCL
Fig1Runtimes.plot_runtimes(
    nsym * "nsym_mp_direct_pardiso_20_belapsed.jld2", nsym * "nsym_mp_krylov_22_belapsed.jld2",
    nsym * "nsym_mp_krylov_ilu_22_belapsed.jld2", nsym * "nsym_mp_krylov_amgcl_spai0_22_belapsed.jld2";
    Plotter = Plots,
    algorithms = ["MKLPardiso", "BiCGSTAB", "GMRES", "BiCGSTAB-ILU0", "GMRES-ILU0", "BiCGSTAB-AMGCL-SA-SPAI0", "GMRES-AMGCL-SA-SPAI0"],
    active_dimensions = 3,
    save_figure = true, title_figure = "nsym_mp_krylov_3d.pdf")
```

The iteration figures use the same calls with `Fig2Iterations.plot_iterations` and result files that contain `niters`, e.g. after rerunning the benchmarks.

### 5. Tables

```julia
includet("post_process/tables/model_problem/tab1_detailed_benchmark_elapsed.jl")
Tab1DetailedBenchmark.export_benchmark_to_latex("jld2_files_server/sym_model_problem/sym_mp_direct_15_elapsed.jld2", "table_elapsed.tex")

includet("post_process/tables/model_problem/tab2_detailed_benchmark_belapsed.jl")
Tab2DetailedBenchmark.export_benchmark_to_latex("jld2_files_server/sym_model_problem/sym_mp_direct_15_belapsed.jld2", "table_belapsed.tex")
```

### 6. Tests

```julia
julia> ]
pkg> test
```

## Citation

If you use this software, please cite it as described in `CITATION.cff`.

## License

MIT, see `LICENSE`.
