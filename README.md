# LinearSolverBenchmarksModelProblem.jl

Benchmarks of sparse direct solvers and preconditioned Krylov methods in Julia for symmetric and nonsymmetric elliptic model problems in one, two and three spatial dimensions.

This repository contains the code, the benchmark results and the post-processing scripts for the JuliaCon proceedings paper *Benchmarking Iterative vs. Direct Linear Solvers for PDEs in Julia* (M. Osewold, J. Fuhrmann, P. Jaap). It is the model-problem part of [`LinearSolverBenchmarks.jl`](https://github.com/mariekeosw/LinearSolverBenchmarks.jl), which additionally contains MOSFET drift-diffusion simulations.

## Model problem and solvers

The matrices have the nearest-neighbor structure of a low-order discretization of $-\Delta u + \delta u = f$ on $(0,1)^d$ with homogeneous Neumann boundary conditions and $\delta = 10^{-2}$ (`src/matrices.jl`):

- symmetric: $A_\mathrm{sym} = L_{d,n} + \delta I$ (`fdmatrix_sym`),
- nonsymmetric: $A_\mathrm{nsym} = L_{d,n} D + \delta I$ with a random diagonal $D$, $d_i \in \{0.8, 0.801, \dots, 1.2\}$, fixed seed (`fdmatrix`).

All solvers are accessed through [`LinearSolve.jl`](https://github.com/SciML/LinearSolve.jl) (`src/solvers.jl`):

| Category | Methods and implementations |
| --- | --- |
| Direct solvers | UMFPACK and KLU from SuiteSparse, MKLPardiso through [`Pardiso.jl`](https://github.com/JuliaSparse/Pardiso.jl), [`Sparspak.jl`](https://github.com/PetrKryslUCSD/Sparspak.jl) |
| Krylov methods | CG (symmetric), BiCGSTAB and GMRES (nonsymmetric) from [`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl) |
| Incomplete factorizations | ILU(0) from [`ILUZero.jl`](https://github.com/mcovalt/ILUZero.jl), threshold ILU (drop tolerances 0.01 and 0.1) from [`IncompleteLU.jl`](https://github.com/haampie/IncompleteLU.jl) |
| Algebraic multigrid | Ruge–Stüben and smoothed aggregation from [`AMGCLWrap.jl`](https://github.com/j-fu/AMGCLWrap.jl) (with SPAI(0) or ILU(0) relaxation), [`AlgebraicMultigrid.jl`](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl) and [`LinearSolvePyAMG.jl`](https://github.com/SciML/LinearSolve.jl/tree/main/lib/LinearSolvePyAMG) |

The linear solver runtime (setup + solve) is measured with `@belapsed` from [`BenchmarkTools.jl`](https://github.com/JuliaCI/BenchmarkTools.jl). The number of Krylov iterations, the return code, the relative residual and the mean of the solution are taken from the solution of the last timed sample (`src/benchmark.jl`).

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
benchmark.sh, benchmark.sub               job script and HTCondor submit file
```

## Usage

You need Julia 1.10 or later. The Manifest records the package versions used for the results in the paper (Julia 1.12).

```bash
git clone https://github.com/mariekeosw/LinearSolverBenchmarksModelProblem.jl
cd LinearSolverBenchmarksModelProblem.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
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

The solver configurations are defined in `src/solvers.jl`. `julia --project scripts/run_benchmarks.jl` runs all benchmarks of the paper and writes the results to `results/` (the largest problem sizes take several days).

A result file contains `labels`, `dimensions`, `refinement`, `problem_sizes`, `tlinsolves` (runtimes), `niters`, `retcodes`, `residuals`, `tests` and `environment` (Julia version, CPU and thread settings). Results are dictionaries with keys `(dimension, label)`.

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

Figures.plot_runtimes(r("sym_cg"), r("sym_cg_ilu"), r("sym_cg_amgcl"); Plotter = Plots,
    algorithms = ["CG", "CG-ILU0", "CG-AMGCL-SA-SPAI0"], active_dimensions = 2,
    references = [(1, 1.0e-6), (3 / 2, 1.0e-8)],
    save_figure = true, title_figure = "sym_mp_cg_ilu_amgcl_2d.pdf")

Figures.plot_runtimes(r("sym_cg"), r("sym_cg_ilu"), r("sym_cg_amgcl"); Plotter = Plots,
    algorithms = ["CG", "CG-ILU0", "CG-AMGCL-SA-SPAI0"], active_dimensions = 3,
    references = [(1, 1.0e-6), (4 / 3, 1.0e-7)],
    save_figure = true, title_figure = "sym_mp_cg_ilu_amgcl_3d.pdf")

Figures.plot_runtimes(r("nsym_direct_pardiso"), r("nsym_krylov"), r("nsym_krylov_ilu"), r("nsym_krylov_amgcl_spai0"); Plotter = Plots,
    algorithms = ["MKLPardiso", "BiCGSTAB", "GMRES", "BiCGSTAB-ILU0", "GMRES-ILU0", "BiCGSTAB-AMGCL-SA-SPAI0", "GMRES-AMGCL-SA-SPAI0"],
    active_dimensions = 3,
    save_figure = true, title_figure = "nsym_mp_krylov_3d.pdf")
```

The iteration figures use the same calls with `Figures.plot_iterations`.

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
