module LinearSolverBenchmarksModelProblem

using LinearAlgebra: LinearAlgebra, BLAS, Diagonal, I, kron, norm
using SparseArrays: spdiagm
using Random: MersenneTwister
using LinearSolve
using Pardiso: Pardiso    # loads the MKLPardisoFactorize extension of LinearSolve
using Sparspak: Sparspak  # loads the SparspakFactorization extension of LinearSolve
using ILUZero: ilu0
using IncompleteLU: ilu
using AMGCLWrap: AMGCLWrap
using AlgebraicMultigrid: AlgebraicMultigrid
using LinearSolvePyAMG: PyAMG
using BenchmarkTools: @belapsed
using JLD2: @save

include("matrices.jl")
include("solvers.jl")
include("benchmark.jl")

export fdmatrix_sym, fdmatrix
export run_single, benchmark
export SolverConfig
export solver_configs_direct, solver_configs_direct_pardiso
export sym_solver_configs_cg, sym_solver_configs_cg_ilu, sym_solver_configs_cg_amgcl
export sym_solver_configs_cg_amg, sym_solver_configs_cg_pyamg
export nsym_solver_configs_krylov, nsym_solver_configs_krylov_ilu
export nsym_solver_configs_krylov_amgcl_spai0, nsym_solver_configs_krylov_amgcl_ilu0
export nsym_solver_configs_krylov_amg, nsym_solver_configs_krylov_pyamg

end # module LinearSolverBenchmarksModelProblem
