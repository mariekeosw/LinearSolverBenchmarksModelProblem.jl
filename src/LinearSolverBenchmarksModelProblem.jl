module LinearSolverBenchmarksModelProblem

module ModelProblem
    using LinearAlgebra: I, lu
    using LinearSolve
    using Pardiso
    using Sparspak
    using ILUZero: ilu0
    using IncompleteLU
    using AMGCLWrap
    using AlgebraicMultigrid
    using LinearSolvePyAMG
    using Krylov

    include("model_problem/benchmark.jl")
    export SolverConfig
    export solver_configs_direct_new
    export solver_configs_direct
    export solver_configs_direct_pardiso
    export sym_solver_configs_cg
    export sym_solver_configs_cg_ilu
    export sym_solver_configs_cg_amgcl
    export sym_solver_configs_cg_amg
    export sym_solver_configs_cg_pyamg
    export nsym_solver_configs_krylov
    export nsym_solver_configs_cgs_bicgstab
    export nsym_solver_configs_gmres
    export nsym_solver_configs_krylov_ilu
    export nsym_solver_configs_krylov_amgcl_spai0
    export nsym_solver_configs_krylov_amgcl_ilu0
    export nsym_solver_configs_krylov_amg
    export nsym_solver_configs_krylov_pyamg

end # module ModelProblem

end # module LinearSolverBenchmarksModelProblem
