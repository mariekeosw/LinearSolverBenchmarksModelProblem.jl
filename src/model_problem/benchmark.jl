################################################################################
# Solver Configurations
################################################################################
struct SolverConfig{T}
    label::String
    alg::T
end

################################################################################
# Solver Configurations: Symmetric and Non-Symmetric Problem
################################################################################
# Direct solver - new pure Julia implementations
const solver_configs_direct_new = (
    SolverConfig("PureKLU", PureKLUFactorization()),
    SolverConfig("SupernodalLU", SupernodalLUFactorization()),
    SolverConfig("UMFPACK", UMFPACKFactorization()),
    SolverConfig("KLU", KLUFactorization()),
    SolverConfig("MKLPardiso", MKLPardisoFactorize()),
    SolverConfig("Sparspak", SparspakFactorization()),
)

# Direct solver from LinearSolve.jl and Pardiso.jl
const solver_configs_direct = (
    SolverConfig("UMFPACK", UMFPACKFactorization()),
    SolverConfig("KLU", KLUFactorization()),
    SolverConfig("MKLPardiso", MKLPardisoFactorize()),
    SolverConfig("Sparspak", SparspakFactorization()),
)

# Direct solver MKLPardiso from Pardiso.jl
const solver_configs_direct_pardiso = (
    SolverConfig("MKLPardiso", MKLPardisoFactorize()),
)

################################################################################
# Solver Configurations: Symmetric Problem
################################################################################
# CG without preconditioning
const sym_solver_configs_cg = (
    SolverConfig("CG", KrylovJL_CG()),
)

# CG with ILU preconditioning from IncompleteLU.jl and ILUZero.jl
const sym_solver_configs_cg_ilu = (
    SolverConfig("CG-ILU-drop-0.01", KrylovJL_CG(precs = (A, p) -> (ilu(A; τ = 0.01), I))),
    SolverConfig("CG-ILU-drop-0.1", KrylovJL_CG(precs = (A, p) -> (ilu(A; τ = 0.1), I))),
    SolverConfig("CG-ILU0", KrylovJL_CG(precs = (A, p) -> (ilu0(A), I))),
)

# CG with AMG preconditioning from AMGCLWrap.jl
const sym_solver_configs_cg_amgcl = (
    SolverConfig("CG-AMGCL-RS-SPAI0", KrylovJL_CG(precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.RugeStubenCoarsening(), relax = AMGCLWrap.SPAI0Relaxation()))),
    SolverConfig("CG-AMGCL-SA-SPAI0", KrylovJL_CG(precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.SmoothedAggregationCoarsening(), relax = AMGCLWrap.SPAI0Relaxation()))), # Default
    SolverConfig("CG-AMGCL-RS-ILU0", KrylovJL_CG(precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.RugeStubenCoarsening(), relax = AMGCLWrap.ILU0Relaxation()))),
    SolverConfig("CG-AMGCL-SA-ILU0", KrylovJL_CG(precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.SmoothedAggregationCoarsening(), relax = AMGCLWrap.ILU0Relaxation()))),
)

# CG with AMG preconditioning from AlgebraicMultigrid.jl
const sym_solver_configs_cg_amg = (
    SolverConfig("CG-AMG-RS", KrylovJL_CG(precs = AlgebraicMultigrid.RugeStubenPreconBuilder())),
    SolverConfig("CG-AMG-SA", KrylovJL_CG(precs = AlgebraicMultigrid.SmoothedAggregationPreconBuilder())),
)

# CG with PyAMG preconditioning from LinearSolvePyAMG.jl
const sym_solver_configs_cg_pyamg = (
    SolverConfig("CG-PyAMG-RS", PyAMG(method = :RugeStuben, accel = "cg")),
    SolverConfig("CG-PyAMG-SA", PyAMG(method = :SmoothedAggregation, accel = "cg")),
)

################################################################################
# Solver Configurations: Non-Symmetric Problem
################################################################################
# Krylov Methods without preconditioning
const nsym_solver_configs_krylov = (
    SolverConfig("CGS", KrylovJL(KrylovAlg = Krylov.cgs!)),
    SolverConfig("BiCGSTAB", KrylovJL_BICGSTAB()),
    SolverConfig("GMRES", KrylovJL_GMRES()),
)

const nsym_solver_configs_cgs_bicgstab = (
    SolverConfig("CGS", KrylovJL(KrylovAlg = Krylov.cgs!)),
    SolverConfig("BiCGSTAB", KrylovJL_BICGSTAB()),
)

const nsym_solver_configs_gmres = (
    SolverConfig("GMRES", KrylovJL_GMRES()),
)

# Krylov Methods with ILU preconditioning from IncompleteLU.jl and ILUZero.jl
const nsym_solver_configs_krylov_ilu = (
    SolverConfig("CGS-ILU-drop-0.01", KrylovJL(KrylovAlg = Krylov.cgs!, precs = (A, p) -> (ilu(A; τ = 0.01), I))),
    SolverConfig("BiCGSTAB-ILU-drop-0.01", KrylovJL_BICGSTAB(precs = (A, p) -> (ilu(A; τ = 0.01), I))),
    SolverConfig("GMRES-ILU-drop-0.01", KrylovJL_GMRES(precs = (A, p) -> (ilu(A; τ = 0.01), I))),

    SolverConfig("CGS-ILU-drop-0.1", KrylovJL(KrylovAlg = Krylov.cgs!, precs = (A, p) -> (ilu(A; τ = 0.1), I))),
    SolverConfig("BiCGSTAB-ILU-drop-0.1", KrylovJL_BICGSTAB(precs = (A, p) -> (ilu(A; τ = 0.1), I))),
    SolverConfig("GMRES-ILU-drop-0.1", KrylovJL_GMRES(precs = (A, p) -> (ilu(A; τ = 0.1), I))),

    SolverConfig("CGS-ILU0", KrylovJL(KrylovAlg = Krylov.cgs!, precs = (A, p) -> (ilu0(A), I))),
    SolverConfig("BiCGSTAB-ILU0", KrylovJL_BICGSTAB(precs = (A, p) -> (ilu0(A), I))),
    SolverConfig("GMRES-ILU0", KrylovJL_GMRES(precs = (A, p) -> (ilu0(A), I))),
)

# Krylov Methods with AMG preconditioning from AMGCLWrap.jl
const nsym_solver_configs_krylov_amgcl_spai0 = (
    SolverConfig("CGS-AMGCL-SA-SPAI0", KrylovJL(KrylovAlg = Krylov.cgs!, precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.SmoothedAggregationCoarsening(), relax = AMGCLWrap.SPAI0Relaxation()))),
    SolverConfig("BiCGSTAB-AMGCL-SA-SPAI0", KrylovJL_BICGSTAB(precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.SmoothedAggregationCoarsening(), relax = AMGCLWrap.SPAI0Relaxation()))),
    SolverConfig("GMRES-AMGCL-SA-SPAI0", KrylovJL_GMRES(precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.SmoothedAggregationCoarsening(), relax = AMGCLWrap.SPAI0Relaxation()))),

    SolverConfig("CGS-AMGCL-RS-SPAI0", KrylovJL(KrylovAlg = Krylov.cgs!, precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.RugeStubenCoarsening(), relax = AMGCLWrap.SPAI0Relaxation()))),
    SolverConfig("BiCGSTAB-AMGCL-RS-SPAI0", KrylovJL_BICGSTAB(precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.RugeStubenCoarsening(), relax = AMGCLWrap.SPAI0Relaxation()))),
    SolverConfig("GMRES-AMGCL-RS-SPAI0", KrylovJL_GMRES(precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.RugeStubenCoarsening(), relax = AMGCLWrap.SPAI0Relaxation()))),
)

const nsym_solver_configs_krylov_amgcl_ilu0 = (
    SolverConfig("CGS-AMGCL-SA-ILU0", KrylovJL(KrylovAlg = Krylov.cgs!, precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.SmoothedAggregationCoarsening(), relax = AMGCLWrap.ILU0Relaxation()))),
    SolverConfig("BiCGSTAB-AMGCL-SA-ILU0", KrylovJL_BICGSTAB(precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.SmoothedAggregationCoarsening(), relax = AMGCLWrap.ILU0Relaxation()))),
    SolverConfig("GMRES-AMGCL-SA-ILU0", KrylovJL_GMRES(precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.SmoothedAggregationCoarsening(), relax = AMGCLWrap.ILU0Relaxation()))),

    SolverConfig("CGS-AMGCL-RS-ILU0", KrylovJL(KrylovAlg = Krylov.cgs!, precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.RugeStubenCoarsening(), relax = AMGCLWrap.ILU0Relaxation()))),
    SolverConfig("BiCGSTAB-AMGCL-RS-ILU0", KrylovJL_BICGSTAB(precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.RugeStubenCoarsening(), relax = AMGCLWrap.ILU0Relaxation()))),
    SolverConfig("GMRES-AMGCL-RS-ILU0", KrylovJL_GMRES(precs = AMGCLWrap.AMGPreconBuilder(coarsening = AMGCLWrap.RugeStubenCoarsening(), relax = AMGCLWrap.ILU0Relaxation()))),
)

# Krylov Methods with AMG preconditioning from AlgebraicMultigrid.jl
const nsym_solver_configs_krylov_amg = (
    SolverConfig("CGS-AMG-RS", KrylovJL(KrylovAlg = Krylov.cgs!, precs = AlgebraicMultigrid.RugeStubenPreconBuilder())),
    SolverConfig("BiCGSTAB-AMG-RS", KrylovJL_BICGSTAB(precs = AlgebraicMultigrid.RugeStubenPreconBuilder())),
    SolverConfig("GMRES-AMG-RS", KrylovJL_GMRES(precs = AlgebraicMultigrid.RugeStubenPreconBuilder())),

    SolverConfig("CGS-AMG-SA", KrylovJL(KrylovAlg = Krylov.cgs!, precs = AlgebraicMultigrid.SmoothedAggregationPreconBuilder())),
    SolverConfig("BiCGSTAB-AMG-SA", KrylovJL_BICGSTAB(precs = AlgebraicMultigrid.SmoothedAggregationPreconBuilder())),
    SolverConfig("GMRES-AMG-SA", KrylovJL_GMRES(precs = AlgebraicMultigrid.SmoothedAggregationPreconBuilder())),
)

# Krylov Methods with PyAMG preconditioning from LinearSolvePyAMG.jl
const nsym_solver_configs_krylov_pyamg = (
    SolverConfig("BiCGSTAB-PyAMG-RS", PyAMG(method = :RugeStuben, accel = "bicgstab")),
    SolverConfig("BiCGSTAB-PyAMG-SA", PyAMG(method = :SmoothedAggregation, accel = "bicgstab")),
    SolverConfig("GMRES-PyAMG-RS", PyAMG(method = :RugeStuben, accel = "gmres")),
    SolverConfig("GMRES-PyAMG-SA", PyAMG(method = :SmoothedAggregation, accel = "gmres")),
)
