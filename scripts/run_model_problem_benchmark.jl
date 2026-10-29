#=
Elliptic Model Problem and Linear Solver Benchmark

This script constructs and solves symmetric and non-symmetric sparse linear
systems based on a discretization of the elliptic model problem
-Δu + δu = f in one, two, and three spatial dimensions.

The non-symmetric matrices are obtained by applying a reproducible random
diagonal scaling to the symmetric discretization.

The file is divided into three main parts:
1. the construction and solution of a single linear system in main(),
2. the benchmarking routine benchmark(), which compares different linear solver
   configurations across dimensions and refinement levels, and
3. the auxiliary routines used to construct the model problem matrices.
=#

module ModelProblemBenchmarkRun

using Random                               # matrix construction: MersenneTwister(rng_seed)
using SparseArrays                         # matrix construction: spdiagm
using LinearAlgebra                        # matrix construction: I, issymmetric
using LinearSolverBenchmarks: ModelProblem # solver configurations
using LinearSolve                          # linear problem definition and solve
using BenchmarkTools                       # benchmark data
using JLD2                                 # export benchmark data


################################################################################
# Model Problem Simulation
################################################################################
function main(;
        Documentation = true,
        ###############################
        symmetric = true,
        dimension = 1,
        refinement = 0,
        ###############################
        strategy = ModelProblem.UMFPACKFactorization(),
        ###############################
        rng_seed = 123,
        ###############################
        benchmark_strategy = :elapsed # :elapsed for testvalues, :belapsed for real benchmarks
    )

    ################################################################################
    if Documentation
        if symmetric
            println("Set up symmetric model problem")
        else
            println("Set up unsymmetric model problem")
        end
    end
    ################################################################################

    # Target number of unknowns. The actual number is slightly larger because
    # the tensor-product grid uses ceil(N^(1/dimension)) points per direction.
    # This scaling yields comparable problem sizes in one, two, and three dimensions.
    target_problem_size = 10 * 2^refinement

    # Use a fixed seed to ensure reproducibility.
    rng = MersenneTwister(rng_seed)

    if symmetric
        A = fdmatrix_sym(target_problem_size; dim = dimension)
        @assert issymmetric(A)
    else
        A = fdmatrix(target_problem_size; dim = dimension)
        @assert !issymmetric(A)
    end

    problem_size = size(A, 1)
    b = rand(rng, problem_size)
    u0 = rand(rng, problem_size)

    @info "dim=$(dimension): $problem_size × $problem_size"

    if Documentation
        println("*** done\n")
    end

    ################################################################################
    if Documentation
        println("Define linear problem")
    end
    ################################################################################

    prob = LinearProblem(
        copy(A),
        copy(b);
        u0 = copy(u0),
        alias = LinearAliasSpecifier(alias_A = true, alias_b = true)
    )

    if Documentation
        println("*** done\n")
    end

    ################################################################################
    if Documentation
        println("Solve linear system")
    end
    ################################################################################

    cache = nothing
    solution = nothing

    if benchmark_strategy == :elapsed
        # @elapsed measures only one run
        # Separate setup (factorization or preconditioner construction) from solve time.
        # Setup
        tlinsolve_setup = @elapsed begin
            cache = init(prob, strategy)
        end
        # Solve
        tlinsolve_solve = @elapsed begin
            solution = solve!(cache).u

        end

        tlinsolve = tlinsolve_setup + tlinsolve_solve

    elseif benchmark_strategy == :belapsed
        println("start benchmark")
        t = time()

        # @belapsed measures multiple runs
        # Summarize setup (factorization or preconditioner construction) and solve time,
        # since separating both phases would require an additional benchmark with a pre-initialized cache.
        # Disable repeated GarbageCollection pauses during the timing loop of many small benchmarks.
        tlinsolve = @belapsed begin
            cache = init($prob, $strategy)
            solve!(cache)
        end samples = 10 evals = 1 seconds = 0.5 gctrial = false #gcsample = false

        println("benchmark took ", time() - t, " seconds")

        tlinsolve_setup = NaN
        tlinsolve_solve = NaN

    else
        error("Unknown benchmark strategy = $benchmark_strategy")
    end

    if Documentation
        println("*** done\n")
    end

    if benchmark_strategy == :elapsed
        testval = sum(solution) / length(solution)
        residual = norm(A * solution - b) / norm(b)
    elseif benchmark_strategy == :belapsed
        testval = NaN
        residual = NaN
    else
        error("Unknown benchmark strategy = $benchmark_strategy")
    end

    return problem_size, testval, residual, tlinsolve, tlinsolve_setup, tlinsolve_solve
end # function main


################################################################################
# Benchmark the Model Problem Simulation
################################################################################
function benchmark(
        filename;
        dimensions = [1, 2, 3],
        refinement_max = 5,
        symmetric = true,
        solver_config = :direct,
        benchmark_strategy = :elapsed
    )

    ################################################################################
    # Define solver configurations
    ################################################################################
    # Symmetric and Non-Symmetric Problem
    if solver_config == :direct_new
        configs = ModelProblem.solver_configs_direct_new
    elseif solver_config == :direct
        configs = ModelProblem.solver_configs_direct
    elseif solver_config == :direct_pardiso
        configs = ModelProblem.solver_configs_direct_pardiso
        # Symmetric Problem
    elseif solver_config == :sym_cg
        configs = ModelProblem.sym_solver_configs_cg
    elseif solver_config == :sym_cg_ilu
        configs = ModelProblem.sym_solver_configs_cg_ilu
    elseif solver_config == :sym_cg_amgcl
        configs = ModelProblem.sym_solver_configs_cg_amgcl
    elseif solver_config == :sym_cg_amg
        configs = ModelProblem.sym_solver_configs_cg_amg
    elseif solver_config == :sym_cg_pyamg
        configs = ModelProblem.sym_solver_configs_cg_pyamg

        # Non-Symmetric Problem
    elseif solver_config == :nsym_krylov
        configs = ModelProblem.nsym_solver_configs_krylov
    elseif solver_config == :nsym_krylov_cgs_bicgstab
        configs = ModelProblem.nsym_solver_configs_cgs_bicgstab
    elseif solver_config == :nsym_krylov_gmres
        configs = ModelProblem.nsym_solver_configs_gmres
    elseif solver_config == :nsym_krylov_ilu
        configs = ModelProblem.nsym_solver_configs_krylov_ilu
    elseif solver_config == :nsym_krylov_amgcl_spai0
        configs = ModelProblem.nsym_solver_configs_krylov_amgcl_spai0
    elseif solver_config == :nsym_krylov_amgcl_ilu0
        configs = ModelProblem.nsym_solver_configs_krylov_amgcl_ilu0
    elseif solver_config == :nsym_krylov_amg
        configs = ModelProblem.nsym_solver_configs_krylov_amg
    elseif solver_config == :nsym_krylov_pyamg
        configs = ModelProblem.nsym_solver_configs_krylov_pyamg

    else
        error("Unknown solver configuration = $solver_config")
    end

    refinement = collect(0:refinement_max)

    labels = [config.label for config in configs]

    ################################################################################
    # Create storage for benchmark data
    ################################################################################
    BenchmarkKey = Tuple{Int, String}

    walltimes = Dict{BenchmarkKey, Vector{Float64}}()
    tlinsolves = Dict{BenchmarkKey, Vector{Float64}}()
    tlinsolve_setups = Dict{BenchmarkKey, Vector{Float64}}()
    tlinsolve_solves = Dict{BenchmarkKey, Vector{Float64}}()

    problem_sizes = Dict{BenchmarkKey, Vector{Int}}()
    tests = Dict{BenchmarkKey, Vector{Float64}}()
    residuals = Dict{BenchmarkKey, Vector{Float64}}()

    ################################################################################
    # Benchmark loop
    ################################################################################
    for dimension in dimensions
        for config in configs
            label = config.label
            alg = config.alg
            key = (dimension, label)

            println("dimension = $(dimension)")
            println("algorithm = $(label)")

            walltimes_value = Vector{Float64}(undef, length(refinement))
            tlinsolves_value = Vector{Float64}(undef, length(refinement))
            tlinsolve_setups_value = Vector{Float64}(undef, length(refinement))
            tlinsolve_solves_value = Vector{Float64}(undef, length(refinement))

            problem_sizes_value = Vector{Int}(undef, length(refinement))
            tests_value = Vector{Float64}(undef, length(refinement))
            residuals_value = Vector{Float64}(undef, length(refinement))

            for j in eachindex(refinement)
                println("refinement = $(refinement[j])")

                # Default values if the benchmark fails
                wt = NaN
                tlinsolve = NaN
                tlinsolve_setup = NaN
                tlinsolve_solve = NaN
                problem_size = 0
                testval = NaN
                residual = NaN

                if benchmark_strategy == :elapsed

                    try
                        wt = @elapsed begin
                            problem_size, testval, residual, tlinsolve, tlinsolve_setup, tlinsolve_solve = main(
                                Documentation = false,
                                symmetric = symmetric,
                                dimension = dimension,
                                refinement = refinement[j],
                                strategy = alg,
                                benchmark_strategy = benchmark_strategy
                            )
                        end
                    catch err
                        @warn "Benchmark failed for algorithm '$label' at dimension $(dimension) and refinement $(refinement[j])" exception = (err, catch_backtrace())
                    end

                elseif benchmark_strategy == :belapsed

                    try
                        problem_size, testval, residual, tlinsolve, tlinsolve_setup, tlinsolve_solve = main(
                            Documentation = false,
                            symmetric = symmetric,
                            dimension = dimension,
                            refinement = refinement[j],
                            strategy = alg,
                            benchmark_strategy = benchmark_strategy
                        )

                    catch err
                        @warn "Benchmark failed for algorithm '$label' at dimension $(dimension) and refinement $(refinement[j])" exception = (err, catch_backtrace())
                    end

                else
                    error("Unknown benchmark strategy = $benchmark_strategy")
                end

                walltimes_value[j] = wt
                tlinsolves_value[j] = tlinsolve
                tlinsolve_setups_value[j] = tlinsolve_setup
                tlinsolve_solves_value[j] = tlinsolve_solve

                problem_sizes_value[j] = problem_size
                tests_value[j] = testval
                residuals_value[j] = residual

                println("walltime = $wt")
                println("tlinsolve = $tlinsolve")
                println("tlinsolve_setup = $tlinsolve_setup")
                println("tlinsolve_solve = $tlinsolve_solve")
                println("problem_size = $problem_size")
                println("testval = $testval")
                println("relative residual = $residual")
                println("*** done\n")
            end

            walltimes[key] = walltimes_value
            tlinsolves[key] = tlinsolves_value
            tlinsolve_setups[key] = tlinsolve_setups_value
            tlinsolve_solves[key] = tlinsolve_solves_value

            problem_sizes[key] = problem_sizes_value
            tests[key] = tests_value
            residuals[key] = residuals_value
        end
    end

    @save filename labels dimensions refinement problem_sizes tests residuals walltimes tlinsolves tlinsolve_setups tlinsolve_solves

    return nothing
end # function benchmark

################################################################################
# Benchmark Setup Server
################################################################################
function several_benchmarks(; benchmark_strategy = :elapsed)
    ################################################################################
    # First: Symmetric Problems
    ################################################################################
    # Direct solver from LinearSolve.jl and Pardiso.jl
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = true, solver_config = :direct, benchmark_strategy = benchmark_strategy)
    tsd = @elapsed ModelProblemBenchmarkRun.benchmark("sym_mp_direct_15_$benchmark_strategy.jld2"; refinement_max = 15, symmetric = true, solver_config = :direct, benchmark_strategy = benchmark_strategy)

    # Direct solver MKLPardiso from Pardiso.jl
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = true, solver_config = :direct_pardiso, benchmark_strategy = benchmark_strategy)
    tsdp = @elapsed ModelProblemBenchmarkRun.benchmark("sym_mp_direct_pardiso_20_$benchmark_strategy.jld2"; refinement_max = 20, symmetric = true, solver_config = :direct_pardiso, benchmark_strategy = benchmark_strategy)

    # CG without preconditioning
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = true, solver_config = :sym_cg, benchmark_strategy = benchmark_strategy)
    tscg = @elapsed ModelProblemBenchmarkRun.benchmark("sym_mp_cg_22_$benchmark_strategy.jld2"; refinement_max = 22, symmetric = true, solver_config = :sym_cg, benchmark_strategy = benchmark_strategy)

    # CG with ILU preconditioning from IncompleteLU.jl and ILUZero.jl
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = true, solver_config = :sym_cg_ilu, benchmark_strategy = benchmark_strategy)
    tscgilu = @elapsed ModelProblemBenchmarkRun.benchmark("sym_mp_cg_ilu_22_$benchmark_strategy.jld2"; refinement_max = 22, symmetric = true, solver_config = :sym_cg_ilu, benchmark_strategy = benchmark_strategy)

    # CG with AMG preconditioning from AMGCLWrap.jl
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = true, solver_config = :sym_cg_amgcl, benchmark_strategy = benchmark_strategy)
    tscgamgcl = @elapsed ModelProblemBenchmarkRun.benchmark("sym_mp_cg_amgcl_22_$benchmark_strategy.jld2"; refinement_max = 22, symmetric = true, solver_config = :sym_cg_amgcl, benchmark_strategy = benchmark_strategy)

    # CG with AMG preconditioning from AlgebraicMultigrid.jl
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = true, solver_config = :sym_cg_amg, benchmark_strategy = benchmark_strategy)
    tscgamg_1 = @elapsed ModelProblemBenchmarkRun.benchmark("sym_mp_cg_amg_20_1d_$benchmark_strategy.jld2"; dimensions = [1], refinement_max = 20, symmetric = true, solver_config = :sym_cg_amg, benchmark_strategy = benchmark_strategy)
    tscgamg_2 = @elapsed ModelProblemBenchmarkRun.benchmark("sym_mp_cg_amg_22_23d_$benchmark_strategy.jld2"; dimensions = [2, 3], refinement_max = 22, symmetric = true, solver_config = :sym_cg_amg, benchmark_strategy = benchmark_strategy)

    # CG with PyAMG preconditioning from LinearSolvePyAMG.jl
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = true, solver_config = :sym_cg_pyamg, benchmark_strategy = benchmark_strategy)
    tscgpyamg = @elapsed ModelProblemBenchmarkRun.benchmark("sym_mp_cg_pyamg_22_$benchmark_strategy.jld2"; refinement_max = 22, symmetric = true, solver_config = :sym_cg_pyamg, benchmark_strategy = benchmark_strategy)

    ################################################################################
    # Second: Non-Symmetric Problems
    ################################################################################
    # Direct solver from LinearSolve.jl and Pardiso.jl
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = false, solver_config = :direct, benchmark_strategy = benchmark_strategy)
    tnsd = @elapsed ModelProblemBenchmarkRun.benchmark("nsym_mp_direct_15_$benchmark_strategy.jld2"; refinement_max = 15, symmetric = false, solver_config = :direct, benchmark_strategy = benchmark_strategy)

    # Direct solver MKLPardiso from Pardiso.jl
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = false, solver_config = :direct_pardiso, benchmark_strategy = benchmark_strategy)
    tnsdp = @elapsed ModelProblemBenchmarkRun.benchmark("nsym_mp_direct_pardiso_20_$benchmark_strategy.jld2"; refinement_max = 20, symmetric = false, solver_config = :direct_pardiso, benchmark_strategy = benchmark_strategy)

    # Krylov Methods without preconditioning
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = false, solver_config = :nsym_krylov, benchmark_strategy = benchmark_strategy)
    tnsk = @elapsed ModelProblemBenchmarkRun.benchmark("nsym_mp_krylov_22_$benchmark_strategy.jld2"; refinement_max = 22, symmetric = false, solver_config = :nsym_krylov, benchmark_strategy = benchmark_strategy)

    # Krylov Methods with ILU preconditioning from IncompleteLU.jl and ILUZero.jl
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = false, solver_config = :nsym_krylov_ilu, benchmark_strategy = benchmark_strategy)
    tnskilu = @elapsed ModelProblemBenchmarkRun.benchmark("nsym_mp_krylov_ilu_22_$benchmark_strategy.jld2"; refinement_max = 22, symmetric = false, solver_config = :nsym_krylov_ilu, benchmark_strategy = benchmark_strategy)

    # Krylov Methods with AMG preconditioning from AMGCLWrap.jl for spai0
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = false, solver_config = :nsym_krylov_amgcl_spai0, benchmark_strategy = benchmark_strategy)
    tnskamgclspai0 = @elapsed ModelProblemBenchmarkRun.benchmark("nsym_mp_krylov_amgcl_spai0_22_$benchmark_strategy.jld2"; refinement_max = 22, symmetric = false, solver_config = :nsym_krylov_amgcl_spai0, benchmark_strategy = benchmark_strategy)

    # Krylov Methods with AMG preconditioning from AMGCLWrap.jl for ilu0
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = false, solver_config = :nsym_krylov_amgcl_ilu0, benchmark_strategy = benchmark_strategy)
    tnskamgclilu0_1 = @elapsed ModelProblemBenchmarkRun.benchmark("nsym_mp_krylov_amgcl_ilu0_20_1d_$benchmark_strategy.jld2"; dimensions = [1], refinement_max = 20, symmetric = false, solver_config = :nsym_krylov_amgcl_ilu0, benchmark_strategy = benchmark_strategy)
    tnskamgclilu0_2 = @elapsed ModelProblemBenchmarkRun.benchmark("nsym_mp_krylov_amgcl_ilu0_22_23d_$benchmark_strategy.jld2"; dimensions = [2, 3], refinement_max = 22, symmetric = false, solver_config = :nsym_krylov_amgcl_ilu0, benchmark_strategy = benchmark_strategy)

    # Krylov Methods with AMG preconditioning from AlgebraicMultigrid.jl
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = false, solver_config = :nsym_krylov_amg, benchmark_strategy = benchmark_strategy)
    tnskamg_1 = @elapsed ModelProblemBenchmarkRun.benchmark("nsym_mp_krylov_amg_20_1d_$benchmark_strategy.jld2"; dimensions = [1], refinement_max = 20, symmetric = false, solver_config = :nsym_krylov_amg, benchmark_strategy = benchmark_strategy)
    tnskamg_2 = @elapsed ModelProblemBenchmarkRun.benchmark("nsym_mp_krylov_amg_22_23d_$benchmark_strategy.jld2"; dimensions = [2, 3], refinement_max = 22, symmetric = false, solver_config = :nsym_krylov_amg, benchmark_strategy = benchmark_strategy)

    # Krylov Methods with PyAMG preconditioning from LinearSolvePyAMG.jl
    ModelProblemBenchmarkRun.benchmark("warmup_$benchmark_strategy.jld2"; refinement_max = 0, symmetric = false, solver_config = :nsym_krylov_pyamg, benchmark_strategy = benchmark_strategy)
    tnskpyamg = @elapsed ModelProblemBenchmarkRun.benchmark("nsym_mp_krylov_pyamg_22_$benchmark_strategy.jld2"; refinement_max = 22, symmetric = false, solver_config = :nsym_krylov_pyamg, benchmark_strategy = benchmark_strategy)

    println("\nBenchmark runtimes:")

    println("\nSymmetric problems:")
    println("tsd       = $tsd seconds")
    println("tsdp      = $tsdp seconds")
    println("tscg      = $tscg seconds")
    println("tscgilu   = $tscgilu seconds")
    println("tscgamgcl = $tscgamgcl seconds")
    println("tscgamg_1   = $tscgamg_1 seconds")
    println("tscgamg_2   = $tscgamg_2 seconds")
    println("tscgpyamg = $tscgpyamg seconds")

    println("\nNon-symmetric problems:")
    println("tnsd      = $tnsd seconds")
    println("tnsdp     = $tnsdp seconds")
    println("tnsk      = $tnsk seconds")
    println("tnskilu   = $tnskilu seconds")
    println("tnskamgclspai0 = $tnskamgclspai0 seconds")
    println("tnskamgclilu0_1 = $tnskamgclilu0_1 seconds")
    println("tnskamgclilu0_2 = $tnskamgclilu0_2 seconds")
    println("tnskamg_1   = $tnskamg_1 seconds")
    println("tnskamg_2   = $tnskamg_2 seconds")
    println("tnskpyamg = $tnskpyamg seconds")

    return nothing
end # several_benchmarks

################################################################################
# Matrix Construction
################################################################################
# Sparse matrix generation on  a n-dimensional rectangular grid. After
# https://discourse.julialang.org/t/seven-lines-of-julia-examples-sought/50416/135
# by A. Braunstein.
A ⊕ B = kron(I(size(B, 1)), A) + kron(B, I(size(A, 1)))

function lattice(n; Tv = Float64)
    d = fill(2 * one(Tv), n)
    d[1] = one(Tv)
    d[end] = one(Tv)
    return spdiagm(1 => -ones(Tv, n - 1), 0 => d, -1 => -ones(Tv, n - 1))
end

lattice(L...; Tv = Float64) = lattice(L[1]; Tv) ⊕ lattice(L[2:end]...; Tv)

# Create a matrix similar to that of a finite difference discretization in a `dim`-dimensional
# unit cube of  ``-Δu + δu`` with approximately N unknowns. It is strictly diagonally dominant.

# Creates a symmetric matrix
function fdmatrix_sym(N; dim = 2, Tv = Float64, δ = 1.0e-2)
    n = N^(1 / dim) |> ceil |> Int
    A = lattice([n for i in 1:dim]...; Tv) + Tv(δ) * I
    return A
end

# Creates a reproducible non-symmetric matrix.
# The diagonal entries of D are randomly selected from
# {0.8, 0.801, ..., 1.2} using a fixed random seed.
function fdmatrix(N; dim = 2, Tv = Float64, δ = 1.0e-2, skew = 0.2, rng_seed = 123)
    n = N^(1 / dim) |> ceil |> Int
    rng = MersenneTwister(rng_seed)
    diagonal_entries = rand(rng, (1 - skew):0.001:(1 + skew), n^dim)
    D = Diagonal(Tv.(diagonal_entries))
    A = lattice([n for i in 1:dim]...; Tv) * D + Tv(δ) * I
    return A
end

end # module
