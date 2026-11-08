################################################################################
# Single Solve
################################################################################
# Solve the model problem with N ≈ 10 * 2^refinement unknowns using `alg` and measure the
# linear solver runtime (setup + solve) with BenchmarkTools (minimum over at most 10 samples
# within 0.5 s, at least one sample). Unlike @belapsed, no extra warm-up evaluation is run:
# scripts/run_benchmarks.jl compiles all solvers beforehand, and for the largest problems the
# warm-up would double the runtime. Right-hand side and initial guess are random vectors with
# a fixed seed. The iteration count, return code, relative residual and mean of the solution
# are taken from the solution of the last timed sample. `scaled`: matrix variant, see
# src/matrices.jl. The maximal number of Krylov iterations is MAXITERS instead of the default N
# of LinearSolve.jl, since CG needs more than N iterations for the scaled matrix in 1D.
const MAXITERS = 100_000

# All Krylov methods stop at the same criterion ‖b - Au_k‖₂ ≤ RELTOL ‖b‖₂, evaluated with the
# unpreconditioned residual r that Krylov.jl updates in every iteration (field `r` of the CG
# and BiCGSTAB workspaces). Krylov.jl's own criterion is switched off (atol = rtol = 0): for
# preconditioned CG it uses the norm ‖r‖_{M⁻¹}, which depends on the preconditioner.
# TIMEMAX (s) ends a single solve that does not converge.
const RELTOL = 1.0e-8
const TIMEMAX = 3600.0
euclidean_stopping(alg, tol) = alg
function euclidean_stopping(alg::KrylovJL, tol)
    callback = workspace -> norm(workspace.r) ≤ tol
    return KrylovJL(
        alg.args...; KrylovAlg = alg.KrylovAlg, gmres_restart = alg.gmres_restart, window = alg.window,
        warm_start = alg.warm_start, precs = alg.precs, alg.kwargs..., atol = 0.0, rtol = 0.0, timemax = TIMEMAX, callback
    )
end

function run_single(alg; symmetric = true, dimension = 2, refinement = 5, rng_seed = 123, scaled = false)
    target_problem_size = 10 * 2^refinement
    A = symmetric ? fdmatrix_sym(target_problem_size; dim = dimension, scaled) : fdmatrix_nsym(target_problem_size; dim = dimension, scaled)

    rng = MersenneTwister(rng_seed)
    N = size(A, 1)
    b = rand(rng, N)
    u0 = rand(rng, N)

    prob = LinearProblem(copy(A), copy(b); u0 = copy(u0), alias = LinearAliasSpecifier(alias_A = true, alias_b = true))
    alg = euclidean_stopping(alg, RELTOL * norm(b))

    # Free the memory of previous solves and of the matrix assembly (not timed). Garbage
    # collection is not triggered before each sample (many small benchmarks).
    GC.gc()
    # The Ref stores the solution of the last sample; this does not change the measured runtime.
    sol_ref = Ref{Any}(nothing)
    timing = @benchmarkable begin
        cache = init($prob, $alg; maxiters = $MAXITERS)
        $sol_ref[] = solve!(cache)
    end samples = 10 evals = 1 seconds = 0.5 gctrial = false
    runtime = minimum(run(timing)).time / 1.0e9 # ns -> s

    sol = sol_ref[]
    # Krylov.jl reports a stop by the callback as "user-requested exit", which LinearSolve.jl
    # turns into ReturnCode.Failure; here it means that the stopping criterion is satisfied.
    stopped_by_criterion = hasproperty(sol.stats, :status) && sol.stats.status == "user-requested exit"
    return (;
        problem_size = N,
        runtime,
        niters = sol.iters, # 0 for direct solvers and PyAMG, which do not report iterations
        retcode = stopped_by_criterion ? "Success" : string(sol.retcode),
        residual = norm(A * sol.u - b) / norm(b),
        testval = sum(sol.u) / N,
    )
end

################################################################################
# Benchmark
################################################################################
# Run all solver configurations for refinements 0:refinement_max in the given dimensions
# and store the results in `filename`. Results are dictionaries with keys (dimension, label).
# The file is written after every solve, so an interrupted run keeps all finished results.
# If a solve takes longer than `time_limit` seconds, the larger refinements of this solver
# configuration and dimension are skipped. In 1D, refinements above `refinement_max_1d` are skipped.
function benchmark(filename, configs; symmetric = true, dimensions = [1, 2, 3], refinement_max = 5, refinement_max_1d = refinement_max, scaled = false, time_limit = Inf)
    labels = [config.label for config in configs]
    refinement = collect(0:refinement_max)
    environment = environment_info()

    BenchmarkKey = Tuple{Int, String}
    problem_sizes = Dict{BenchmarkKey, Vector{Int}}()
    tlinsolves = Dict{BenchmarkKey, Vector{Float64}}()
    niters = Dict{BenchmarkKey, Vector{Int}}()
    retcodes = Dict{BenchmarkKey, Vector{String}}()
    residuals = Dict{BenchmarkKey, Vector{Float64}}()
    tests = Dict{BenchmarkKey, Vector{Float64}}()

    failed = (; problem_size = 0, runtime = NaN, niters = -1, retcode = "failed", residual = NaN, testval = NaN)

    for dimension in dimensions, config in configs
        for r in refinement
            dimension == 1 && r > refinement_max_1d && break
            result = try
                run_single(config.alg; symmetric, dimension, refinement = r, scaled)
            catch err
                @warn "Benchmark failed for $(config.label), dimension $dimension, refinement $r" exception = (err, catch_backtrace())
                failed
            end
            @info "$(dimension)D $(config.label) refinement $r" result

            key = (dimension, config.label)
            push!(get!(problem_sizes, key, Int[]), result.problem_size)
            push!(get!(tlinsolves, key, Float64[]), result.runtime)
            push!(get!(niters, key, Int[]), result.niters)
            push!(get!(retcodes, key, String[]), result.retcode)
            push!(get!(residuals, key, Float64[]), result.residual)
            push!(get!(tests, key, Float64[]), result.testval)

            @save filename labels dimensions refinement problem_sizes tlinsolves niters retcodes residuals tests scaled environment

            if result.runtime > time_limit
                @info "$(dimension)D $(config.label): time limit of $time_limit s exceeded, larger refinements skipped"
                break
            end
        end
    end

    return nothing
end

################################################################################
# Benchmark Environment
################################################################################
# Julia version, CPU and thread settings, stored with the results.
function environment_info()
    return Dict(
        "julia_version" => string(VERSION),
        "cpu_name" => Sys.CPU_NAME,
        "cpu_threads" => string(Sys.CPU_THREADS),
        "julia_threads" => string(Threads.nthreads()),
        "blas_threads" => string(BLAS.get_num_threads()),
        "blas_config" => string(BLAS.get_config()),
        "OMP_NUM_THREADS" => get(ENV, "OMP_NUM_THREADS", "unset"),
        "MKL_NUM_THREADS" => get(ENV, "MKL_NUM_THREADS", "unset"),
        "mkl_pardiso_library" => string(Pardiso.libmkl_rt),
        "hostname" => gethostname(),
    )
end
