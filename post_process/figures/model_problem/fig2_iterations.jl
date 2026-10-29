module Fig2Iterations

#=
Number of Krylov iterations (`niters`) over the number of unknowns.

Requires .jld2 files that contain `niters` and `retcodes`, i.e. files created with the
current version of scripts/run_model_problem_benchmark.jl. Direct solvers and PyAMG do not
report iteration counts; they keep their color (so colors match Fig1Runtimes for the same
`algorithms` list) but are not plotted.

How to use this function?
1) includet("post_process/figures/model_problem/fig2_iterations.jl")
2) using Plots
3) Fig2Iterations.plot_iterations(
       "sym_mp_cg_22_belapsed.jld2",
       "sym_mp_cg_ilu_22_belapsed.jld2",
       "sym_mp_cg_amgcl_22_belapsed.jld2";
       Plotter = Plots,
       algorithms = ["CG", "CG-ILU0", "CG-AMGCL-SA-SPAI0"],
       active_dimensions = 3,
       save_figure = true,
       title_figure = "sym_mp_cg_ilu_amgcl_3d_iterations.pdf",
   )

Arguments: as in Fig1Runtimes.plot_runtimes. Points where the Krylov method did not
report convergence (return code other than Success) are additionally marked with a cross.
=#

include("common.jl")

function plot_iterations(
        filenames...;
        Plotter = nothing,
        algorithms = nothing,
        active_dimensions = nothing,
        references = [],
        ylims = :auto,
        yticks = :auto,
        xlims = DEFAULT_XLIMS,
        save_figure = false,
        title_figure = nothing,
    )

    if isnothing(Plotter)
        error("Please add Plots.jl in local environment and use Plotter = Plots")
    end

    labels, dimensions, data = load_benchmarks(filenames...; fields = ("problem_sizes", "niters", "retcodes"))
    labels = select_labels(labels, algorithms)
    dimensions = select_dimensions(dimensions, active_dimensions)

    problem_sizes = data["problem_sizes"]
    niters = data["niters"]
    retcodes = data["retcodes"]

    p = base_plot(Plotter; ylabel = L"\mathrm{number\ of\ iterations}", ylims, yticks, xlims)

    without_iterations = Set(labels)
    not_converged_in_legend = false

    for dimension in dimensions, (i, label) in enumerate(labels)
        key = (dimension, label)
        haskey(niters, key) || continue

        # Iteration count 0: direct solver or PyAMG (no count reported); -1: the run failed
        valid = (problem_sizes[key] .> 0) .& (niters[key] .> 0)
        any(valid) || continue
        delete!(without_iterations, label)

        plot_series!(p, Plotter, problem_sizes[key][valid], niters[key][valid], COLORS[i], linestyle_for(dimension, dimensions))

        # Mark problem sizes where the Krylov method did not report convergence
        not_converged = valid .& .!endswith.(retcodes[key], "Success")
        if any(not_converged)
            Plotter.scatter!(
                p,
                problem_sizes[key][not_converged],
                niters[key][not_converged];
                marker = :xcross,
                markersize = 6,
                color = :black,
                label = "not converged",
                primary = !not_converged_in_legend,
            )
            not_converged_in_legend = true
        end
    end

    isempty(without_iterations) ||
        @info "No iteration counts (direct solver or PyAMG): $(join(sort(collect(without_iterations)), ", "))"

    add_legend!(p, Plotter, labels, dimensions; skip = without_iterations)
    add_references!(p, Plotter, references, xlims)

    if save_figure
        Plotter.savefig(p, title_figure)
    end

    return p
end

end # module
