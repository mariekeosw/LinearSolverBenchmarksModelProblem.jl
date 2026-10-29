module Fig1Runtimes

#=
Linear solver runtimes (setup + solve, i.e. `tlinsolves`) over the number of unknowns.

How to use this function?
1) includet("post_process/figures/model_problem/fig1_runtimes.jl")
2) using Plots
3) Fig1Runtimes.plot_runtimes(
       "jld2_files_server/sym_model_problem/sym_mp_direct_pardiso_20_belapsed.jld2",
       "jld2_files_server/sym_model_problem/sym_mp_cg_ilu_22_belapsed.jld2";
       Plotter = Plots,
       algorithms = ["MKLPardiso", "CG-ILU-drop-0.01", "CG-ILU-drop-0.1", "CG-ILU0"],
       save_figure = true,
       title_figure = "sym_mp_cg_ilu.pdf",
   )

Arguments:
- any number of .jld2 files; their results are merged
- `algorithms`: algorithms to plot, in this order (defines the colors); `nothing` plots all
- `active_dimensions`: e.g. 3 or [2, 3]; `nothing` plots all dimensions in the files
- `references`: reference lines factor * N^exponent as (exponent, factor) pairs
- `ylims`, `yticks`, `xlims`: axis settings

The calls for all figures of the paper are listed in the README.
=#

include("common.jl")

function plot_runtimes(
        filenames...;
        Plotter = nothing,
        algorithms = nothing,
        active_dimensions = nothing,
        references = [(1, 1.0e-4), (1, 1.0e-7), (2, 1.0e-9)],
        ylims = (1.0e-4, 1.0e4),
        yticks = 10.0 .^ (-4:2:4),
        xlims = DEFAULT_XLIMS,
        save_figure = false,
        title_figure = nothing,
    )

    if isnothing(Plotter)
        error("Please add Plots.jl in local environment and use Plotter = Plots")
    end

    labels, dimensions, data = load_benchmarks(filenames...; fields = ("problem_sizes", "tlinsolves"))
    labels = select_labels(labels, algorithms)
    dimensions = select_dimensions(dimensions, active_dimensions)

    problem_sizes = data["problem_sizes"]
    tlinsolves = data["tlinsolves"]

    p = base_plot(Plotter; ylabel = L"\mathrm{runtime\ [s]}", ylims, yticks, xlims)

    for dimension in dimensions, (i, label) in enumerate(labels)
        key = (dimension, label)
        haskey(tlinsolves, key) || continue

        # Problem size 0 or NaN runtime: the run failed or was not performed (e.g. GMRES with PyAMG)
        valid = (problem_sizes[key] .> 0) .& isfinite.(tlinsolves[key])

        plot_series!(p, Plotter, problem_sizes[key][valid], tlinsolves[key][valid], COLORS[i], linestyle_for(dimension, dimensions))
    end

    add_legend!(p, Plotter, labels, dimensions)
    add_references!(p, Plotter, references, xlims)

    if save_figure
        Plotter.savefig(p, title_figure)
    end

    return p
end

end # module
