module Figures

#=
Figures of linear solver runtimes (plot_runtimes) and Krylov iteration counts (plot_iterations)
over the number of unknowns N.

Usage (Plots.jl in your global environment):
    using Plots
    include("postprocess/figures.jl")
    Figures.plot_runtimes(
        "results/sym_direct_pardiso.jld2", "results/sym_cg_ilu.jld2";
        Plotter = Plots,
        algorithms = ["MKLPardiso", "CG-ILU-drop-0.01", "CG-ILU-drop-0.1", "CG-ILU0"],
        save_figure = true,
        title_figure = "sym_mp_cg_ilu.pdf",
    )
    Figures.plot_iterations(<same files and keyword arguments>)

Keyword arguments of both functions:
- any number of .jld2 files; their results are merged
- `algorithms`: algorithms to plot, in this order (defines the colors, so the same list
  gives the same colors in both functions); `nothing` plots all
- `active_dimensions`: e.g. 3 or [2, 3]; `nothing` plots all dimensions in the files
- `references`: reference lines factor * N^exponent as (exponent, factor) pairs
- `ylims`, `yticks`, `xlims`: axis settings

Direct solvers and PyAMG do not report iteration counts and are skipped in plot_iterations.
Points where a Krylov method did not report convergence are marked with a cross.
The calls for all figures of the paper are listed in the README.
=#

using JLD2
using LaTeXStrings
using ColorSchemes

# Line style per spatial dimension (1D, 2D, 3D). A figure with a single dimension uses solid lines.
const LINESTYLES = [:dot, :dash, :solid]

# Colors per algorithm, assigned in the order of the selected algorithms.
const COLORS = ColorSchemes.seaborn_deep

# Default axis limits: up to refinement 22, i.e. N_max = 42144192.
const DEFAULT_XLIMS = (1.0e2 / 1.3, 42144192 * 1.3)

################################################################################
# Loading and selecting data
################################################################################
# Load one or more benchmark files and merge the requested result fields.
function load_benchmarks(filenames...; fields)
    isempty(filenames) && error("Please provide at least one .jld2 file.")

    data = [JLD2.load(filename) for filename in filenames]

    for (filename, d) in zip(filenames, data), field in fields
        haskey(d, field) || error(
            "File $filename does not contain `$field`."
        )
    end

    labels = unique(vcat([d["labels"] for d in data]...))
    dimensions = sort(unique(vcat([collect(d["dimensions"]) for d in data]...)))
    merged = Dict(field => merge([d[field] for d in data]...) for field in fields)

    return labels, dimensions, merged
end

# Keep the algorithms in the order given by the user (this order defines the colors).
function select_labels(labels, algorithms)
    isnothing(algorithms) && return labels

    not_found = setdiff(algorithms, labels)
    isempty(not_found) || @warn "Not found in the given files: $(join(not_found, ", "))"

    return [algorithm for algorithm in algorithms if algorithm in labels]
end

function select_dimensions(dimensions, active_dimensions)
    isnothing(active_dimensions) && return dimensions

    selected = active_dimensions isa Integer ? [active_dimensions] : collect(active_dimensions)
    dims = filter(dimension -> dimension in selected, dimensions)
    isempty(dims) && error("None of the selected dimensions occur in the benchmark files.")

    return dims
end

linestyle_for(dimension, dimensions) = length(dimensions) == 1 ? :solid : LINESTYLES[dimension]

################################################################################
# Plot elements
################################################################################
function base_plot(Plotter; ylabel, ylims, yticks, xlims)
    return Plotter.plot(;
        ylabel = ylabel,
        xlabel = L"\mathrm{number\ of\ unknowns\ } N",
        yscale = :log10,
        xscale = :log10,
        ylims = ylims,
        xlims = xlims,
        yticks = yticks,
        xticks = 10.0 .^ (2:8),
        left_margin = 6 * Plotter.px,
        bottom_margin = 3 * Plotter.px,
        legend = :topleft,
        guidefontsize = 16,
        framestyle = :box,
        foreground_color_border = :black,
        linewidth = 2,
        grid = true,
        gridlinewidth = 0.15,
        gridcolor = :black,
        gridalpha = 0.6,
    )
end

function plot_series!(p, Plotter, N, values, color, linestyle)
    return Plotter.plot!(
        p,
        N,
        values;
        linecolor = color,
        linestyle = linestyle,
        marker = :circle,
        markersize = 3,
        markerstrokewidth = 0,
        markercolor = color,
        primary = false,
    )
end

# Custom legend: colors represent algorithms, line styles represent dimensions.
# Algorithms in `skip` keep their color index but get no legend entry.
function add_legend!(p, Plotter, labels, dimensions; skip = Set{String}())
    for (i, label) in enumerate(labels)
        label in skip && continue

        Plotter.plot!(
            p,
            Float64[],
            Float64[];
            color = COLORS[i],
            linestyle = :solid,
            linewidth = 1.5,
            marker = :circle,
            markersize = 3,
            markerstrokewidth = 0,
            label = label,
        )
    end

    # One free line in legend
    Plotter.scatter!(p, [], []; label = " ", ms = 0, mc = :white, msc = :white)

    for dimension in dimensions
        Plotter.plot!(
            p,
            Float64[],
            Float64[];
            color = :black,
            linestyle = linestyle_for(dimension, dimensions),
            linewidth = 1.5,
            label = "$(dimension)D",
        )
    end

    return p
end

# Reference lines factor * N^exponent, given as a vector of (exponent, factor) pairs,
# e.g. [(1, 1.0e-4), (1, 1.0e-7), (2, 1.0e-9)]. Each exponent appears once in the legend.
function add_references!(p, Plotter, references, xlims)
    isempty(references) && return p

    N_ref = 10.0 .^ range(log10(xlims[1] * 1.3), log10(xlims[2] / 1.3), length = 200)

    # One free line in legend
    Plotter.scatter!(p, [], []; label = " ", ms = 0, mc = :white, msc = :white)

    labeled = Float64[]
    for (exponent, factor) in references
        e = Float64(exponent)
        first_occurrence = !any(x -> isapprox(x, e), labeled)

        Plotter.plot!(
            p,
            N_ref,
            factor .* N_ref .^ e;
            color = :black,
            linestyle = reference_linestyle(e),
            linewidth = 1,
            alpha = 0.8,
            label = reference_label(e),
            primary = first_occurrence,
        )

        first_occurrence && push!(labeled, e)
    end

    return p
end

function reference_label(e)
    isapprox(e, 1) && return L"\mathcal{O}(N)"

    r = rationalize(e; tol = 1.0e-6)
    exponent_string = denominator(r) == 1 ? string(numerator(r)) : "$(numerator(r))/$(denominator(r))"

    return latexstring("\\mathcal{O}(N^{", exponent_string, "})")
end

function reference_linestyle(e)
    isapprox(e, 1) && return :dash
    isapprox(e, 2) && return :dot
    isapprox(e, 3 / 2) && return :dashdotdot
    return :dashdot # e.g. 4/3
end

################################################################################
# Runtimes
################################################################################
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

################################################################################
# Iteration counts
################################################################################
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
