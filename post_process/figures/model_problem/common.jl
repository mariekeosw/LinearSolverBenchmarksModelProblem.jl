#=
Shared helpers for the model-problem figures.
Included by fig1_runtimes.jl and fig2_iterations.jl; not meant to be used on its own.
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
            "File $filename does not contain `$field`. " *
                "Files created before iteration counts were recorded have to be regenerated " *
                "with scripts/run_model_problem_benchmark.jl."
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
