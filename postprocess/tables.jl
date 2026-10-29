module Tables

#=
LaTeX table (longtable) with linear solver runtimes and iteration counts of one result file.

Usage:
    include("postprocess/tables.jl")
    Tables.export_table("results/sym_direct.jld2", "sym_direct.tex")

Requires \usepackage{booktabs} and \usepackage{longtable}. Files without iteration counts
(results created before they were recorded) give runtime columns only.
=#

using JLD2
using Printf

function export_table(filename_in, filename_out)
    data = JLD2.load(filename_in)
    labels, refinement = data["labels"], data["refinement"]
    dims = sort(collect(data["dimensions"]))
    problem_sizes, tlinsolves = data["problem_sizes"], data["tlinsolves"]
    niters = get(data, "niters", nothing)

    ncols = isnothing(niters) ? 1 : 2

    open(filename_out, "w") do io
        println(io, "\\setlength{\\tabcolsep}{6pt}\n\\footnotesize")
        println(io, "\\begin{longtable}{l ", repeat("r ", ncols * length(dims)), "}")
        println(io, "\\caption{Linear solver runtimes (setup + solve) obtained with \\texttt{@belapsed}", isnothing(niters) ? "" : " and Krylov iterations", ".}")
        println(io, "\\label{tab:xx}\\\\")

        header = "\\toprule" * join([" & \\multicolumn{$ncols}{c}{$(dim)D}" for dim in dims]) * " \\\\\n" *
            "Method" * repeat(isnothing(niters) ? " & Time [s]" : " & Time [s] & It.", length(dims)) * " \\\\\n\\midrule"
        println(io, header, "\n\\endfirsthead\n", header, "\n\\endhead")
        println(io, "\\bottomrule\n\\endlastfoot")

        for iref in eachindex(refinement)
            print(io, "Refinement $(refinement[iref])")
            for dim in dims
                print(io, " & \\multicolumn{$ncols}{c}{\$N = $(format_int(problem_size(dim, labels, iref, problem_sizes)))\$}")
            end
            println(io, " \\\\\n\\midrule")

            for label in labels
                print(io, replace(label, "_" => "\\_"))
                for dim in dims
                    key = (dim, label)
                    available = haskey(tlinsolves, key) && iref <= length(tlinsolves[key])
                    print(io, " & ", available ? format_runtime(tlinsolves[key][iref]) : "--")
                    isnothing(niters) || print(io, " & ", available && niters[key][iref] > 0 ? string(niters[key][iref]) : "--")
                end
                println(io, " \\\\")
            end

            iref == lastindex(refinement) || println(io, "\\midrule\n\\addlinespace")
        end

        println(io, "\\end{longtable}\n\\normalsize")
    end

    return nothing
end

# Problem size of a refinement level (the same for all algorithms; 0 marks failed runs)
function problem_size(dim, labels, iref, problem_sizes)
    for label in labels
        key = (dim, label)
        if haskey(problem_sizes, key) && iref <= length(problem_sizes[key]) && problem_sizes[key][iref] > 0
            return problem_sizes[key][iref]
        end
    end
    return missing
end

format_runtime(x) = isnan(x) ? "--" : @sprintf("%.3e", x)

# 1234567 -> 1\,234\,567
format_int(x) = ismissing(x) ? "--" : replace(string(Int(x)), r"(?<=\d)(?=(\d{3})+$)" => "\\,")

end # module
