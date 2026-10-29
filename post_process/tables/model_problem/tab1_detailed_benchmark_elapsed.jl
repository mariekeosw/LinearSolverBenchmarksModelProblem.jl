module Tab1DetailedBenchmark

using JLD2
using Printf

# How to use this function?
# 1) includet("post_process/tables/model_problem/tab1_detailed_benchmark_elapsed.jl")
# 2) Tab1DetailedBenchmark.export_benchmark_to_latex("jld2_files_server/sym_model_problem/sym_mp_direct_15_elapsed.jld2", "xxx.tex")
function export_benchmark_to_latex(filename_in, filename_out)

    @load filename_in labels dimensions refinement problem_sizes tests residuals walltimes tlinsolves tlinsolve_setups tlinsolve_solves

    dims = sort(collect(dimensions))

    open(filename_out, "w") do io
        println(io, "% Requires \\usepackage{booktabs}")
        println(io, "% Requires \\usepackage{longtable}")
        println(io)
        println(io, "\\setlength{\\tabcolsep}{3pt}")
        println(io, "\\footnotesize")
        println(io)
        println(io, "\\begin{longtable}{l rrr rrr rrr}")
        println(io, "\\caption{Detailed benchmark results corresponding to Figure~\\ref{xx}. Reported runtimes were obtained using Julia's built-in \\texttt{@elapsed} macro}")
        println(io, "\\label{tab:xx}\\\\")
        println(io)

        print_header(io)
        println(io, "\\endfirsthead")
        print_header(io)
        println(io, "\\endhead")
        println(io, "\\midrule")
        println(io, "\\multicolumn{10}{r}{\\scriptsize Continued on next page} \\\\")
        println(io, "\\endfoot")
        println(io, "\\bottomrule")
        println(io, "\\endlastfoot")

        for (k, iref) in enumerate(eachindex(refinement))
            println(io, "% --- Block $k ---")

            print(io, "\\multicolumn{1}{l}{\\footnotesize Refinement $(refinement[iref])}")

            for dim in dims
                N = problem_size_for_block(dim, labels, iref, problem_sizes)
                print(io, " & \\multicolumn{3}{c}{\\footnotesize \$N = $(format_int_latex(N))\$}")
            end

            println(io, " \\\\")
            println(io, "\\midrule")

            for label in labels
                method = latex_escape(label)
                print(io, method)

                for dim in dims
                    key = (dim, label)

                    if !haskey(problem_sizes, key) || iref > length(problem_sizes[key])
                        print(io, " & -- & -- & --")
                    else
                        wall = walltimes[key][iref]
                        total = tlinsolves[key][iref]
                        setup = tlinsolve_setups[key][iref]
                        solve = tlinsolve_solves[key][iref]
                        test = tests[key][iref]

                        print(
                            io,
                            " & ",
                            format_runtime(wall),
                            " & ",
                            format_runtime(total),
                            " {\\scriptsize (",
                            format_runtime(setup),
                            " / ",
                            format_runtime(solve),
                            ")} & ",
                            format_test(test)
                        )
                    end
                end

                println(io, " \\\\")
            end

            if k != length(refinement)
                println(io, "\\midrule")
                println(io, "\\addlinespace")
            end
        end

        println(io, "\\end{longtable}")
        println(io, "\\normalsize")
    end

    return nothing
end


function print_header(io)
    println(io, "\\toprule & \\multicolumn{3}{c}{1D} & \\multicolumn{3}{c}{2D} & \\multicolumn{3}{c}{3D} \\\\")
    println(io, "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}")
    println(io, "Method & Wall [s] & Linear [s] & Test & Wall [s] & Linear [s] & Test & Wall [s] & Linear [s] & Test \\\\")
    return println(io, "\\midrule")
end


function problem_size_for_block(dim, labels, iref, problem_sizes)
    for label in labels
        key = (dim, label)

        if haskey(problem_sizes, key) && iref <= length(problem_sizes[key])
            return problem_sizes[key][iref]
        end
    end

    return missing
end


function format_runtime(x)
    if ismissing(x)
        return "--"
    elseif isnan(x)
        return "NaN"
    else
        return @sprintf("%.2f", x)
    end
end


function format_test(x)
    if ismissing(x)
        return "--"
    elseif isnan(x)
        return "NaN"
    else
        return @sprintf("%.1f", x)
    end
end


function format_int_latex(x)
    if ismissing(x)
        return "--"
    end

    s = string(Int(x))
    parts = String[]

    while length(s) > 3
        pushfirst!(parts, s[(end - 2):end])
        s = s[1:(end - 3)]
    end

    pushfirst!(parts, s)
    return join(parts, "\\,")
end


function latex_escape(s::AbstractString)
    s = replace(s, "\\" => "\\textbackslash{}")
    s = replace(s, "_" => "\\_")
    s = replace(s, "&" => "\\&")
    s = replace(s, "%" => "\\%")
    s = replace(s, "#" => "\\#")
    s = replace(s, "{" => "\\{")
    s = replace(s, "}" => "\\}")
    return s
end

end # module
