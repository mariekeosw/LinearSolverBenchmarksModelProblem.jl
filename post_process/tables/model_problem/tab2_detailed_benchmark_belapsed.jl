module Tab2DetailedBenchmark

using JLD2
using Printf

# How to use this function?
# 1) includet("post_process/tables/model_problem/tab2_detailed_benchmark_belapsed.jl")
# 2) Tab2DetailedBenchmark.export_benchmark_to_latex("jld2_files_server/sym_model_problem/sym_mp_direct_15_belapsed.jld2", "xxx.tex")
function export_benchmark_to_latex(filename_in, filename_out)

    @load filename_in labels dimensions refinement problem_sizes tests residuals walltimes tlinsolves tlinsolve_setups tlinsolve_solves

    dims = sort(collect(dimensions))

    open(filename_out, "w") do io
        println(io, "% Requires \\usepackage{booktabs}")
        println(io, "% Requires \\usepackage{longtable}")
        println(io)
        println(io, "\\setlength{\\tabcolsep}{6pt}")
        println(io, "\\footnotesize")
        println(io)
        println(io, "\\begin{longtable}{l r r r}")
        println(io, "\\caption{Model Problem Benchmark: Linear solver runtimes obtained with \\texttt{@belapsed}.}")
        println(io, "\\label{tab:benchmark-mp-final-belapsed}\\\\")
        println(io)

        print_header(io)
        println(io, "\\endfirsthead")
        print_header(io)
        println(io, "\\endhead")
        println(io, "\\midrule")
        println(io, "\\multicolumn{4}{r}{\\scriptsize Continued on next page} \\\\")
        println(io, "\\endfoot")
        println(io, "\\bottomrule")
        println(io, "\\endlastfoot")

        for (k, iref) in enumerate(eachindex(refinement))
            println(io, "% --- Block $k ---")

            print(io, "\\multicolumn{1}{l}{\\footnotesize Refinement $(refinement[iref])}")

            for dim in dims
                N = problem_size_for_block(dim, labels, iref, problem_sizes)
                print(io, " & \\multicolumn{1}{c}{\\footnotesize \$N = $(format_int_latex(N))\$}")
            end

            println(io, " \\\\")
            println(io, "\\midrule")

            for label in labels
                method = latex_escape(label)
                print(io, method)

                for dim in dims
                    key = (dim, label)

                    if !haskey(tlinsolves, key) || iref > length(tlinsolves[key])
                        print(io, " & --")
                    else
                        total = tlinsolves[key][iref]
                        print(io, " & ", format_runtime_belapsed(total))
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
    println(io, "\\toprule & \\multicolumn{1}{c}{1D} & \\multicolumn{1}{c}{2D} & \\multicolumn{1}{c}{3D} \\\\")
    println(io, "\\cmidrule(lr){2-2} \\cmidrule(lr){3-3} \\cmidrule(lr){4-4}")
    println(io, "Method & Linear [s] & Linear [s] & Linear [s] \\\\")
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


function format_runtime_belapsed(x)
    if ismissing(x)
        return "--"
    elseif isnan(x)
        return "--"
    else
        return @sprintf("%.3e", x)
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
