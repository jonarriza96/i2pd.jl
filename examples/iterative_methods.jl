using Pkg
Pkg.activate(".")


using LinearAlgebra
using Infiltrator
using Plots

include("../ext/maros_meszaros/load_qp.jl")
include("../src/types.jl")
include("../src/problem.jl")
include("../src/solver.jl")


function load_problem(problem_name)
    qp = load_qp_maros_meszaros(problem_name; split_box=true)
    Q, q, A, b, C, d, c = qp_to_ge_eq_form(qp; use_split_box=true)
    # @infiltrate
    if size(C, 1) > 0
        C, d = reduce_system(C, d)
    end
    rAC = rank([A; C])
    println("n = $(size(Q, 1)), m = $(size(A, 1)), p = $(size(C, 1))")
    println("rows(A, C, A+C) = $(size(A, 1)), $(size(C, 1)), $(size([A; C], 1))")
    # @infiltrate

    return QPProblem(Q, q, A, b, C, d, c)
end

ε = 1e-4
solver_modes = [
    MODE_PARTIALLY_CONDENSED,
    MODE_PARTIALLY_CONDENSED_IMPLICIT,
]
problem_names = [
    "data/CVXQP1_s",
    "data/CONT-050",
    "data/stcqp1",
    "data/mosarqp1",
]

# ----------------------------------- Solve ---------------------------------- #
data = Dict(
    problem_name => Dict{typeof(first(solver_modes)),Any}(
        solver_mode => nothing for solver_mode in solver_modes
    ) for problem_name in problem_names
)

for problem_name in problem_names

    # load problem
    println("\n+++++++++++++ Loading $(problem_name) +++++++++++++\n")
    qp = load_problem(problem_name)

    # solve problems
    for (mode_index, solver_mode) in enumerate(solver_modes)

        solution, e, d = Solver.solver_kernel(qp;
            solver_mode=solver_mode,
            solver_backend=BACKEND_ITERATIVE,
            σ=0.5,
            max_iters=100,
            ls_τ=0.1, # avoids line search failure in CONT-050
            ε_s=ε, ε_p=ε, ε_g=ε, ε_c=ε,
            monitor=Dict(:cond => true, :eigvals => false, :ΔK => true),
            kr_method=:minres,
            kr_precondition=true,
            kr_maxit=60000,
            kr_tol=1e-10,
            kr_ρ=0.0,
            kr_δ=1e-8,
            kr_ε=1e-4,
        )

        # store data
        data[problem_name][solver_mode] = Dict(
            "eigvals" => e, "kr_it" => d[:, 12], "gap" => d[:, 2], "t" => cumsum(d[:, 14]))
    end

end


## --------------------------------- Visualize -------------------------------- #
p_leg = plot(; framestyle=:none, grid=false, xticks=false, yticks=false, legend=:top, legend_column=length(solver_modes))
scatter!(p_leg, [NaN], [NaN], color=1, label="explicit")
scatter!(p_leg, [NaN], [NaN], color=2, label="implicit")

plots = [p_leg]
metrics = [
    ("eigvals", "eigvals(K)", scatter!, nothing, ""),
    ("kr_it", "kr_it", plot!, nothing, "iterations"),
    ("gap", "gap", plot!, "t", "time"),
]

for (row, (key, ylabel, plot_fun, xkey, xlabel)) in enumerate(metrics)
    for (j, problem_name) in enumerate(problem_names)
        p = plot(;
            xlabel=xlabel,
            ylabel=j == 1 ? ylabel : "",
            title=key == "eigvals" ? basename(problem_name) : "",
            yscale=key in ("eigvals", "gap") ? :log10 : :identity,
            tickfontsize=5,
            legend=false,
        )
        for (i, solver_mode) in enumerate(solver_modes)
            values = data[problem_name][solver_mode][key]
            if key == "eigvals"
                all(isnan, values) && continue
                plot_fun(p, values, color=i, alpha=0.05, markerstrokewidth=0, label="")
            elseif isnothing(xkey)
                plot_fun(p, values, color=i, label="")
            else
                plot_fun(p, data[problem_name][solver_mode][xkey], values, color=i, label="")
            end
        end
        key == "gap" && hline!(p, [ε], color=:black, linestyle=:dash, label="")
        push!(plots, p)
    end
end

l = @layout [a{0.08h}; grid(3, length(problem_names))]
display(plot(plots..., layout=l))
