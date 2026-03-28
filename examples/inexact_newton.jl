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

θs = [0.0, 0.0, 0.1, 0.6, 1.0]
ε = 1e-4
problem_names = [
    "data/CVXQP1_s",
    "data/CONT-050",
    "data/stcqp1",
    "data/mosarqp1",
]

# ----------------------------------- Solve ---------------------------------- #
data = Dict(
    problem_name => Vector{NamedTuple{(:θ, :gap, :t, :refacts),Tuple{Float64,Vector,Vector,Float64}}}() for problem_name in problem_names
)


for problem_name in problem_names

    # load problem
    println("\n+++++++++++++ Loading $(problem_name) +++++++++++++\n")
    qp = load_problem(problem_name)

    # solve problems
    for θ in θs

        println("  Solving with θ = $(θ)")
        solution, _, d = Solver.solver_kernel(qp;
            solver_mode=MODE_PARTIALLY_CONDENSED_IMPLICIT,
            solver_backend=BACKEND_DIRECT,
            σ=0.8, # given that we are reusing factorizations, lets be careful
            max_iters=300,
            inexact_newton=true,
            in_θ=θ,
            ε_s=ε, ε_p=ε, ε_g=ε, ε_c=ε)

        # store data
        push!(data[problem_name], (
            θ=θ,
            gap=d[:, 2],
            t=cumsum(d[:, 14]),
            refacts=sum(d[:, 13]),
        ))
    end

end


## --------------------------------- Visualize -------------------------------- #
plot_runs = θs[2:end]

p_leg = plot(; framestyle=:none, grid=false, xticks=false, yticks=false, legend=:top, legend_column=length(plot_runs))
for (i, θ) in enumerate(plot_runs)
    scatter!(p_leg, [NaN], [NaN], color=i, label="θ = $(θ)")
end

plots = [p_leg]
gap_plots = Plots.Plot[]
refact_plots = Plots.Plot[]

for (j, problem_name) in enumerate(problem_names)
    p_gap = plot(;
        xlabel="time",
        ylabel=j == 1 ? "gap" : "",
        title=basename(problem_name),
        yscale=:log10,
        tickfontsize=5,
        legend=false,
    )
    for (i, run) in enumerate(data[problem_name][2:end])
        plot!(p_gap, run.t, run.gap, color=i, label="")
    end
    hline!(p_gap, [ε], color=:black, linestyle=:dash, label="")
    push!(gap_plots, p_gap)

    p_refacts = plot(;
        xlabel="θ",
        ylabel=j == 1 ? "refacts" : "",
        tickfontsize=5,
        legend=false,
    )
    θ_vals = [run.θ for run in data[problem_name][2:end]]
    refact_vals = [run.refacts for run in data[problem_name][2:end]]
    plot!(
        p_refacts,
        θ_vals,
        refact_vals,
        color=:black,
        label="",
    )
    for (i, (θ, refacts)) in enumerate(zip(θ_vals, refact_vals))
        scatter!(p_refacts, [θ], [refacts], color=i, marker=:circle, label="")
    end
    push!(refact_plots, p_refacts)
end

append!(plots, gap_plots)
append!(plots, refact_plots)

l = @layout [a{0.08h}; grid(2, length(problem_names))]
display(plot(plots..., layout=l))
