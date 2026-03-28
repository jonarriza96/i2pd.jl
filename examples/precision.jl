using Pkg
Pkg.activate(".")


using LinearAlgebra
using Infiltrator
using Plots

include("../ext/maros_meszaros/load_qp.jl")
include("../src/types.jl")
include("../src/problem.jl")
include("../src/solver.jl")


function load_problem(problem_name, precision)
    qp = load_qp_maros_meszaros(problem_name; split_box=true)
    Q, q, A, b, C, d, c = qp_to_ge_eq_form(qp; use_split_box=true)

    if size(C, 1) > 0
        C, d = reduce_system(C, d)
    end
    rAC = rank([A; C])
    println("n = $(size(Q, 1)), m = $(size(A, 1)), p = $(size(C, 1))")
    println("rows(A, C, A+C) = $(size(A, 1)), $(size(C, 1)), $(size([A; C], 1))")
    # @infiltrate

    # convert
    if precision == Float32
        Q = Matrix{Float32}(Q)
        q = Vector{Float32}(q)
        A = Matrix{Float32}(A)
        b = Vector{Float32}(b)
        C = Matrix{Float32}(C)
        d = Vector{Float32}(d)
        c = Float32(c)
    elseif precision == Float64
        Q = Matrix{Float64}(Q)
        q = Vector{Float64}(q)
        A = Matrix{Float64}(A)
        b = Vector{Float64}(b)
        C = Matrix{Float64}(C)
        d = Vector{Float64}(d)
        c = Float64(c)
    end

    return QPProblem(Q, q, A, b, C, d, c)
end

ε = 1e-20 # we let it stall to see which precision we can reach
precisions = [Float64, Float32]
solver_modes = [MODE_PARTIALLY_CONDENSED, MODE_PARTIALLY_CONDENSED_IMPLICIT]
problem_names = [
    "data/CVXQP1_s",
    # "data/stcqp1", # takes a long time
]

# ----------------------------------- Solve ---------------------------------- #
data = Dict(
    problem_name => Dict(
        solver_mode => Dict(
            precision => Vector{Float64}() for precision in precisions
        ) for solver_mode in [MODE_PARTIALLY_CONDENSED_IMPLICIT, MODE_PARTIALLY_CONDENSED]
    ) for problem_name in problem_names
)


for problem_name in problem_names
    for solver_mode in solver_modes
        for precision in precisions

            # load problem
            println("\n+++++++++++++ Loading $(problem_name) with $(precision) +++++++++++++\n")
            qp = load_problem(problem_name, precision)

            # solve problem
            solution, _, d = Solver.solver_kernel(qp;
                solver_mode=solver_mode,
                solver_backend=BACKEND_DIRECT,
                σ=0.5,
                max_iters=100,
                ls_αm=0.9 * ε, # avoid line search failures to let it stall
                ε_s=ε, ε_p=ε, ε_g=ε, ε_c=ε)

            # store data
            max_res = max.(d[:, 5], d[:, 6], d[:, 7])
            data[problem_name][solver_mode][precision] = max_res
        end
    end

end

## --------------------------------- Visualize -------------------------------- #
p_leg = plot(; framestyle=:none, grid=false, xticks=false, yticks=false, legend=:top, legend_column=4)
plot!(p_leg, [NaN], [NaN]; color=1, linestyle=:solid, label="explicit (f64)")
plot!(p_leg, [NaN], [NaN]; color=2, linestyle=:solid, label="implicit (f64)")
plot!(p_leg, [NaN], [NaN]; color=1, linestyle=:dash, label="explicit (f32)")
plot!(p_leg, [NaN], [NaN]; color=2, linestyle=:dash, label="implicit (f32)")

plots = [p_leg]

for (j, problem_name) in enumerate(problem_names)
    p = plot(;
        xlabel="iterations",
        ylabel=j == 1 ? "max residual" : "",
        title=basename(problem_name),
        yscale=:log10,
        legend=false,
    )

    for precision in precisions
        for solver_mode in [MODE_PARTIALLY_CONDENSED, MODE_PARTIALLY_CONDENSED_IMPLICIT]
            color = solver_mode == MODE_PARTIALLY_CONDENSED ? 1 : 2
            linestyle = precision == Float64 ? :solid : :dash
            label = ""
            plot!(
                p,
                data[problem_name][solver_mode][precision];
                color=color,
                linestyle=linestyle,
                label=label,
            )
        end
    end

    push!(plots, p)
end

l = @layout [a{0.08h}; grid(1, length(problem_names))]
display(plot(plots..., layout=l))