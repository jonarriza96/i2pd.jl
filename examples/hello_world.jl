using Pkg
Pkg.activate(".")


using LinearAlgebra
using SparseArrays

using Plots


include("../ext/maros_meszaros/load_qp.jl")
include("../src/types.jl")
include("../src/problem.jl")
include("../src/solver.jl")


function load_problem(maros_meszaros=false)

    if maros_meszaros

        qp = load_qp_maros_meszaros("data/CVXQP1_s"; split_box=true)
        Q, q, A, b, C, d, c = qp_to_ge_eq_form(qp; use_split_box=true)

    else
        Q = sparse([1.0 0.0;
            0.0 1.0])
        q = [0.0, 0.0]
        c = 0.0

        A = sparse([
            1.0 1.0;
            0.0 1.0;
            -1.0 0.0;
            0.0 -1.0])
        b = [0.65, -0.1, -0.85, -0.8]

        C = spzeros(Float64, 0, 2)
        d = Float64[]

    end

    if size(C, 1) > 0
        C, d = reduce_system(C, d)
    end


    return QPProblem(Q, q, A, b, C, d, c)


end


# ----------------------------------- Solve ---------------------------------- #
eigvals = []
condK = []
ΔK = []
solver_modes = [MODE_PARTIALLY_CONDENSED, MODE_PARTIALLY_CONDENSED_IMPLICIT]

for solver_mode in solver_modes

    qp = load_problem(false)

    solution, e, d = Solver.solver_kernel(qp;
        solver_mode=solver_mode,
        solver_backend=BACKEND_DIRECT,
        σ=0.5,
        max_iters=100,
        monitor=Dict(:cond => true, :eigvals => true, :ΔK => true),
    )

    push!(eigvals, e)
    push!(condK, d[:, 11])
    push!(ΔK, d[:, 15])
end

## --------------------------------- Visualize -------------------------------- #
p_eigs = plot(; xlabel="iterations", ylabel="eigvals(K)", legend=false)
p_condK = plot(; xlabel="iterations", ylabel="cond(K)", legend=false)
p_ΔK = plot(; xlabel="iterations", ylabel="ΔK", legend=false)

for i in 1:length(solver_modes)
    scatter!(p_eigs, eigvals[i], color=i, label="")
    plot!(p_condK, condK[i], color=i, label="")
    plot!(p_ΔK, ΔK[i], color=i, label="")
end

p_leg = plot(; framestyle=:none, grid=false, xticks=false, yticks=false, legend=:top, legend_column=2)
scatter!(p_leg, [NaN], [NaN], color=1, label="explicit")
scatter!(p_leg, [NaN], [NaN], color=2, label="implicit")

l = @layout [a{0.08h}; b; c; d]
display(plot(p_leg, p_eigs, p_condK, p_ΔK, layout=l))