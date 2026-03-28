module Solver

using LinearAlgebra
using SparseArrays
using LinearOperators
using Infiltrator
using Printf
using Krylov
using Plots

using Main: QPProblem
using Main: Solution
using Main: STATUS_UNKNOWN, STATUS_CONVERGED, STATUS_MAX_ITER, STATUS_LINESEARCH_FAILED
using Main: SolverMode, MODE_UNCONDENSED, MODE_PARTIALLY_CONDENSED, MODE_FULLY_CONDENSED, MODE_UNCONDENSED_IMPLICIT, MODE_PARTIALLY_CONDENSED_IMPLICIT
using Main: SolverBackend, BACKEND_DIRECT, BACKEND_ITERATIVE
using Main: StateIndices
using Main: RetractionMap, RETRACTION_MAP_SOFTPLUS, RETRACTION_MAP_EXP
using Main: ruiz_scale


is_sparse_problem(problem::QPProblem) = issparse(problem.Q) || issparse(problem.A)

maybe_sparse(M, use_sparse) = use_sparse ? sparse(M) : M
diag_matrix(v, use_sparse) = use_sparse ? spdiagm(0 => collect(v)) : Diagonal(collect(v))
identity_matrix(T, n, use_sparse) = use_sparse ? spdiagm(0 => ones(T, n)) : Diagonal(ones(T, n))
zero_matrix(T, m, n, use_sparse) = use_sparse ? spzeros(T, m, n) : zeros(T, m, n)

function primal_cost(x, prob::QPProblem)
    return 0.5 * dot(x, prob.Q, x) + dot(prob.q, x) + prob.c
end

function dual_cost(λ, prob::QPProblem, Qinv)
    # x_dual = Qchol \ (prob.A' * λ - prob.q)
    # x_dual = Qinv * (prob.A' * λ - prob.q)
    x_dual = (prob.Q + 1e-4 * I(prob.n)) \ (prob.A' * λ - prob.q)

    return -0.5 * dot(x_dual, prob.Q, x_dual) + dot(λ, prob.b) #! +c constant ignored
end

# ------------------------------ complementarity ----------------------------- #
@inline function rm(x::T, β::T, retraction_map::RetractionMap) where {T<:Real}
    if retraction_map == RETRACTION_MAP_SOFTPLUS
        return rm_sp(x, β)
    elseif retraction_map == RETRACTION_MAP_EXP
        return rm_exp(x, β)
    else
        @error "Invalid retraction map"
        return nothing
    end
end

@inline function rm_inv(y::T, β::T, retraction_map::RetractionMap) where {T<:Real}
    if retraction_map == RETRACTION_MAP_SOFTPLUS
        return rm_inv_sp(y, β)
    elseif retraction_map == RETRACTION_MAP_EXP
        return rm_inv_exp(y, β)
    else
        @error "Invalid retraction map"
        return nothing
    end
end

@inline function drm(x::T, β::T, retraction_map::RetractionMap) where {T<:Real}
    if retraction_map == RETRACTION_MAP_SOFTPLUS
        return drm_sp(x, β)
    elseif retraction_map == RETRACTION_MAP_EXP
        return drm_exp(x, β)
    else
        @error "Invalid retraction map"
        return nothing
    end
end


@inline function rm_sp(x::T, β::T) where {T<:Real}
    return (x + sqrt(x * x + T(4) * β)) / T(2)
end

@inline function rm_inv_sp(y::T, β::T) where {T<:Real}
    return y - β / y
end

@inline function drm_sp(x::T, β::T) where {T<:Real}
    t = sqrt(x * x + T(4) * β)
    return (one(T) + x / t) / T(2)
end

@inline function rm_exp(x::T, β::T) where {T<:Real}
    return sqrt(β) * exp(x)
end

@inline function rm_inv_exp(y::T, β::T) where {T<:Real}
    return log(y / sqrt(β))
end

@inline function drm_exp(x::T, β::T) where {T<:Real}
    return sqrt(β) * exp(x)
end

# -------------------------------- kkt system -------------------------------- #
function kkt_matrix(z, problem::QPProblem, solver_mode::SolverMode, retraction_map::RetractionMap; β=0.0)
    if solver_mode == MODE_UNCONDENSED
        return kkt_matrix_uncondensed(z, problem)
    elseif solver_mode == MODE_PARTIALLY_CONDENSED
        return kkt_matrix_partially_condensed(z, problem)
    elseif solver_mode == MODE_FULLY_CONDENSED
        return kkt_matrix_fully_condensed(z, problem)
    elseif solver_mode == MODE_UNCONDENSED_IMPLICIT
        return kkt_matrix_uncondensed_implicit(z, β, problem)
    elseif solver_mode == MODE_PARTIALLY_CONDENSED_IMPLICIT
        return kkt_matrix_partially_condensed_implicit(z, β, problem, retraction_map)
    else
        @error "Invalid solver mode"
        return nothing
    end
end

function kkt_matrix!(K, z, problem::QPProblem, solver_mode::SolverMode, retraction_map::RetractionMap; β=0.0)
    if solver_mode == MODE_UNCONDENSED
        return kkt_matrix_uncondensed!(K, z, problem)
    elseif solver_mode == MODE_PARTIALLY_CONDENSED
        return kkt_matrix_partially_condensed!(K, z, problem)
    elseif solver_mode == MODE_FULLY_CONDENSED
        return kkt_matrix_fully_condensed!(K, z, problem)
    elseif solver_mode == MODE_UNCONDENSED_IMPLICIT
        return kkt_matrix_uncondensed_implicit!(K, z, β, problem)
    elseif solver_mode == MODE_PARTIALLY_CONDENSED_IMPLICIT
        return kkt_matrix_partially_condensed_implicit!(K, z, β, problem, retraction_map)
    else
        @error "Invalid solver mode"
        return nothing
    end
end

function kkt_matrix_uncondensed(z, problem::QPProblem)
    n = problem.n
    m = problem.m
    p = problem.p
    Q = problem.Q
    A = problem.A
    C = problem.C
    d = problem.d
    T = eltype(Q)

    # Extract λ and s from the state vector z
    x = z[1:n]
    λ = z[n+1:n+m]
    γ = z[n+1:n+m+p]
    s = z[n+m+p+1:n+m+p+m]
    use_sparse = is_sparse_problem(problem)

    # Eq. 6: S = diag(s), Λ = diag(λ)
    S = diag_matrix(T.(s), use_sparse)
    Λ = diag_matrix(T.(λ), use_sparse)

    Id_m = identity_matrix(T, m, use_sparse)

    # Existing zero matrices
    z_nm = zero_matrix(T, n, m, use_sparse)
    z_mm = zero_matrix(T, m, m, use_sparse)
    z_mn = zero_matrix(T, m, n, use_sparse)

    # New zero matrices for the equality constraints (p)
    z_mp = zero_matrix(T, m, p, use_sparse)
    z_pm = zero_matrix(T, p, m, use_sparse)
    z_pp = zero_matrix(T, p, p, use_sparse)

    # Matrix blocks
    Q_block = maybe_sparse(Q, use_sparse)
    A_block = maybe_sparse(A, use_sparse)
    C_block = maybe_sparse(C, use_sparse) # New C block

    # Building the matrix from updated Eq. 6
    K = [
        Q_block -A_block' -C_block' z_nm;
        A_block z_mm z_mp -Id_m;
        C_block z_pm z_pp z_pm;
        z_mn S z_mp Λ
    ]

    return K
end
function kkt_matrix_uncondensed!(K, z, problem::QPProblem)
    n = problem.n
    m = problem.m
    p = problem.p

    @views begin
        λ = z[n+1:n+m]
        s = z[n+m+p+1:n+m+p+m]
    end

    @inbounds for i in 1:m
        K[n+m+p+i, n+i] = s[i]
        K[n+m+p+i, n+m+p+i] = λ[i]
    end

    return K
end

function kkt_matrix_partially_condensed(z, problem::QPProblem)
    # 1. Unpack p and C
    n, m, p = problem.n, problem.m, problem.p
    Q, A, C = problem.Q, problem.A, problem.C
    T = eltype(Q)
    use_sparse = is_sparse_problem(problem)

    # 2. Shift s extraction by p
    x = z[1:n]
    λ = z[n+1:n+m]
    s = z[n+m+p+1:n+m+p+m]

    D = diag_matrix(T.(-(s ./ λ)), use_sparse)

    Q_block = maybe_sparse(Q, use_sparse)
    A_block = maybe_sparse(A, use_sparse)
    C_block = maybe_sparse(C, use_sparse)

    # 3. Create padding matrices for the γ columns/rows
    z_mp = zero_matrix(T, m, p, use_sparse)
    z_pm = zero_matrix(T, p, m, use_sparse)
    z_pp = zero_matrix(T, p, p, use_sparse)

    # 4. Build the expanded symmetric matrix
    K = [
        Q_block -A_block' -C_block';
        -A_block D z_mp;
        -C_block z_pm z_pp
    ]

    return K
end
function kkt_matrix_partially_condensed!(K, z, problem::QPProblem)
    n, m, p = problem.n, problem.m, problem.p

    @views begin
        λ = z[n+1:n+m]
        s = z[n+m+p+1:n+m+p+m]
    end

    @inbounds for i in 1:m
        K[n+i, n+i] = -s[i] / λ[i]
    end

    return K
end


function kkt_matrix_fully_condensed(z, problem::QPProblem)
    # 1. Unpack p and C
    n, m, p = problem.n, problem.m, problem.p
    Q, A, C = problem.Q, problem.A, problem.C
    T = eltype(Q)
    use_sparse = is_sparse_problem(problem)

    # 2. Extract variables
    x = z[1:n]
    λ = z[n+1:n+m]
    s = z[n+m+p+1:n+m+p+m]

    # 3. Compute W = S^{-1} Λ
    W_vec = T.(λ ./ s)
    W_mat = diag_matrix(W_vec, use_sparse)

    Q_block = maybe_sparse(Q, use_sparse)
    A_block = maybe_sparse(A, use_sparse)
    C_block = maybe_sparse(C, use_sparse)

    # 4. Compute the condensed top-left block: Q + Aᵀ * W * A
    H_block = Q_block + Symmetric(A_block' * W_mat * A_block)

    # 5. Create padding matrix for the γ block
    z_pp = zero_matrix(T, p, p, use_sparse)

    # 6. Build the fully condensed matrix
    K = [
        H_block -C_block';
        -C_block z_pp
    ]

    return K
end
function kkt_matrix_fully_condensed!(K, z, problem::QPProblem)
    n, m, p = problem.n, problem.m, problem.p
    Q, A = problem.Q, problem.A

    @views begin
        λ = z[n+1:n+m]
        s = z[n+m+p+1:n+m+p+m]
    end

    # Calculate W = λ ./ s
    W_vec = λ ./ s

    # Recompute the top-left n x n block. 
    # (W_vec .* A) efficiently broadcasts W_vec across the rows of A
    H_new = Q + Symmetric(A' * (W_vec .* A))

    # Update the top-left block of K in-place
    K[1:n, 1:n] .= H_new

    return K
end

function kkt_matrix_uncondensed_implicit(z, β, problem::QPProblem)
    n = problem.n
    m = problem.m
    p = problem.p
    Q = problem.Q
    A = problem.A
    C = problem.C
    T = eltype(Q)

    # Extract variables from the state vector z
    # z = [x; λ; γ; s; v]
    # x = z[1:n] (not explicitly needed for the matrix itself)
    # λ = z[n+1:n+m]
    # γ = z[n+m+1:n+m+p] # Corrected index
    # s = z[n+m+p+1:n+m+p+m]
    v = z[n+m+p+m+1:n+m+p+2m]

    use_sparse = is_sparse_problem(problem)

    # Calculate retraction map derivatives (replace with your actual functions)
    # Based on Eq. 11a and 11b: db_β(v) and db_β(-v)
    b_plus_vals = drm.(v, β)
    b_minus_vals = drm.(-v, β)

    # Eq. 12: B⁺_β(v) = diag(db_β(v)), B⁻_β(v) = diag(db_β(-v))
    B_plus = diag_matrix(T.(b_plus_vals), use_sparse)
    B_minus = diag_matrix(T.(b_minus_vals), use_sparse)

    Id_m = identity_matrix(T, m, use_sparse)

    # Zero matrices
    z_nm = zero_matrix(T, n, m, use_sparse)
    z_mm = zero_matrix(T, m, m, use_sparse)
    z_mn = zero_matrix(T, m, n, use_sparse)
    z_mp = zero_matrix(T, m, p, use_sparse)
    z_pm = zero_matrix(T, p, m, use_sparse)
    z_pp = zero_matrix(T, p, p, use_sparse)

    # Matrix blocks
    Q_block = maybe_sparse(Q, use_sparse)
    A_block = maybe_sparse(A, use_sparse)
    C_block = maybe_sparse(C, use_sparse)

    # Building the matrix from Eq. 12
    K = [
        Q_block -A_block' -C_block' z_nm z_nm;
        A_block z_mm z_mp -Id_m z_mm;
        C_block z_pm z_pp z_pm z_pm;
        z_mn Id_m z_mp z_mm -B_plus;
        z_mn z_mm z_mp Id_m B_minus
    ]

    return K
end
function kkt_matrix_uncondensed_implicit!(K, z, β, problem::QPProblem)
    n = problem.n
    m = problem.m
    p = problem.p

    @views begin
        # Extract v vector from the end of z
        v = z[n+m+p+m+1:n+m+p+2m]
    end

    # Calculate retraction map derivatives (replace with your actual functions)
    b_plus_vals = drm.(v, β)
    b_minus_vals = drm.(-v, β)

    # The only blocks in Eq. 12 that depend on the state variables 
    # are -B⁺_β(v) and B⁻_β(v) in the final column.
    @inbounds for i in 1:m
        # Update -B⁺_β(v) block (Row 4, Column 5)
        K[n+m+p+i, n+m+p+m+i] = -b_plus_vals[i]

        # Update B⁻_β(v) block (Row 5, Column 5)
        K[n+m+p+m+i, n+m+p+m+i] = b_minus_vals[i]
    end

    return K
end

function kkt_matrix_partially_condensed_implicit(z, β, problem::QPProblem, retraction_map::RetractionMap)
    n = problem.n
    m = problem.m
    p = problem.p
    Q = problem.Q
    A = problem.A
    C = problem.C
    T = eltype(Q)

    # Extract v from the state vector z
    # z = [x; λ; γ; s; v]
    v = z[n+m+p+m+1:n+m+p+2m]

    use_sparse = is_sparse_problem(problem)

    # Calculate retraction map derivatives
    b_minus_vals = drm.(-v, β, retraction_map)
    B_minus = diag_matrix(T.(b_minus_vals), use_sparse)
    w = drm.(v, β, retraction_map) + drm.(-v, β, retraction_map)
    W = diag_matrix(T.(w), use_sparse)


    # Zero matrices for empty blocks
    z_mp = zero_matrix(T, m, p, use_sparse)
    z_pm = zero_matrix(T, p, m, use_sparse)
    z_pp = zero_matrix(T, p, p, use_sparse)

    # Matrix blocks
    Q_block = maybe_sparse(Q, use_sparse)
    A_block = maybe_sparse(A, use_sparse)
    C_block = maybe_sparse(C, use_sparse)

    # Building the matrix from Eq. 17
    K = [
        Q_block-A_block'*A_block -A_block'*W -C_block';
        -A_block -B_minus z_mp;
        -C_block z_pm z_pp
    ]

    return K
end

function kkt_matrix_partially_condensed_implicit!(K, z, β, problem::QPProblem, retraction_map::RetractionMap)
    n = problem.n
    m = problem.m
    p = problem.p

    # Assuming problem.A is accessible. 
    A = problem.A

    @views begin
        # Extract v vector from the end of z (assumes z = [x; λ; γ; s; v])
        v = z[n+m+p+m+1:n+m+p+2m]
    end

    # Calculate retraction map derivatives for -v
    b_minus_vals = drm.(-v, β, retraction_map)

    # General Case: W = B_μ(v) + B_μ(-v) ≠ I
    # We must update the block (1, 2) which is -Aᵀ * W
    if retraction_map == RETRACTION_MAP_EXP
        b_plus_vals = drm.(v, β, retraction_map)

        @inbounds for j in 1:m
            w_j = b_plus_vals[j] + b_minus_vals[j]

            # Update the (1, 2) block spanning rows 1:n and columns (n+1):(n+m)
            for i in 1:n
                # K[i, n+j] maps to the i-th row, j-th column of -Aᵀ * W
                # mathematically this is -Aᵀ[i, j] * w_j  =>  -A[j, i] * w_j
                K[i, n+j] = -A[j, i] * w_j
            end
        end
    end

    # Update the diagonal of block (2, 2): -B_μ(-v)
    @inbounds for j in 1:m
        K[n+j, n+j] = -b_minus_vals[j]
    end

    return K
end

# ------------------------------ preconditioner ------------------------------ #
function preconditioner(z, problem::QPProblem, solver_mode::SolverMode, retraction_map::RetractionMap; β=0.0, ρ=0.0, δ=1e-8, ε=1e-4, exact_chol=false)
    if solver_mode == MODE_PARTIALLY_CONDENSED
        return preconditioner_partially_condensed(z, problem; ρ=ρ, δ=δ, ε=ε, exact_chol=exact_chol)
    elseif solver_mode == MODE_FULLY_CONDENSED
        return preconditioner_fully_condensed(z, problem; ρ=ρ, ε=ε)
    elseif solver_mode == MODE_PARTIALLY_CONDENSED_IMPLICIT
        return preconditioner_partially_condensed_implicit(z, problem, retraction_map; β=β, ρ=ρ, δ=δ, ε=ε)
    else
        error("Invalid solver mode")
    end
end

function preconditioner_partially_condensed(z, problem::QPProblem; ρ=0.0, δ=1e-8, ε=1e-4, exact_chol=false)
    # 1. Unpack dimensions and variables
    n, m, p = problem.n, problem.m, problem.p
    Q = problem.Q
    T = eltype(Q)
    total_size = n + m + p

    # Extract λ and s
    λ = z[n+1:n+m]
    s = z[n+m+p+1:n+m+p+m]

    # Precompute the diagonal scales for blocks 2 and 3
    d2 = 1.0 ./ ((s ./ λ) .+ δ)

    # Branch based on the desired preconditioner type
    if exact_chol
        # --- EXPENSIVE / EXACT OPTION ---
        F_chol = cholesky(Q + ρ * I)

        function apply_exact!(res, v)
            res[1:n] .= F_chol \ @view(v[1:n])
            res[n+1:n+m] .= d2 .* @view(v[n+1:n+m])
            if p > 0
                res[n+m+1:end] .= (1.0 / ε) .* @view(v[n+m+1:end])
            end
            return res
        end

        return LinearOperator(T, total_size, total_size, true, true, (res, v) -> apply_exact!(res, v))

    else
        # --- CHEAP / DIAGONAL OPTION ---
        F_diag = 1.0 ./ (max.(diag(Q), 0.0) .+ ρ)

        function apply_diag!(res, v)
            res[1:n] .= F_diag .* @view(v[1:n])
            res[n+1:n+m] .= d2 .* @view(v[n+1:n+m])
            if p > 0
                res[n+m+1:end] .= (1.0 / ε) .* @view(v[n+m+1:end])
            end
            return res
        end

        return LinearOperator(T, total_size, total_size, true, true, (res, v) -> apply_diag!(res, v))
    end
end

function preconditioner_fully_condensed(z, problem::QPProblem; ρ=0.0, ε=1e-4)
    # 1. Unpack dimensions and variables
    n, m, p = problem.n, problem.m, problem.p
    Q, A = problem.Q, problem.A
    T = eltype(Q)

    # We only need λ and s to form the condensed H block
    @views begin
        λ = z[n+1:n+m]
        s = z[n+m+p+1:n+m+p+m]
    end

    # 2. Compute W = S^{-1} Λ and the condensed (1,1) block H
    W_vec = λ ./ s
    H = Q + A' * (W_vec .* A)

    # 3. Factorize the (1,1) block
    # Wrapping in Symmetric is crucial here to ensure Cholesky succeeds 
    # despite any floating-point asymmetry in H.
    F = cholesky(Symmetric(H) + ρ * I)

    # 4. Precompute the diagonal scale for the γ block (equality constraints)
    d_gamma = 1.0 / ε

    # 5. Define the in-place action of P^{-1} on a vector v
    function apply_P_inv!(res, v)
        # Block 1 (Δx): Apply inverse using the precomputed factorization
        res[1:n] .= F \ @view(v[1:n])

        # Block 2 (Δγ): Apply the diagonal scaling for the equality constraints
        res[n+1:end] .= d_gamma .* @view(v[n+1:end])

        return res
    end

    # 6. Wrap the function in a LinearOperator
    total_size = n + p
    P_inv_op = LinearOperator(
        T, total_size, total_size,
        true,  # is_symmetric
        true,  # is_hermitian
        (res, v) -> apply_P_inv!(res, v)
    )

    return P_inv_op
end



function preconditioner_partially_condensed_implicit(z, problem::QPProblem, retraction_map::RetractionMap; β=0.0, ρ=0.0, δ=1e-8, ε=1e-4)
    """
    preconditioner_partially_condensed_implicit(z, problem; β=0.0, ρ=0.0, δ=1e-8, ε=1e-4)

    Constructs a strictly diagonal, Symmetric Positive Definite (SPD) preconditioner 
    for the partially condensed implicit KKT system (Eq. 19), designed for use with MINRES.

    ### Mathematical Formulation
    The exact system matrix for the partially condensed form is a saddle-point matrix:

        I(v,b) = [ Q - AᵀA   -Aᵀ        -Cᵀ ]
                [ -A        -B_β⁻(v)    0  ]
                [ -C         0          0  ]

    Because MINRES strictly requires an SPD preconditioner and the exact (1,1) block 
    H = Q - AᵀA is indefinite, we construct a block-diagonal SPD approximation 
    P = diag(H_hat, S_v_hat, S_γ_hat):

    1. **(1,1) Block (Δx)**: 
    We approximate the indefinite block H with a strictly positive diagonal matrix:
    H_hat = max(diag(Q), 0) + diag(AᵀA) + ρ*I + δ*I
    (Note: diag(AᵀA) is efficiently computed using the sum of squared elements of the columns of A).

    2. **(2,2) Block (Δv)**: 
    We approximate the negative Schur complement for the inequality constraints:
    S_v_hat = diag(A * H_hat⁻¹ * Aᵀ) + B_β⁻(v) + δ*I
    where B_β⁻(v) = diag(db_β(-v)) evaluates the derivative of the smoothed ReLU retraction map.

    3. **(3,3) Block (Δγ)**: 
    We approximate the negative Schur complement for the equality constraints:
    S_γ_hat = diag(C * H_hat⁻¹ * Cᵀ) + ε*I

    The returned `LinearOperator` efficiently applies P⁻¹ to a vector by performing 
    element-wise multiplication with the reciprocal of these precomputed diagonals.
    """
    # 1. Unpack dimensions and variables
    n, m, p = problem.n, problem.m, problem.p
    Q = problem.Q
    A = problem.A
    C = problem.C # Assuming problem object contains C

    T = eltype(Q)

    # Extract v from the augmented state vector
    v_start = n + m + p + m + 1
    v_end = n + m + p + 2m
    v = z[v_start:v_end]

    # 2. Approximate the (1,1) block: H_hat ≈ diag(Q) + diag(AᵀA) + ρI + δI
    # We use abs2.(A) to compute the diagonal of AᵀA efficiently (preserves sparsity if A is sparse)
    diag_Q = diag(Q)
    diag_AtA = vec(sum(abs2.(A), dims=1))

    # Ensure strict positive definiteness for MINRES
    H_hat_diag = max.(diag_Q, 0.0) .+ diag_AtA .+ ρ .+ δ
    H_inv_diag = 1.0 ./ H_hat_diag

    # 3. Approximate the Schur complements for blocks 2 and 3
    # db_minus = db_β(-v) using 4β (as per paper Eq 25) instead of 4β^2
    db_minus = 0.5 .* (1.0 .- v ./ sqrt.(v .^ 2 .+ 4 * β))

    # Block 2 (Δv): S_v ≈ diag(A * H_hat⁻¹ * Aᵀ) + B_β⁻(v)
    diag_A_Hinv_At = vec(sum(abs2.(A) .* H_inv_diag', dims=2))
    S2_diag = diag_A_Hinv_At .+ db_minus .+ δ
    S2_inv_diag = 1.0 ./ S2_diag

    # Block 3 (Δγ): S_γ ≈ diag(C * H_hat⁻¹ * Cᵀ) + εI
    # Wrap in a check in case there are no equality constraints (p = 0)
    if p > 0
        diag_C_Hinv_Ct = vec(sum(abs2.(C) .* H_inv_diag', dims=2))
        S3_diag = diag_C_Hinv_Ct .+ ε
        S3_inv_diag = 1.0 ./ S3_diag
    end

    # 4. Define the in-place action of P^{-1} on a vector vec
    function apply_P_inv!(res, vec)
        # Block 1: Δx
        res[1:n] .= H_inv_diag .* @view(vec[1:n])

        # Block 2: Δv
        res[n+1:n+m] .= S2_inv_diag .* @view(vec[n+1:n+m])

        # Block 3: Δγ
        if p > 0
            res[n+m+1:end] .= S3_inv_diag .* @view(vec[n+m+1:end])
        end

        return res
    end

    # 5. Wrap the function in a LinearOperator
    total_size = n + m + p
    P_inv_op = LinearOperator(
        T, total_size, total_size,
        true,  # is_symmetric
        true,  # is_hermitian
        (res, vec) -> apply_P_inv!(res, vec)
    )

    return P_inv_op
end


# --------------------------- linear system related -------------------------- #
function uncompact_z!(Δz, Δz_compact, z, r, inds::StateIndices, problem::QPProblem,
    solver_mode::SolverMode, retraction_map::RetractionMap; β=0.0)
    n, m, p = problem.n, problem.m, problem.p

    @views begin
        λ = z[inds.λ]
        s = z[inds.s]
        r2 = r[inds.λ]
        r4 = r[inds.s]
        r5 = inds.implicit ? r[inds.v] : 0

    end


    if solver_mode == MODE_UNCONDENSED || solver_mode == MODE_UNCONDENSED_IMPLICIT
        # Mutate Δz in-place
        Δz .= Δz_compact

    elseif solver_mode == MODE_PARTIALLY_CONDENSED
        @views begin
            Δx = Δz_compact[1:n]
            Δλ = Δz_compact[n+1:n+m]
            Δγ = Δz_compact[n+m+1:n+m+p]
        end
        # Calculate Δs directly into the correct view of Δz to avoid allocations
        Δs = @view Δz[n+m+p+1:end]
        @. Δs = -(r4 + s * Δλ) / λ

        # Copy the rest into Δz in-place
        Δz[1:n] .= Δx
        Δz[n+1:n+m] .= Δλ
        Δz[n+m+1:n+m+p] .= Δγ

    elseif solver_mode == MODE_FULLY_CONDENSED
        @views begin
            Δx = Δz_compact[1:n]
            Δγ = Δz_compact[n+1:n+p]
        end

        # Determine views for the calculated parts
        Δλ = @view Δz[n+1:n+m]
        Δs = @view Δz[n+m+p+1:end]

        # Calculate and assign in-place
        Δs .= problem.A * Δx .+ r2
        Δλ .= -(r4 + λ * Δs) / s

        # Copy the extracted views
        Δz[1:n] .= Δx
        Δz[n+m+1:n+m+p] .= Δγ

    elseif solver_mode == MODE_PARTIALLY_CONDENSED_IMPLICIT
        @views begin
            Δx = Δz_compact[1:n]
            Δv = Δz_compact[n+1:n+m]
            Δγ = Δz_compact[n+m+1:n+m+p]
        end

        # Assuming Δz holds [Δx; Δλ; Δγ; Δs; Δv] in this mode
        Δλ = @view Δz[n+1:n+m]
        Δs = @view Δz[n+m+p+1:n+m+p+m] # Adjust sizing as necessary based on your math

        Δs .= problem.A * Δx .+ r2
        # Δλ .= problem.A * Δx + Δv + (r2 - r4 + r5)
        # if retraction_map == RETRACTION_MAP_EXP
        b_plus_vals = drm.(z[inds.v], β, retraction_map)
        Δλ .= b_plus_vals .* Δv .- r4
        # else
        #     # Original softplus specific relation (assumes B_μ(v) + B_μ(-v) = I)
        #     Δλ .= problem.A * Δx + Δv + (r2 - r4 + r5)
        # end

        Δz[1:n] .= Δx
        Δz[n+m+1:n+m+p] .= Δγ
        Δz[end-m+1:end] .= Δv # Adjust indices as per your exact vector layout
    end

    return Δz
end


function get_rhs!(rhs, z, r, inds::StateIndices, problem::QPProblem, solver_mode::SolverMode)


    @views begin
        λ, s = z[inds.λ], z[inds.s]
        r1, r2, r3, r4 = r[inds.x], r[inds.λ], r[inds.γ], r[inds.s]
        r5 = inds.implicit ? r[inds.v] : 0
        n, m, p = problem.n, problem.m, problem.p

        if solver_mode == MODE_UNCONDENSED
            rhs .= -r
        elseif solver_mode == MODE_PARTIALLY_CONDENSED
            rhs[inds.x] .= .-r1
            rhs[inds.λ] .= (r2 .+ r4 ./ λ)
            rhs[inds.γ] .= r3

        elseif solver_mode == MODE_FULLY_CONDENSED
            rhs[inds.x] .= (-r1 - problem.A' * ((λ .* r2 .+ r4) ./ s))
            rhs[n+1:n+p] .= r3
        elseif solver_mode == MODE_UNCONDENSED_IMPLICIT
            rhs .= -r
        elseif solver_mode == MODE_PARTIALLY_CONDENSED_IMPLICIT
            rhs[inds.x] .= -r1 + problem.A' * (r2 - r4 + r5)
            rhs[inds.λ] .= (r2 + r5)
            rhs[inds.γ] .= r3
        else
            error("Invalid solver mode")
        end
    end

    return rhs
end

# --------------------------------- residuals -------------------------------- #

function residuals!(r, z, μ, σ, inds::StateIndices, problem::QPProblem, retraction_map::RetractionMap, β::Real; Δz_aff=nothing)
    T = eltype(z)

    # Create views into qp problem, z and r
    @views begin
        x, λ, γ, s = z[inds.x], z[inds.λ], z[inds.γ], z[inds.s]
        r_stat, r_prim1, r_prim2, r_compl1 = r[inds.x], r[inds.λ], r[inds.γ], r[inds.s]
        if inds.implicit
            v = z[inds.v]
            r_compl2 = r[inds.v]
        end

        Q, q = problem.Q, problem.q
        A, b = problem.A, problem.b
        C, d = problem.C, problem.d
    end


    # 1. Stationarity: Q*x + q - A'*λ - C'*γ
    mul!(r_stat, Q, x)
    r_stat .+= q
    mul!(r_stat, A', λ, -one(T), one(T))   # r_stat -= A'*λ
    mul!(r_stat, C', γ, -one(T), one(T))   # r_stat -= C'*γ

    # 2.1. Primal Feasibility: A*x - b - s
    mul!(r_prim1, A, x)
    r_prim1 .-= b
    r_prim1 .-= s

    # 2.2. Primal Feasibility: C*x - d
    mul!(r_prim2, C, x)
    r_prim2 .-= d

    # 3. Complementarity: s ⊙ λ - σ * μ
    if inds.implicit
        @. r_compl1 = λ - rm(v, β, retraction_map)
        @. r_compl2 = s - rm(-v, β, retraction_map)
    else
        @. r_compl1 = λ * s - β

        if !isnothing(Δz_aff) # correction step predictor-corrector
            @. r_compl1 = r_compl1 + Δz_aff[inds.λ] * Δz_aff[inds.s]
        end
    end

    return r
end

function residuals(z, μ, σ, inds::StateIndices, problem::QPProblem, retraction_map::RetractionMap, β::Real)
    n, m, p = problem.n, problem.m, problem.p

    # Allocate r on the same device and with same type as z
    r = similar(z, size(z, 1))
    fill!(r, zero(eltype(z)))

    # Call the in-place version
    return residuals!(r, z, μ, σ, inds, problem, retraction_map, β)
end

function residual_norms(r, inds::StateIndices)
    nrc2 = inds.implicit ? norm(r[inds.v]) : 0
    return norm(r[inds.x]), norm(r[inds.λ]), norm(r[inds.γ]), norm(r[inds.s]), nrc2
end

# -------------------------------- line search ------------------------------- #
@inline function norm2_f64(r)
    return mapreduce(x -> (Float64(x) * Float64(x)), +, r)
end

function line_search(
    z, Δz, r, μ, σ, γ_armijo, τ, problem, inds::StateIndices, retraction_map::RetractionMap, β::Real;
    α0=one(eltype(z)),
    α_min=1e-6,
    ztrial=nothing
)
    n, m, p = problem.n, problem.m, problem.p
    T = eltype(z)

    # Keep scalar types compatible with z (Float32/Float64)
    α = convert(T, α0)
    τ_step = convert(T, τ)
    α_minT = convert(T, α_min)
    γ64 = Float64(γ_armijo) # Renamed to avoid confusion with the equality duals

    # Work buffers
    z_trial = (ztrial === nothing) ? similar(z) : ztrial
    r_trial = similar(r)

    # Merit function at current point
    φ0 = 0.5 * norm2_f64(r)

    while true
        # Candidate point
        @. z_trial = z + α * Δz

        # In implicit mode, merit must be evaluated on the projected
        # trial point because the accepted iterate overwrites λ and s
        # from v immediately after the step.
        # if inds.implicit
        #     @views begin
        #         z_trial[inds.λ] .= rm.(z_trial[inds.v], T(μ))
        #         z_trial[inds.s] .= rm.(-z_trial[inds.v], T(μ))
        #     end
        # end

        @views begin
            λ = z_trial[inds.λ]
            s = z_trial[inds.s]
        end

        # Residual and merit at candidate
        residuals!(r_trial, z_trial, μ, σ, inds, problem, retraction_map, β)
        φ_trial = 0.5 * norm2_f64(r_trial)

        # Armijo-like sufficient decrease
        α64 = Float64(α)
        sufficient_decrease = φ_trial <= (1.0 - γ64 * α64) * φ0

        # Feasibility for dual/slack variables (only λ and s must be ≥ 0)
        nonnegative_dual_slack = all(λ .>= 0) && all(s .>= 0)

        condition = sufficient_decrease && nonnegative_dual_slack
        # if inds.implicit
        #     condition = sufficient_decrease
        # end
        if condition
            return α, r_trial, true
        end

        # @infiltrate
        # return nothing
        α *= τ_step
        if α < α_minT
            return α, r_trial, false
        end
    end
end

# ------------------------------ starting point ------------------------------ #
function starting_point(problem::QPProblem, inds::StateIndices, use_sparse, T, retraction_map::RetractionMap; ε0=1e0)
    # s = ones(T, m)
    # λ = s
    # x = zeros(T, n)#T.((problem.Q + 1e-4 * I(n)) \ (-problem.q + problem.A' * λ))

    n, m, p = problem.n, problem.m, problem.p

    # Set up blocks for K0
    Id_m = identity_matrix(T, m, use_sparse)
    z_mp = zero_matrix(T, m, p, use_sparse)
    z_pm = zero_matrix(T, p, m, use_sparse)
    z_pp = zero_matrix(T, p, p, use_sparse)

    Q_block = maybe_sparse(problem.Q, use_sparse)
    A_block = maybe_sparse(problem.A, use_sparse)
    C_block = maybe_sparse(problem.C, use_sparse)

    # Expanded relaxed KKT system for initialization
    K0 = [
        Q_block -A_block' -C_block';
        A_block Id_m z_mp;
        C_block z_pm z_pp
    ]
    K0_reg = copy(K0)
    # K0_reg[inds.x, inds.x] = K0[inds.x, inds.x] + 1e-4 * I(size(K0[inds.x, inds.x], 1))
    # K0_reg[inds.λ, inds.λ] = K0[inds.λ, inds.λ] - 0.0 * I(size(K0[inds.λ, inds.λ], 1))
    # K0_reg[inds.γ, inds.γ] = K0[inds.γ, inds.γ] - 1e-4 * I(size(K0[inds.γ, inds.γ], 1))

    # Expanded right-hand side
    r0 = [-problem.q; problem.b; problem.d]

    z_hat = K0_reg \ r0

    # Extract initial guesses
    x_hat = z_hat[inds.x]
    λ_hat = z_hat[inds.λ]
    γ_hat = z_hat[inds.γ]
    s_hat = -λ_hat

    x0 = x_hat
    λ0 = λ_hat
    γ0 = γ_hat
    s0 = s_hat

    # Shift only the inequality multipliers and slacks strictly positive
    αλ, αs = minimum(λ_hat), minimum(s_hat)
    if ε0 > αλ
        λ0 = λ_hat + (ε0 - αλ) * ones(m)
    end
    if ε0 > αs
        s0 = s_hat + (ε0 - αs) * ones(m)
    end

    d0 = s0 ./ λ0
    # println(minimum(abs.(d0)), " ", maximum(abs.(d0)))
    # @infiltrate

    z = nothing
    gap = dot(λ0, s0)
    if inds.implicit
        μ = gap / m
        v0 = rm_inv.(λ0, μ, retraction_map)
        z = [x0; λ0; γ0; s0; v0]
    else
        z = [x0; λ0; γ0; s0]
    end


    return z, gap
end

# ------------------------------- solver kernel ------------------------------ #

function solver_kernel(problem::QPProblem;
    solver_mode=MODE_PARTIALLY_CONDENSED, # solver mode
    solver_backend=BACKEND_DIRECT, # solver backend
    max_iters=100, # maximum number of iterations
    scale=true, # scale problem
    σ=0.8, # centering parameter
    ls_γ=1e-4, ls_τ=0.9, ls_αm=1e-12, # line search parameters
    ε_s=1e-4, ε_p=1e-4, ε_g=1e-4, ε_c=1e-4, # stopping criteria
    monitor=Dict(:cond => false, :eigvals => false, :ΔK => false),
    retraction_map=RETRACTION_MAP_SOFTPLUS,
    ##### krylov settings ####
    kr_method=:minres, # Krylov method
    kr_maxit=60000,
    kr_tol=1e-10,
    kr_precondition=true,
    kr_ρ=0.0,
    kr_δ=1e-8,
    kr_ε=1e-4,
    ##### inexact Newton settings ####
    inexact_newton=false, # use inexact Newton
    in_θ=0.2, # inexact Newton tolerance
    in_conservative=true, # use conservative inexact Newton
)
    # check if implicit
    use_implicit = false
    if solver_mode in [MODE_UNCONDENSED_IMPLICIT, MODE_PARTIALLY_CONDENSED_IMPLICIT]
        use_implicit = true
    end

    # safety checks
    if solver_mode == MODE_UNCONDENSED || solver_mode == MODE_UNCONDENSED_IMPLICIT
        kr_precondition = false
        kr_method = :gmres
    end

    # inexact newton only with direct and implicit mode
    if inexact_newton && solver_backend != BACKEND_DIRECT && !use_implicit
        @error "Inexact Newton only supported with direct backend and implicit mode"
        return nothing
    end

    # unpack problem dimensions
    T = eltype(problem.Q)
    n = problem.n
    m = problem.m
    p = problem.p
    use_sparse = is_sparse_problem(problem)

    println("\n")
    println("-------------------------------------------------------------")
    println("                            i²PD                             ")
    println("           Implicit Interior Primal-Dual Optimization        ")
    println("-------------------------------------------------------------")

    @printf("\nproblem:\n")
    @printf(" variables (n): %d\n", n)
    @printf(" inequality constraints (m): %d\n", m)
    @printf(" equality constraints (p): %d\n", p)
    @printf(" sparse: %s\n", use_sparse)
    @printf(" scale: %s\n", scale)

    @printf("\nsettings:\n")
    @printf(" backend: %s\n", solver_backend)
    @printf(" mode: %s\n", solver_mode)
    solver_mode == MODE_PARTIALLY_CONDENSED_IMPLICIT && @printf(" retraction map: %s\n", retraction_map)
    inexact_newton && @printf(" inexact newton: %s, η: %1.2e\n", inexact_newton, in_θ)
    @printf(" precision: %s\n", T)
    @printf(" max_iters: %d\n", max_iters)
    @printf(" σ: %1.2f\n", σ)
    @printf(" ls_γ, ls_τ, ls_αm: %1.2e, %1.2f, %1.2e\n", ls_γ, ls_τ, ls_αm)
    @printf(" ε_s, ε_p, ε_g: %1.2e, %1.2e, %1.2e\n", ε_s, ε_p, ε_g)
    if solver_backend == BACKEND_ITERATIVE
        @printf("\nkrylov settings:\n")
        @printf(" kr_method, kr_precondition: %s, %s\n", kr_method, kr_precondition)
        @printf(" kr_maxit, kr_tol: %d, %1.2e\n", kr_maxit, kr_tol)
        @printf(" kr_ρ, kr_δ, kr_ε: %1.2e, %1.2e, %1.2e\n", kr_ρ, kr_δ, kr_ε)
    end
    @printf("\n")

    # scale problem
    problem_unscaled = problem
    if scale
        Q = problem.Q
        q = problem.q
        A = problem.A
        b = problem.b
        C = problem.C
        d = problem.d
        c = problem.c

        Qes, qes, Aes, bes, Ces, des, d_scale, ea_scale, ec_scale = ruiz_scale(
            Q, q, A, b, C, d; max_iters=40)

        problem = QPProblem(Qes, qes, Aes, bes, Ces, des, c)
    end


    # get state indices
    inds = StateIndices(problem; implicit=use_implicit)

    # get starting point
    z, gap = starting_point(problem, inds, use_sparse, T, retraction_map)
    μ = gap / m
    # println(maximum(abs.(rm.(z[inds.v], μ) - z[inds.λ])))

    # initialize variables
    Δz = zeros(T, size(z, 1))
    α = 0.0

    β = σ * μ
    r = residuals(z, μ, σ, inds, problem, retraction_map, β)
    K = kkt_matrix(z, problem, solver_mode, retraction_map; β=β)
    Δz_compact = zeros(T, size(K, 1))
    rhs = zeros(T, size(K, 1))

    K_prev = copy(K)

    K_old = copy(K)
    K_fact = lu(K)
    inexact_newton_refact = true
    in_upper = 0.0
    in_lower = 0.0

    # initialize Krylov related
    kr_ws = Krylov.krylov_workspace(Val(kr_method), size(K, 1), size(K, 2), Vector{Float64})
    kr_r = 0.0
    kr_it = NaN
    if solver_backend == BACKEND_ITERATIVE && kr_precondition
        P_inv = preconditioner(z, problem, solver_mode, retraction_map; β=σ * μ, ρ=kr_ρ, δ=kr_δ, ε=kr_ε)
    end

    # initialize data storage
    data = zeros(T, max_iters, 16)
    x_hist = zeros(T, max_iters, n)
    λ_hist = zeros(T, max_iters, m)
    γ_hist = zeros(T, max_iters, p)
    s_hist = zeros(T, max_iters, m)
    v_hist = zeros(T, max_iters, m)
    e_hist = zeros(T, max_iters, n + m)
    t_analyze, t_residuals, t_kkt_constr, t_kkt_solve, t_line_search, t_iter = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


    # solver loop
    it = 1
    solver_status = STATUS_UNKNOWN
    while it <= max_iters
        t_iter = @elapsed begin

            # ---------------------------------- Analyze --------------------------------- #
            @views begin
                x = z[inds.x]
                λ = z[inds.λ]
                γ = z[inds.γ]
                s = z[inds.s]
            end

            t_analyze = @elapsed begin

                # verbose
                cp = primal_cost(x, problem)
                cd = 0.0#dual_cost(λ, problem, Q_inv)
                gap = dot(λ, s) #same as cp - cd
                refact = !inexact_newton || (inexact_newton && inexact_newton_refact)

                nrs, nrpi, nrpe, nrc1, nrc2 = residual_norms(r, inds)
                nrc = max(nrc1, nrc2)
                nz = norm(Δz)

                cond_K = NaN
                sdvals_K = zeros(n + m) * NaN
                nΔK = NaN

                if !use_sparse
                    cond_K = monitor[:cond] ? cond(K) : NaN
                    sdvals_K = monitor[:eigvals] ? svdvals(K) : zeros(n + m) * NaN
                else
                    if solver_mode in [MODE_PARTIALLY_CONDENSED, MODE_PARTIALLY_CONDENSED_IMPLICIT]
                        cond_K = monitor[:cond] ? cond(K[1:n+m, 1:n+m], 1) : NaN
                        sdvals_K = monitor[:eigvals] ? svdvals(Array(K[1:n+m, 1:n+m])) : zeros(n + m) * NaN
                        nΔK = monitor[:ΔK] ? norm(K_prev - K) : NaN
                    elseif solver_mode == MODE_FULLY_CONDENSED
                        cond_K = cond(K[1:n, 1:n], 1)
                    else
                        cond_K = cond(K, 1)
                    end
                end


                show_timers = false# (it % monitor) == 0
                timing_suffix = show_timers ? @sprintf(
                    "| t_analyze: %.3e, t_residuals: %.3e, t_kkt_constr: %.3e, t_kkt_solve: %.3e, t_line_search: %.3e, t_iter: %.3e",
                    t_analyze, t_residuals, t_kkt_constr, t_kkt_solve, t_line_search, t_iter,
                ) : ""
                krylov_suffis = solver_backend == BACKEND_ITERATIVE ? @sprintf(
                    " | kr_it: %2d, kr_r: %1.2e", kr_it, kr_r) : ""
                @printf(
                    "it: %2d | gap: %1.2e μ: %1.2e, cp: %2.3e | rs: %1.3e, rpi: %1.3e, rpe: %1.3e, rc: %1.3e | α: %1.2e, Δz: %1.3e %s %s\n",
                    it, gap, μ, cp, nrs, nrpi, nrpe, nrc, α, nz, krylov_suffis, timing_suffix,
                )

                # store data
                data[it, :] = [it, gap, cp, cd, nrs, max(nrpi, nrpe), nrc, 0, α, nz, cond_K, kr_it, refact, t_kkt_solve, nΔK, in_upper]
                x_hist[it, :] = z[inds.x]
                λ_hist[it, :] = z[inds.λ]
                γ_hist[it, :] = z[inds.γ]
                s_hist[it, :] = z[inds.s]
                inds.implicit && (v_hist[it, :] = z[inds.v])
                e_hist[it, :] = sdvals_K

                # reduce μ
                stat_primal_cond = nrs <= ε_s && max(nrpi, nrpe) <= ε_p
                gap_cond = gap <= ε_g
                compl_cond = nrc <= ε_c
                if stat_primal_cond && gap_cond && compl_cond
                    println("Converged !")
                    solver_status = STATUS_CONVERGED
                    break
                end
                μ = gap / m
            end

            # ---------------------------------- Update ---------------------------------- #
            if retraction_map == RETRACTION_MAP_EXP
                if nrc < 1e-3
                    β = σ * μ
                else
                    β = μ
                end
            else
                β = σ * μ
            end

            t_residuals = @elapsed begin
                # get linear system
                residuals!(r, z, μ, σ, inds, problem, retraction_map, β)
                @views begin
                    r1, r2, r3, r4 = r[inds.x], r[inds.λ], r[inds.γ], r[inds.s]
                    r5 = inds.implicit ? r[inds.v] : 0
                end
            end

            t_kkt_constr = @elapsed begin
                monitor[:ΔK] && (K_prev = copy(K))
                get_rhs!(rhs, z, r, inds, problem, solver_mode)
                kkt_matrix!(K, z, problem, solver_mode, retraction_map; β=β)
                # TEST  ------------- -------------
                # K[inds.x, inds.x] = K[inds.x, inds.x] + 1e-4 * I(size(K[inds.x, inds.x], 1))
                # K[inds.λ, inds.λ] = K[inds.λ, inds.λ] + 1e-12 * I(size(K[inds.λ, inds.λ], 1))
                # K[inds.γ, inds.γ] = K[inds.γ, inds.γ] - 1e-4 * I(size(K[inds.γ, inds.γ], 1))
                # TEST  ------------- -------------
                if kr_precondition
                    P_inv = preconditioner(z, problem, solver_mode, retraction_map;
                        β=μ, ρ=kr_ρ, δ=kr_δ, ε=kr_ε, exact_chol=false)
                end
            end


            # solve
            t_kkt_solve = @elapsed begin

                if solver_backend == BACKEND_DIRECT
                    # Δz_compact = K \ rhs

                    if inexact_newton

                        # solve with frozen K
                        Δz_compact_candidate = K_fact \ rhs

                        # check if the candidate is a good enough solution
                        θ_k = in_θ
                        if in_θ == -1.0
                            θ_k = min(1.0, norm(rhs))
                        end

                        if in_conservative # might contradict line search
                            in_lower = norm(K - K_old)
                            in_upper = θ_k * norm(rhs) / norm(Δz_compact_candidate)
                        else
                            in_lower = norm((K - K_old) * Δz_compact_candidate)
                            in_upper = θ_k * norm(rhs)
                        end
                        inexact_newton_refact = in_lower > in_upper

                        if !inexact_newton_refact # no refactorization needed
                            Δz_compact = Δz_compact_candidate
                        else # refactorization needed
                            K_old = copy(K)
                            K_fact = lu(K)
                            Δz_compact = K_fact \ rhs
                        end

                    else
                        K_fact = lu(K)
                        Δz_compact = K_fact \ rhs
                        # @infiltrate
                    end

                    uncompact_z!(Δz, Δz_compact, z, r, inds, problem, solver_mode,
                        retraction_map; β=β)

                    # println(typeof(K), " ", typeof(rhs), " ", typeof(Δz_compact), " ", typeof(Δz))


                elseif solver_backend == BACKEND_ITERATIVE

                    if kr_method == :minres
                        if !issymmetric(K)
                            # @infiltrate
                            error("KKT matrix K is not symmetric!")
                            return nothing
                        end
                    end

                    if kr_precondition
                        Krylov.krylov_solve!(kr_ws, K, rhs,
                            itmax=kr_maxit,
                            atol=kr_tol, rtol=kr_tol,
                            M=P_inv, ldiv=false,
                            # verbose=1,
                        )
                    else
                        Krylov.krylov_solve!(kr_ws, K, rhs,
                            itmax=kr_maxit,
                            atol=kr_tol, rtol=kr_tol,
                            # verbose=1,
                        )
                    end
                    Δz_compact, stats = Krylov.results(kr_ws)
                    uncompact_z!(Δz, Δz_compact, z, r, inds, problem, solver_mode,
                        retraction_map; β=β)


                    # @infiltrate
                    # if stats.inconsistent
                    #     max_res = maximum(abs.(K * Δz_compact - rhs))
                    #     println("\nmax residual: ", max_res)#, " ", cond(K2))
                    #     println(stats)
                    #     @error "Krylov solver found inconsistent solution"
                    #     break
                    # end

                    kr_it = stats.niter
                    kr_r = maximum(abs.(K * Δz_compact - rhs))
                    if !stats.solved
                        @warn "Krylov solver did not converge"
                        println(stats)
                        break
                    end
                end
            end

            # line search
            t_line_search = @elapsed begin
                α, _, ls_success = line_search(z, Δz, r, μ, σ, ls_γ, ls_τ,
                    problem, inds, retraction_map, β; α_min=ls_αm)
                if !ls_success
                    println("Line search failed in iteration $it/$max_iters !")
                    solver_status = STATUS_LINESEARCH_FAILED
                    break
                    # α = 0.0
                end
            end
            # println(typeof(α), " ", typeof(Δz), " ", typeof(z))


            # update step
            z .+= α * Δz
            # if use_implicit
            #     z[inds.λ] .= rm.(z[inds.v], T(σ * μ), retraction_map)
            #     z[inds.s] .= rm.(-z[inds.v], T(σ * μ), retraction_map)
            # end

            # iterate
            it += 1

            # @infiltrate
            # return nothing
        end
    end

    if it == max_iters + 1
        solver_status = STATUS_MAX_ITER
    end

    if it < max_iters
        data = data[1:it, :]
        x_hist = x_hist[1:it, :]
        λ_hist = λ_hist[1:it, :]
        γ_hist = γ_hist[1:it, :]
        s_hist = s_hist[1:it, :]
        v_hist = v_hist[1:it, :]
        e_hist = e_hist[1:it, :]
    end

    cp = data[end, 3]
    if scale
        # unscale all stored iterates, columnwise
        x_hist .*= reshape(d_scale, 1, :)
        λ_hist .*= reshape(ea_scale, 1, :)
        γ_hist .*= reshape(ec_scale, 1, :)
        s_hist ./= reshape(ea_scale, 1, :)
        cp = primal_cost(x_hist[end, :], problem_unscaled)
    end
    println("-------------------------------------------------------------")
    if solver_status == STATUS_CONVERGED
        @printf("terminated with status: %s\n", solver_status)
    else
        @warn "terminated with status: $solver_status"
    end
    @printf("cost: %1.4f\n", cp)
    @printf("number of iterations: %d\n\n", it)

    return Solution(x_hist, λ_hist, s_hist, v_hist, data[:, 3]), e_hist, data
end

end # module SolverExplicit