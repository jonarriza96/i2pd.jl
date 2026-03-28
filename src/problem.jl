
function ruiz_scale(Q, q, A, b, C, d; max_iters=10)
    n = size(Q, 1)
    m = size(A, 1)
    p = size(C, 1)

    # Work in floating point and preserve sparsity if the inputs are sparse.
    Q_bar = float(copy(Q))
    A_bar = float(copy(A))
    C_bar = float(copy(C))

    T = eltype(Q_bar)
    tiny = T(1e-12)

    d_scale = ones(T, n)
    ea_scale = ones(T, m)
    ec_scale = ones(T, p)

    norm_Q = zeros(T, n)
    norm_A_col = zeros(T, n)
    norm_C_col = zeros(T, n)
    epsA = zeros(T, m)
    epsC = zeros(T, p)

    inv_sqrt_delta = ones(T, n)
    inv_sqrt_epsA = ones(T, m)
    inv_sqrt_epsC = ones(T, p)

    function col_max_abs!(out, M)
        fill!(out, zero(eltype(out)))

        if issparse(M)
            vals = nonzeros(M)
            for j in 1:size(M, 2)
                maxval = zero(eltype(out))
                for k in nzrange(M, j)
                    v = abs(vals[k])
                    if v > maxval
                        maxval = v
                    end
                end
                out[j] = maxval
            end
        else
            @inbounds for j in 1:size(M, 2)
                maxval = zero(eltype(out))
                for i in 1:size(M, 1)
                    v = abs(M[i, j])
                    if v > maxval
                        maxval = v
                    end
                end
                out[j] = maxval
            end
        end

        return out
    end

    function row_max_abs!(out, M)
        fill!(out, zero(eltype(out)))

        if issparse(M)
            rows = rowvals(M)
            vals = nonzeros(M)
            for j in 1:size(M, 2)
                for k in nzrange(M, j)
                    i = rows[k]
                    v = abs(vals[k])
                    if v > out[i]
                        out[i] = v
                    end
                end
            end
        else
            @inbounds for i in 1:size(M, 1)
                maxval = zero(eltype(out))
                for j in 1:size(M, 2)
                    v = abs(M[i, j])
                    if v > maxval
                        maxval = v
                    end
                end
                out[i] = maxval
            end
        end

        return out
    end

    function scale_matrix!(M, row_scale, col_scale)
        if issparse(M)
            rows = rowvals(M)
            vals = nonzeros(M)
            for j in 1:size(M, 2)
                cj = col_scale[j]
                for k in nzrange(M, j)
                    vals[k] *= row_scale[rows[k]] * cj
                end
            end
        else
            @inbounds for j in 1:size(M, 2)
                cj = col_scale[j]
                for i in 1:size(M, 1)
                    M[i, j] *= row_scale[i] * cj
                end
            end
        end

        return M
    end

    function canonicalize_sparse(M)
        if issparse(M)
            I, J, V = findnz(M)
            return sparse(I, J, V, size(M, 1), size(M, 2))
        end
        return M
    end

    for _ in 1:max_iters
        col_max_abs!(norm_Q, Q_bar)

        if m > 0
            col_max_abs!(norm_A_col, A_bar)
            row_max_abs!(epsA, A_bar)
        else
            fill!(norm_A_col, zero(T))
        end

        if p > 0
            col_max_abs!(norm_C_col, C_bar)
            row_max_abs!(epsC, C_bar)
        else
            fill!(norm_C_col, zero(T))
        end

        @inbounds for i in 1:n
            delta_i = max(norm_Q[i], norm_A_col[i], norm_C_col[i])
            s = delta_i <= tiny ? one(T) : inv(sqrt(delta_i))
            inv_sqrt_delta[i] = s
            d_scale[i] *= s
        end

        if m > 0
            @inbounds for i in 1:m
                s = epsA[i] <= tiny ? one(T) : inv(sqrt(epsA[i]))
                inv_sqrt_epsA[i] = s
                ea_scale[i] *= s
            end
        end

        if p > 0
            @inbounds for i in 1:p
                s = epsC[i] <= tiny ? one(T) : inv(sqrt(epsC[i]))
                inv_sqrt_epsC[i] = s
                ec_scale[i] *= s
            end
        end

        scale_matrix!(Q_bar, inv_sqrt_delta, inv_sqrt_delta)

        if m > 0
            scale_matrix!(A_bar, inv_sqrt_epsA, inv_sqrt_delta)
        end

        if p > 0
            scale_matrix!(C_bar, inv_sqrt_epsC, inv_sqrt_delta)
        end
    end

    q_bar = d_scale .* q
    b_bar = ea_scale .* b
    d_bar = ec_scale .* d

    # Some Maros-Meszaros problems end up with non-canonical CSC column order
    # after in-place sparse scaling, which SuiteSparse rejects as "invalid".
    Q_bar = canonicalize_sparse(Q_bar)
    A_bar = canonicalize_sparse(A_bar)
    C_bar = canonicalize_sparse(C_bar)


    return Q_bar, q_bar, A_bar, b_bar, C_bar, d_bar, d_scale, ea_scale, ec_scale
end

"""
    reduce_system(A, b; tol=nothing)

Reduces the linear system Ax = b by removing linearly dependent equations (rows).
Returns the full-row-rank matrix A_reduced and corresponding vector b_reduced.
"""

# ---------------------------------------------------------
# METHOD 1: DENSE MATRICES
# ---------------------------------------------------------
function reduce_system(A::AbstractMatrix, b::AbstractVector; tol=nothing)
    # 1. Perform Pivoted QR on the TRANSPOSE of A
    # We use copy(A') to safely materialize the transposed matrix for LAPACK
    At = copy(A')
    F = qr(At, ColumnNorm())

    if tol === nothing
        tol = maximum(size(At)) * eps(real(eltype(At))) * abs(F.R[1, 1])
    end

    # 2. Determine rank (number of independent rows)
    r = count(x -> abs(x) > tol, diag(F.R))

    if r == 0
        return A[Int[], :], b[Int[]]
    end

    # 3. Extract the independent row indices and sort them to maintain order
    keep_rows = sort(F.p[1:r])
    println("Rows (original, reduced, removed): ",
        length(F.pcol), " ", length(keep_rows), " ",
        length(F.pcol) - length(keep_rows)), "\n"

    # 4. Slice both A and b
    return A[keep_rows, :], b[keep_rows]
end

# ---------------------------------------------------------
# METHOD 2: SPARSE MATRICES
# ---------------------------------------------------------
function reduce_system(A::SparseMatrixCSC, b::AbstractVector; tol=nothing)
    # 1. Materialize transpose explicitly to ensure it stays in SuiteSparse format
    At = sparse(A')

    # 2. Perform Sparse QR
    F = tol === nothing ? qr(At) : qr(At; tol=tol)

    # 3. Get the numerical rank
    r = rank(F)

    if r == 0
        return spzeros(eltype(A), 0, size(A, 2)), b[Int[]]
    end

    # 4. Extract independent row indices using sparse permutation vector
    keep_rows = sort(F.pcol[1:r])

    println("Rows (original, reduced, removed): ", length(F.pcol), " ", length(keep_rows), " ", length(F.pcol) - length(keep_rows))

    # 5. Slice both A and b
    return A[keep_rows, :], b[keep_rows]
end