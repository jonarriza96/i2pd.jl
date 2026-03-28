using MAT
using SparseArrays

# --- small conversion helpers ---

_scalar(x) = x isa Number ? x : x[begin]

function _to_vec_float(x)::Vector{Float64}
    x isa AbstractVector && return Float64.(x)
    x isa AbstractArray && return vec(Float64.(x))
    return [Float64(x)]
end

function _to_mat_float(x)
    if x isa SparseMatrixCSC
        M = SparseMatrixCSC{Float64,Int}(
            size(x, 1),
            size(x, 2),
            Int.(x.colptr),
            Int.(x.rowval),
            Float64.(x.nzval),
        )
        # Some MAT files store sparse columns with unsorted row indices, which
        # later breaks row slicing in Julia's sparse machinery.
        I, J, V = findnz(M)
        return sparse(I, J, V, size(M, 1), size(M, 2), +)
    else
        return Matrix{Float64}(x)
    end
end

function _replace_big_with_inf!(x; thresh::Float64=9e19)
    if x isa SparseMatrixCSC{Float64,Int}
        @inbounds for i in eachindex(x.nzval)
            v = x.nzval[i]
            if v > thresh
                x.nzval[i] = Inf
            elseif v < -thresh
                x.nzval[i] = -Inf
            end
        end
        return x
    elseif x isa AbstractArray{Float64}
        @inbounds for i in eachindex(x)
            v = x[i]
            if v > thresh
                x[i] = Inf
            elseif v < -thresh
                x[i] = -Inf
            end
        end
        return x
    else
        return x
    end
end

"""
    load_qp(name; data_dir=@__DIR__, inf_threshold=9e19, split_box=false)

Load Maros–Mészáros QP data from `\$(data_dir)/\$(name).mat` (or pass `name` ending in `.mat`).

Expected MAT variables (as produced by `sif2mat.m` in proxqp_benchmark):
- `P`, `q`, `A`, `l`, `u`, `r`, `n`, `m`

Returns a `NamedTuple` with Julia arrays:
- `P::Union{SparseMatrixCSC{Float64,Int}, Matrix{Float64}}`
- `q::Vector{Float64}`
- `A::Union{SparseMatrixCSC{Float64,Int}, Matrix{Float64}}`
- `l::Vector{Float64}`, `u::Vector{Float64}`
- `r::Float64`, `n::Int`, `m::Int`

If `split_box=true`, also includes:
- `C`, `l_c`, `u_c` (constraints without the trailing identity block)
- `lb`, `ub` (box bounds, last `n` entries of `l`/`u`)
"""
function load_qp_maros_meszaros(
    name::AbstractString;
    data_dir::AbstractString=@__DIR__,
    inf_threshold::Real=9e19,
    split_box::Bool=false,
)
    fname = endswith(lowercase(name), ".mat") ? name : string(name, ".mat")
    path = joinpath(data_dir, fname)
    isfile(path) || error("MAT file not found: $path")

    d = matread(path)

    P = _to_mat_float(d["P"])
    q = _to_vec_float(d["q"])
    A = _to_mat_float(d["A"])
    l = _to_vec_float(d["l"])
    u = _to_vec_float(d["u"])
    r = haskey(d, "r") ? Float64(_scalar(d["r"])) : 0.0
    n = Int(_scalar(d["n"]))
    m = Int(_scalar(d["m"]))

    size(A, 1) == m || error("A has wrong row dimension: size(A,1)=$(size(A,1)) but m=$m")
    size(A, 2) == n || error("A has wrong col dimension: size(A,2)=$(size(A,2)) but n=$n")

    # same cleanup as the Python loader: 1e20 is treated as infinity
    _replace_big_with_inf!(A; thresh=Float64(inf_threshold))
    _replace_big_with_inf!(l; thresh=Float64(inf_threshold))
    _replace_big_with_inf!(u; thresh=Float64(inf_threshold))

    probname = splitext(basename(fname))[1]

    if !split_box
        return (; name=probname, P, q, A, l, u, r, n, m)
    end

    # A == [C; I]  => last n rows correspond to box constraints on x
    lb = l[end-n+1:end]
    ub = u[end-n+1:end]
    C = A[1:end-n, :]
    l_c = l[1:end-n]
    u_c = u[1:end-n]

    return (; name=probname, P, q, A, l, u, r, n, m, C, l_c, u_c, lb, ub)
end

"""
    qp_to_ge_form(qp; eq_atol=0.0, use_split_box=false)

Convert a loaded Maros–Mészáros QP into the one-sided inequality form

    minimize    1/2 x' Q x + q' x
    subject to  A * x ≥ b

Input qp is the NamedTuple returned by load_qp.

Behavior
- For every finite lower bound lᵢ, adds    Aᵢ * x ≥ lᵢ
- For every finite upper bound uᵢ, adds  -Aᵢ * x ≥ -uᵢ
- If both bounds are finite and equal (within eq_atol), this produces two inequalities, equivalent to an equality.
- If use_split_box=true and qp contains C, l_c, u_c, lb, ub (from load_qp(...; split_box=true)),
  then box bounds are added explicitly as x ≥ lb and -x ≥ -ub, instead of relying on the stacked identity
  inside qp.A.

Returns `(Q, q, A, b, c)` where `c` is the constant objective term.
"""
function qp_to_ge_form(qp; eq_atol::Real=0.0, use_split_box::Bool=false)
    Q = 0.5 * (qp.P + qp.P')   # symmetrize defensively
    q = qp.q
    c = hasproperty(qp, :r) ? qp.r : 0.0

    if use_split_box
        hasproperty(qp, :C) || error("use_split_box=true requires qp from load_qp(...; split_box=true)")
        C = qp.C
        l_c = qp.l_c
        u_c = qp.u_c
        lb = qp.lb
        ub = qp.ub

        mC, n = size(C)
        length(l_c) == mC || error("length(l_c) != size(C,1)")
        length(u_c) == mC || error("length(u_c) != size(C,1)")
        length(lb) == n || error("length(lb) != n")
        length(ub) == n || error("length(ub) != n")

        lf = isfinite.(l_c)
        uf = isfinite.(u_c)

        A_ge = vcat(C[lf, :], -C[uf, :])
        b_ge = vcat(l_c[lf], -u_c[uf])

        # Add box bounds: x >= lb and -x >= -ub, skipping infinite bounds
        I = spdiagm(0 => ones(Float64, n))  # sparse identity
        lbf = isfinite.(lb)
        ubf = isfinite.(ub)

        A_ge = vcat(A_ge, I[lbf, :], -I[ubf, :])
        b_ge = vcat(b_ge, lb[lbf], -ub[ubf])

        return Q, q, A_ge, b_ge, c
    else
        A = qp.A
        l = qp.l
        u = qp.u

        m, n = size(A)
        length(l) == m || error("length(l) != size(A,1)")
        length(u) == m || error("length(u) != size(A,1)")

        lf = isfinite.(l)
        uf = isfinite.(u)

        A_ge = vcat(A[lf, :], -A[uf, :])
        b_ge = vcat(l[lf], -u[uf])

        return Q, q, A_ge, b_ge, c
    end
end

function load_qp(name::AbstractString)
    qp = load_qp_maros_meszaros(name; split_box=true)
    Q, q, A, b, c = qp_to_ge_form(qp; use_split_box=true)
    return Q, q, A, b, c
end


function qp_to_ge_eq_form(qp; eq_atol::Real=1e-10, use_split_box::Bool=false)
    Q = 0.5 * (qp.P + qp.P')
    q = qp.q
    c = hasproperty(qp, :r) ? qp.r : 0.0

    Aineq_blocks = AbstractMatrix[]
    bineq_blocks = Vector{Float64}[]
    Aeq_blocks = AbstractMatrix[]
    deq_blocks = Vector{Float64}[]

    function process_rows(M, l, u)
        eq = isfinite.(l) .& isfinite.(u) .& (abs.(l .- u) .<= eq_atol)
        lower = isfinite.(l) .& .!eq
        upper = isfinite.(u) .& .!eq

        if any(lower)
            push!(Aineq_blocks, M[lower, :])
            push!(bineq_blocks, l[lower])
        end
        if any(upper)
            push!(Aineq_blocks, -M[upper, :])
            push!(bineq_blocks, -u[upper])
        end
        if any(eq)
            push!(Aeq_blocks, M[eq, :])
            push!(deq_blocks, 0.5 .* (l[eq] .+ u[eq]))
        end
    end

    if use_split_box
        hasproperty(qp, :C) || error("use_split_box=true requires qp from load_qp_maros_meszaros(...; split_box=true)")

        process_rows(qp.C, qp.l_c, qp.u_c)

        n = length(q)
        I = spdiagm(0 => ones(Float64, n))
        process_rows(I, qp.lb, qp.ub)
    else
        process_rows(qp.A, qp.l, qp.u)
    end

    n = length(q)

    A = isempty(Aineq_blocks) ? spzeros(Float64, 0, n) : vcat(Aineq_blocks...)
    b = isempty(bineq_blocks) ? Float64[] : vcat(bineq_blocks...)
    C = isempty(Aeq_blocks) ? spzeros(Float64, 0, n) : vcat(Aeq_blocks...)
    d = isempty(deq_blocks) ? Float64[] : vcat(deq_blocks...)

    if !issymmetric(Q)
        @warn "Q is not symmetric"
    end
    return Q, q, A, b, C, d, c
end