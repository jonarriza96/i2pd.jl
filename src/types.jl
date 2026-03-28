# -------------------------------- QP problem -------------------------------- #
struct QPProblem{T,QT,QTv,AT,BTv,CT,DTv,MT}
    Q::QT
    q::QTv
    A::AT
    b::BTv
    C::CT
    d::DTv
    c::T  # Constant objective term
    M::MT  # Optional matrix (same size as Q)
    n::Int # Number of variables
    m::Int # Number of inequality constraints
    p::Int # Number of equality constraints
end

function QPProblem(
    Q::QT,
    q::QTv,
    A::AT,
    b::BTv,
) where {T,QT<:AbstractMatrix{T},QTv<:AbstractVector{T},AT<:AbstractMatrix{T},BTv<:AbstractVector{T}}
    n = length(q)
    C = zeros(T, 0, n)
    d = Vector{T}()
    return QPProblem(Q, q, A, b, C, d, zero(T))
end

function QPProblem(
    Q::QT,
    q::QTv,
    A::AT,
    b::BTv,
    M::MT,
) where {T,QT<:AbstractMatrix{T},QTv<:AbstractVector{T},AT<:AbstractMatrix{T},BTv<:AbstractVector{T},MT<:AbstractMatrix{T}}
    n = length(q)
    C = zeros(T, 0, n)
    d = Vector{T}()
    return QPProblem(Q, q, A, b, C, d, zero(T), M)
end

function QPProblem(
    Q::QT,
    q::QTv,
    A::AT,
    b::BTv,
    C::CT,
    d::DTv,
) where {
    T,
    QT<:AbstractMatrix{T},
    QTv<:AbstractVector{T},
    AT<:AbstractMatrix{T},
    BTv<:AbstractVector{T},
    CT<:AbstractMatrix{T},
    DTv<:AbstractVector{T},
}
    n = length(q)
    m = length(b)
    p = length(d)

    @assert size(Q) == (n, n)
    @assert size(A) == (m, n)
    @assert size(C) == (p, n)

    return QPProblem(Q, q, A, b, C, d, zero(T))
end

function QPProblem(
    Q::QT,
    q::QTv,
    A::AT,
    b::BTv,
    C::CT,
    d::DTv,
    c::Real,
) where {
    T,
    QT<:AbstractMatrix{T},
    QTv<:AbstractVector{T},
    AT<:AbstractMatrix{T},
    BTv<:AbstractVector{T},
    CT<:AbstractMatrix{T},
    DTv<:AbstractVector{T},
}
    n = length(q)
    m = length(b)
    p = length(d)
    cT = T(c)

    @assert size(Q) == (n, n)
    @assert size(A) == (m, n)
    @assert size(C) == (p, n)

    return QPProblem{T,QT,QTv,AT,BTv,CT,DTv,Nothing}(Q, q, A, b, C, d, cT, nothing, n, m, p)
end

function QPProblem(
    Q::QT,
    q::QTv,
    A::AT,
    b::BTv,
    C::CT,
    d::DTv,
    M::MT,
) where {
    T,
    QT<:AbstractMatrix{T},
    QTv<:AbstractVector{T},
    AT<:AbstractMatrix{T},
    BTv<:AbstractVector{T},
    CT<:AbstractMatrix{T},
    DTv<:AbstractVector{T},
    MT<:AbstractMatrix{T},
}
    n = length(q)
    m = length(b)
    p = length(d)

    @assert size(Q) == (n, n)
    @assert size(A) == (m, n)
    @assert size(C) == (p, n)
    @assert size(M) == (n, n)

    return QPProblem(Q, q, A, b, C, d, zero(T), M)
end

function QPProblem(
    Q::QT,
    q::QTv,
    A::AT,
    b::BTv,
    C::CT,
    d::DTv,
    c::Real,
    M::MT,
) where {
    T,
    QT<:AbstractMatrix{T},
    QTv<:AbstractVector{T},
    AT<:AbstractMatrix{T},
    BTv<:AbstractVector{T},
    CT<:AbstractMatrix{T},
    DTv<:AbstractVector{T},
    MT<:AbstractMatrix{T},
}
    n = length(q)
    m = length(b)
    p = length(d)
    cT = T(c)

    @assert size(Q) == (n, n)
    @assert size(A) == (m, n)
    @assert size(C) == (p, n)
    @assert size(M) == (n, n)

    return QPProblem{T,QT,QTv,AT,BTv,CT,DTv,MT}(Q, q, A, b, C, d, cT, M, n, m, p)
end

struct Solution{TX,TL,TS,TV,TF}
    x::TX
    λ::TL
    s::TS
    v::TV
    f::TF
end

# ---------------------------- Solver enumerations --------------------------- #
@enum SolverStatus begin
    STATUS_UNKNOWN = -1
    STATUS_CONVERGED = 0
    STATUS_MAX_ITER = 1
    STATUS_LINESEARCH_FAILED = 2
end

@enum SolverMode begin
    MODE_UNCONDENSED = 1
    MODE_PARTIALLY_CONDENSED = 2
    MODE_FULLY_CONDENSED = 3
    MODE_UNCONDENSED_IMPLICIT = 4
    MODE_PARTIALLY_CONDENSED_IMPLICIT = 5
end

@enum SolverBackend begin
    BACKEND_DIRECT = 1
    BACKEND_ITERATIVE = 2
end

@enum RetractionMap begin
    RETRACTION_MAP_SOFTPLUS = 1
    RETRACTION_MAP_EXP = 2
end

# ---------------------------------- Indices --------------------------------- #
struct StateIndices
    x::UnitRange{Int}
    λ::UnitRange{Int}
    γ::UnitRange{Int}
    s::UnitRange{Int}
    v::UnitRange{Int}
    implicit::Bool
end

function StateIndices(problem::QPProblem; implicit::Bool=false)
    n, m, p = problem.n, problem.m, problem.p

    return StateIndices(
        1:n,
        n+1:n+m,
        n+m+1:n+m+p,
        n+m+p+1:n+m+p+m,
        (n+m+p+m+1):(n+m+p+m+m),
        implicit,
    )
end