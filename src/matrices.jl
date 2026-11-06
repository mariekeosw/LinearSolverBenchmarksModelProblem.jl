################################################################################
# Matrix Construction
################################################################################
# Sparse matrix generation on  a n-dimensional rectangular grid. After
# https://discourse.julialang.org/t/seven-lines-of-julia-examples-sought/50416/135
# by A. Braunstein.
A ⊕ B = kron(I(size(B, 1)), A) + kron(B, I(size(A, 1)))

function lattice(n; Tv = Float64)
    d = fill(2 * one(Tv), n)
    d[1] = one(Tv)
    d[end] = one(Tv)
    return spdiagm(1 => -ones(Tv, n - 1), 0 => d, -1 => -ones(Tv, n - 1))
end

lattice(L...; Tv = Float64) = lattice(L[1]; Tv) ⊕ lattice(L[2:end]...; Tv)

# Create a matrix with the structure of a cell-centered finite volume discretization of
# ``-Δu + δu`` with homogeneous Neumann boundary conditions in the `dim`-dimensional unit cube
# with approximately N unknowns: n = ceil(N^(1/dim)) cells per direction, mesh size h = 1/n.
#
# Default (`scaled = false`): L + δI, as in benchmarks/sparselu.jl of LinearSolve.jl. The
# reaction term is not scaled with h², so the matrix is strictly diagonally dominant by the
# fixed margin δ and its condition number is bounded by (4dim + δ)/δ independently of N.
#
# Alternative (`scaled = true`): L + δh²I, which is the finite volume matrix up to the factor
# h^(dim-2). Its condition number grows like 4dim/(δh²). In 1D, the solution is then only
# accurate to about ε_mach * 4/(δh²) for any solver, i.e. useless from N ≈ 10^5 on.

# Creates a symmetric matrix
function fdmatrix_sym(N; dim = 2, Tv = Float64, δ = 1.0e-2, scaled = false)
    n = N^(1 / dim) |> ceil |> Int
    reaction = scaled ? δ / n^2 : δ
    A = lattice([n for i in 1:dim]...; Tv) + Tv(reaction) * I
    return A
end

# Creates a reproducible non-symmetric matrix.
# The diagonal entries of D are randomly selected from
# {0.8, 0.801, ..., 1.2} using a fixed random seed.
function fdmatrix_nsym(N; dim = 2, Tv = Float64, δ = 1.0e-2, scaled = false, skew = 0.2, rng_seed = 123)
    n = N^(1 / dim) |> ceil |> Int
    rng = MersenneTwister(rng_seed)
    diagonal_entries = rand(rng, (1 - skew):0.001:(1 + skew), n^dim)
    D = Diagonal(Tv.(diagonal_entries))
    reaction = scaled ? δ / n^2 : δ
    A = lattice([n for i in 1:dim]...; Tv) * D + Tv(reaction) * I
    return A
end
