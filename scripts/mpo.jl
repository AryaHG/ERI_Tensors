using LinearAlgebra

"""
    group_operator_indices(H::AbstractMatrix, N::Int)

Reshape an N-qubit operator matrix into a tensor with indices ordered as
(output₁, input₁, output₂, input₂, …, outputₙ, inputₙ).

H must have dimensions (2^N, 2^N). Site 1 corresponds to the leftmost
factor in the Kronecker-product basis ordering.

Return a rank-2N tensor with every dimension equal to 2.
"""

function group_operator_indices(H::AbstractMatrix, N::Int)
    @assert N >= 1 "The number of qubit sites must be positive."
    @assert size(H) == (2^N, 2^N) "H must be a 2^N × 2^N matrix."

    tensor = reshape(H, ntuple(_ -> 2, 2 * N))

    permutation = Int[]

    for j in 1:N
        push!(permutation, N - j + 1)
        push!(permutation, 2 * N - j + 1)
    end

    return permutedims(tensor, permutation)
end

"""
    tensor_to_mpo(T; rtol=1e-12, maxbond=typemax(Int))

Decompose a tensor with paired qubit indices into an MPO using
successive singular value decompositions (SVDs), from left to right.

T must have indices ordered as
(output₁, input₁, …, outputₙ, inputₙ), each of dimension 2.

At each split, retain singular values greater than rtol times the
largest singular value, keeping at most maxbond values.
Always retain at least one bond component.

Return a vector of rank-four site tensors with index order
(left bond, output, input, right bond). The outer boundary bonds
have dimension 1.

This implementation starts from a dense tensor and is intended
for small examples.
"""

function tensor_to_mpo(
    T::AbstractArray;
    rtol::Real = 1e-12,
    maxbond::Int = typemax(Int)
)

   # Validate the inputs
    @assert ndims(T) >= 2 && iseven(ndims(T)) """ T must have an even number of indices, with at least one output–input pair. """
    @assert all(d -> d == 2, size(T)) """ Every tensor dimension must equal 2 for qubit sites."""
    @assert 0 <= rtol <1 "rtol must satisfy 0 ≤ rtol < 1."
    @assert maxbond >= 1 "maxbond must be at least 1."

    N = ndims(T) ÷ 2

# Initialize the remaining tensor
    remainder = float.(T)
    D_left = 1

    # Allocate the MPO site tensors
    mpo = Vector{Array{eltype(remainder), 4}}(undef, N)

    for j in 1:(N-1)
        M = reshape(remainder, D_left * 4, :)
        F = svd(M)
        threshold = rtol * F.S[1]
        D_right = count(s -> s > threshold, F.S)
        D_right = max(1, min(D_right, maxbond))

        U_keep = F.U[:, 1:D_right]
        mpo[j] = reshape(U_keep, D_left, 2, 2, D_right)

        remainder = Diagonal(F.S[1:D_right]) * F.Vt[1:D_right, :]

        D_left = D_right
    end

    mpo[N] = reshape(remainder, D_left, 2, 2, 1)

    return mpo
end

