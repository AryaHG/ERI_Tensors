# Two-qubit MPO construction example
#
# Hamiltonian: H = X ⊗ X + 0.5 Z ⊗ Z
#
# This example demonstrates two approaches:
# 1. Decompose a dense operator into MPO tensors using successive SVDs.
# 2. Convert a Paulimorphic PauliSum to an ITensorMPS MPO.
#
# Reconstruct the dense operator from each representation and check
# agreement with the original Hamiltonian.
#
# Run from the repository root:
# julia --project=. experiments/03_mpo_toy.jl

using LinearAlgebra
include(joinpath(@__DIR__, "..", "scripts", "mpo.jl"))
include(joinpath(@__DIR__, "..", "scripts", "pauli_mpo.jl"))



# single-qubit Pauli matrices
X = [0.0 1.0;
1.0 0.0]

Z = [1.0 0.0;
0.0 -1.0]

#Two-qubit Hamiltonian
H_toy = kron(X, X) + 0.5 * kron(Z, Z)

@show size(H_toy)
display(H_toy)

# Expose the four individual qubit indices
H_tensor = reshape(H_toy, 2, 2, 2, 2)

# Pair each site's output and input indices.
H_grouped = permutedims(H_tensor, (2, 4, 1, 3))

# Group the first site's pair into rows and the second site's pair into columns.
H_split = reshape(H_grouped, 4, 4)

display(H_split)

# Apply SVD across the bond between the two sites.
F = svd(H_split)

U = F.U
singular_values = F.S
Vt = F.Vt

@show singular_values

# Verify that the factors reproduce H_split
reconstruction_error =
    norm(H_split - U * Diagonal(singular_values) * Vt)

@show reconstruction_error

T = group_operator_indices(H_toy, 2)
mpo = tensor_to_mpo(T)

@show size(mpo[1])
@show size(mpo[2])


# Our existing toy Hamiltonian, now stored as a PauliSum
H_pauli_toy = Paulimorphic.PauliSum(
    [
        Paulimorphic.PauliStr(2, Paulimorphic.symX),
        Paulimorphic.PauliStr(2, Paulimorphic.symZ)
    ],
    [1.0, 0.5]
)

os_toy = paulisum_to_opsum(H_pauli_toy)

#Two qubit sites 
sites_toy = ITensorMPS.siteinds("Qubit", 2)

#Construct the MPO directly from the operator sum 
H_mpo_toy = ITensorMPOConstruction.MPO_new(
    os_toy,
    sites_toy;
    alg = "QR"
)

@show length(H_mpo_toy)
@show ITensorMPS.maxlinkdim(H_mpo_toy)


H_tensor_check = H_mpo_toy[1] * H_mpo_toy[2]

# Reverse site order within rows and columns to match Julia's
# column-major storage and our original kron basis ordering
H_matrix_check = reshape(
    Array(
        H_tensor_check,
        ITensors.prime(sites_toy[2]),
        ITensors.prime(sites_toy[1]),
        sites_toy[2],
        sites_toy[1]
    ),
    4, 4
)

@show norm(H_matrix_check - H_toy)

@assert isapprox(H_matrix_check, H_toy; atol=1e-10, rtol=0) """
PauliSum-to-MPO conversion did not reproduce the toy Hamiltonian.
"""