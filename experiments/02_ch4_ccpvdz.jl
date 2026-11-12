# Methane Hamiltonian construction in the cc-pVDZ basis
#
# Workflow:
# 1. Run restricted Hartree–Fock (RHF).
# 2. Transform one- and two-electron integrals to molecular orbitals.
# 3. Encode the total Hamiltonian using Jordan–Wigner.
# 4. Construct an MPO and compare its HF expectation with the RHF energy.
#
# Coordinates are specified in Ångström; energies are in Hartree.
#
# Run from the repository root:
# julia --project=. experiments/02_ch4_ccpvdz.jl


using Quiqbox
using LinearAlgebra
using Paulimorphic
include(joinpath(@__DIR__, "scripts", "pauli_mpo.jl"))



# Molecular specification: neutral, closed-shell methane (10 electrons).
nuc = [:C, :H, :H, :H, :H]
basis_name = "cc-pVDZ"
n_occupied = 5  # Doubly occupied spatial orbitals in the RHF reference.


# Nuclear coordinates in Ångström, in the same order as nuc.
nucCoords_angstrom = [(0.0, 0.0, 0.0), (0.63, 0.63, 0.63), (-0.63, -0.63, 0.63), (-0.63, 0.63, -0.63), (0.63, -0.63, -0.63)]


# Convert coordinates to Bohr for the integral calculation.
nucCoords = [
    Tuple(x / 0.529177 for x in r)
    for r in nucCoords_angstrom
]


# Build the Gaussian orbital basis.
bs = reduce(vcat, genGaussTypeOrbSeq.(nucCoords, nuc, basis_name))

# Run restricted Hartree–Fock for neutral, closed-shell methane.
molecule = NuclearCluster(nuc, nucCoords)
resRHF = runHartreeFock(molecule, bs)

# RHF molecular-orbital coefficients: columns represent spatial orbitals.
C = resRHF.coeff[1]

# One-electron Hamiltonian and electron-repulsion integrals in the AO basis.
h_AO = coreHamiltonian(molecule, bs)
g_AO = elecRepulsions(bs)

# Transform the integrals from atomic orbitals (AO) to molecular orbitals (MO).
h_MO = changeOrbitalBasis(h_AO, C)
g_MO = changeOrbitalBasis(g_AO, C)

# Check their dimensions 
@show size(C)
@show size(h_MO)
@show size(g_MO)

# Check that the molecular orbitals are orthonormal
S_AO = overlaps(bs)
orthogonality_error = norm(C' * S_AO * C - I)
@show orthogonality_error
@assert orthogonality_error < 1e-10 "The molecular orbitals are not orthonormal."

# Reconstruct the electronic RHF energy for methane
occupied = 1:n_occupied

E_reconstructed =
    2 * sum(h_MO[i, i] for i in occupied) +
    sum(
        2 * g_MO[i, i, j, j] - g_MO[i, j, j, i]
        for i in occupied, j in occupied
    )

E_HF = resRHF.energy[1]

@show E_reconstructed E_HF
@show abs(E_reconstructed - E_HF)

@assert isapprox(E_reconstructed, E_HF; atol=1e-8, rtol=0) """
The reconstructed electronic RHF energy does not match the RHF result.
"""

# Number of spatial orbitals and spin-orbital modes
K = size(h_MO, 1)
n_modes = 2 * K


# Construct the Jordan-Wigner representation of the fermionic operators
jw_majorana = genJordanWignerEnc(n_modes)
jw_dirac = toDiracEnc(Float64, jw_majorana)


# Separate annihilation and creation operators
annihilation_ops = jw_dirac.first
creation_ops = jw_dirac.second

# Interval spin ordering: 1α, 1β, 2α, 2β, ...
alpha_encoding = annihilation_ops[1:2:end] => creation_ops[1:2:end]
beta_encoding = annihilation_ops[2:2:end] => creation_ops[2:2:end]
spin_encoding = (alpha_encoding, beta_encoding)

# Check and remove small numerical violations of real-orbital integral symmetries.
    # Check the size of the one-body symmetry error
    @show maximum(abs, h_MO - h_MO')

    # Only proceed if the discrepancy is numerically small
    @assert isapprox(h_MO, h_MO'; rtol=1e-10, atol=1e-12) """
    The one-body matrix has a significant Hermiticity error.
    """

    # Average symmetric entries and enforce exact symmetry
    h_MO = Matrix(Symmetric((h_MO + h_MO') / 2))

    # Enforce the three generators of real-orbital ERI symmetry
    for permutation in (
        (2, 1, 3, 4),  # (pq|rs) = (qp|rs)
        (1, 2, 4, 3),  # (pq|rs) = (pq|sr)
        (3, 4, 1, 2),  # (pq|rs) = (rs|pq)
    )
        g_permuted = permutedims(g_MO, permutation)

        @show permutation maximum(abs, g_MO - g_permuted)

        @assert isapprox(g_MO, g_permuted; rtol=1e-10, atol=1e-12) """
        The ERI tensor has a significant symmetry error.
        """

     g_MO .= (g_MO .+ g_permuted) ./ 2
    end

# Encode the electronic Hamiltonian 
H_electronic = encodeElecHam(
    NormalOrder(),
    spin_encoding,
    (h_MO, g_MO)
)

# Add nuclear repulsion 
E_nuclear = resRHF.energy[2]
H_nuclear = PauliSum([PauliStr(n_modes)], [E_nuclear])

H_total = H_electronic + H_nuclear

# Inspect the results
@show countSites(H_total)
@show countTerms(H_total)
@show isHermitian(H_total)


"""
    occupation_energy(H, occupied_bits)

Evaluate a PauliSum on an occupation-basis product state.
Each true entry of occupied_bits denotes an occupied spin orbital.

Terms containing X or Y have zero expectation. Each Z acting on an
occupied site contributes a factor of -1.
"""

function occupation_energy(H, occupied_bits)
    @assert length(occupied_bits) == countSites(H)
    
    energy = 0.0 + 0.0im

    for (pauli_string, coefficient) in collectTerms(H)
        symbols = toString(pauli_string; denseString =true)
        sign = 1

        for (j, symbol) in enumerate(symbols)
            if symbol == 'X' || symbol == 'Y'
                sign = 0 
                break
            elseif symbol == 'Z'
                if occupied_bits[j]
                    sign = -sign
                end
            elseif symbol != 'I'
                error("Unexpected Pauli-string format: $symbols")
            end
        end

        energy += coefficient * sign
    end

    return energy
end


# Energy check on the encoded Hamiltonian

hf_bits = falses(n_modes)
hf_bits[1:(2 * n_occupied)] .= true

E_encoded = occupation_energy(H_total, hf_bits)
E_reference = sum(resRHF.energy)

@show abs(E_encoded - E_reference)
@assert isapprox(E_mpo, E_encoded; atol=1e-8, rtol=0) """
The MPO does not reproduce the encoded Hamiltonian's HF expectation.
"""

# Constructing the MPO, using ITensors package with a function defined in pauli_mpo.jl file

n_sites = Paulimorphic.countSites(H_total)
sites_total = ITensorMPS.siteinds("Qubit", n_sites)
@time os_total = paulisum_to_opsum(H_total)

@time H_mpo_total = ITensorMPOConstruction.MPO_new(
    os_total,
    sites_total;
    alg = "QR",
    output_level = 1
);

@show length(H_mpo_total)
@show ITensorMPS.maxlinkdim(H_mpo_total)


# Checking the energy conversion in MPO representation
hf_states = [j <= 2 * n_occupied ? "1" : "0" for j in 1:n_sites]
psi_hf = ITensorMPS.MPS(sites_total, hf_states)

E_mpo = ITensors.inner(psi_hf', H_mpo_total, psi_hf)

@show norm(psi_hf)
@show E_encoded E_mpo E_reference
@show abs(E_mpo - E_encoded)
