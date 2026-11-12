# Water ground-state calculation in the 6-31G basis
#
# Construct the total molecular Hamiltonian as an MPO, validate its
# HF expectation, and run DMRG with electron-number penalties.
#
# Coordinates are in Ångström; energies are in Hartree.
# Target sector: five alpha and five beta electrons.
#
# Run from the repository root:
# julia --project=. experiments/04_H2O_6-31g.jl


using Quiqbox
using LinearAlgebra
using Paulimorphic
using ITensors
include(joinpath(@__DIR__, "..", "scripts", "pauli_mpo.jl"))



# Molecular specification: neutral, closed-shell water (10 electrons).
nuc = [:O, :H, :H]
basis_name = "6-31G"
n_occupied = 5  # Doubly occupied spatial orbitals in the RHF reference.


# Speci# Nuclear coordinates in Ångström, in the same order as nuc.fy their coordinates, in the same order
nucCoords_angstrom = [(0.0, 0.0, -0.0644484), (0.7499151, 0.0, 0.5114913), (-0.7499151, 0.0, 0.5114913)]


# Convert coordinates to Bohr for the integral calculation.
nucCoords = [
    Tuple(x / 0.52917721092 for x in r)
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

#--------------------------DMRG Calculate-------------------
# DMRG with separate alpha- and beta-electron number penalties.
#
# Optimize H + λ(N_alpha - target_alpha)^2 + λ(N_beta - target_beta)^2.
# Check the final penalties afterward to assess electron-number leakage.

# Interleaved spin ordering: odd sites are alpha, even sites are beta.

alpha_sites = collect(1:2:n_sites)
beta_sites  = collect(2:2:n_sites)

N_alpha_target = 5
N_beta_target  = 5
penalty_strength = 10.0


"""
    number_penalty_mpo(sites, selected_sites, target; strength=10.0)

Construct strength * (N_selected - target)^2 as an MPO,
where N_selected is the sum of occupation operators on selected_sites.

For qubit occupation states, n_j = Proj1_j and n_j^2 = n_j.
Expanding the square gives a constant, single-site terms, and
two-site occupation terms.

The penalty vanishes in the target number sector. Its expectation
divided by strength gives the mean squared deviation from target.
"""

function number_penalty_mpo(
    sites,
    selected_sites,
    target;
    strength = 10.0,
)
    selected = collect(selected_sites)

    @assert strength > 0 "Penalty strength must be positive."
    @assert length(unique(selected)) == length(selected) "Selected sites must be unique."
    @assert all(j -> 1 <= j <= length(sites), selected) "Selected site is out of bounds."
    @assert target isa Integer && 0 <= target <= length(selected) "Invalid target occupation."


    os = ITensorMPS.OpSum()

    # Constant term: λ * target^2.
    os += strength * target^2, "I", 1

    # Single-site terms: λ * (1 - 2*target) * n_j.
    for j in selected
        os += strength * (1 - 2 * target), "Proj1", j
    end

    # Pair terms: 2λ * n_j * n_k, with each pair counted once.
    for a in 1:(length(selected) - 1)
        for b in (a + 1):length(selected)
            j = selected[a]
            k = selected[b]

            os += 2 * strength, "Proj1", j, "Proj1", k
        end
    end

    return ITensorMPS.MPO(os, sites; cutoff = 1e-14)
end

# Construct the penalties for the two spin sectors.
P_alpha = number_penalty_mpo(
    sites_total,
    alpha_sites,
    N_alpha_target;
    strength = penalty_strength,
)

P_beta = number_penalty_mpo(
    sites_total,
    beta_sites,
    N_beta_target;
    strength = penalty_strength,
)

# The HF reference has exactly the target electron numbers.
P_alpha_HF = ITensors.inner(psi_hf', P_alpha, psi_hf)
P_beta_HF  = ITensors.inner(psi_hf', P_beta, psi_hf)

@show P_alpha_HF P_beta_HF

@assert abs(P_alpha_HF) < 1e-10 "The alpha penalty does not vanish on the HF state."
@assert abs(P_beta_HF) < 1e-10 "The beta penalty does not vanish on the HF state."


# Passing a vector of MPOs lets DMRG optimize their sum.
H_for_dmrg = [H_mpo_total, P_alpha, P_beta]


# Total energies, including nuclear repulsion.
E_HF_total = sum(resRHF.energy)

# Supplied FCI comparison value; add its source before publication.
E_FCI_paper = -76.11969704


# DMRG settings.
nsweeps = 12
mps_maxdims = [32, 64, 128, 256, 512]
dmrg_cutoff = 1e-8
noise_schedule = [1e-6, 1e-7, 1e-8, 0.0]

# Start from a copy of the HF product state.
E_dmrg_opt, psi_dmrg = ITensorMPS.dmrg(
    H_for_dmrg,
    deepcopy(psi_hf);
    nsweeps = nsweeps,
    maxdim = mps_maxdims,
    cutoff = dmrg_cutoff,
    noise = noise_schedule,
    eigsolve_krylovdim = 8,
    eigsolve_maxiter = 4,
    outputlevel = 1,
)

# Evaluate the final state, normalizing all expectation values.
norm2 = real(ITensorMPS.inner(psi_dmrg, psi_dmrg))
@assert isfinite(norm2) && norm2 > 0 "The final MPS has an invalid norm."

# Physical total energy, excluding the artificial number penalties.
E_physical = ITensorMPS.inner(
    psi_dmrg', H_mpo_total, psi_dmrg,
) / norm2

P_alpha_final = ITensorMPS.inner(
    psi_dmrg', P_alpha, psi_dmrg,
) / norm2

P_beta_final = ITensorMPS.inner(
    psi_dmrg', P_beta, psi_dmrg,
) / norm2

# Mean squared electron-number deviations: ideally close to zero.
alpha_number_mse = real(P_alpha_final) / penalty_strength
beta_number_mse = real(P_beta_final) / penalty_strength

@show E_HF_total E_dmrg_opt E_physical
@show P_alpha_final P_beta_final
@show alpha_number_mse beta_number_mse
@show ITensorMPS.maxlinkdim(psi_dmrg)
