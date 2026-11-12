using Pkg
Pkg.activate(@__DIR__)
Pkg.status()

using Serialization
using Quiqbox
using LinearAlgebra
using Paulimorphic
using Plots
using ITensors
using ITensorMPS
using ITensorMPOConstruction
include(joinpath(@__DIR__, "scripts", "pauli_mpo.jl"))

BLAS.set_num_threads(
    parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
)

@show BLAS.get_num_threads()

# 1. Specify the three atoms
nuc = [:O, :H, :H]


# 2. Specify their coordinates, in the same order
nucCoords_angstrom = [(0.0, 0.0, -0.0644484), (0.7499151, 0.0, 0.5114913), (-0.7499151, 0.0, 0.5114913)]


# 2.5 Converting angstrom to Bohr
nucCoords = [
    Tuple(x / 0.52917721092 for x in r)
    for r in nucCoords_angstrom
]


# 3. Generate the cc-pVDZ basis
bs = reduce(vcat, genGaussTypeOrbSeq.(nucCoords, nuc, "6-31G"))

p1 = uniqueParams(bs[1])[3]


# 4. Run restricted Hartree-Fock
resRHF = runHartreeFock(NuclearCluster(nuc, nucCoords), bs)


# 5. Display the results
#@show resRHF.energy
#@show resRHF.coeff
#@show resRHF.occu

# 6. Extract the HF molecular-orbital coefficients
C = resRHF.coeff[1]

# 7. Calculate integrals in the original
h_AO = coreHamiltonian(NuclearCluster(nuc, nucCoords), bs)
g_AO = elecRepulsions(bs)



# 8. Transform AO to the HF molecular-orbital basis
h_MO = changeOrbitalBasis(h_AO, C)
g_MO = changeOrbitalBasis(g_AO, C)

# 9. Check their dimensions 
@show size(C)
@show size(h_MO)
@show size(g_MO)

# 10. Check that the molecular orbitals are orthonormal
S_AO = overlaps(bs)
orthogonality_error = norm(C' * S_AO * C - I)
@show orthogonality_error

# 11. Reconstruct the electronic RHF energy for methane
occupied = 1:5

E_reconstructed =
    2 * sum(h_MO[i, i] for i in occupied) +
    sum(
        2 * g_MO[i, i, j, j] - g_MO[i, j, j, i]
        for i in occupied, j in occupied
    )

E_HF = resRHF.energy[1]

@show E_reconstructed E_HF
@show abs(E_reconstructed - E_HF)

# 12. Number of spatial orbitals and spin-orbital modes
K = size(h_MO, 1)
n_modes = 2 * K


# 13. Construct the Jordan-Wigner representation of the fermionic operators
jw_majorana = genJordanWignerEnc(n_modes)
jw_dirac = toDiracEnc(Float64, jw_majorana)


# 14. Separate annihilation and creation operators
annihilation_ops = jw_dirac.first
creation_ops = jw_dirac.second

# 15. Interval spin ordering: 1α, 1β, 2α, 2β, ...
alpha_encoding = annihilation_ops[1:2:end] => creation_ops[1:2:end]
beta_encoding = annihilation_ops[2:2:end] => creation_ops[2:2:end]
spin_encoding = (alpha_encoding, beta_encoding)

# 15.5 Some symmetry Fixing
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


# 16. Encode the electronic Hamiltonian 
H_electronic = encodeElecHam(
    NormalOrder(),
    spin_encoding,
    (h_MO, g_MO)
)

# 17. Add nuclear repulsion 
E_nuclear = resRHF.energy[2]
H_nuclear = PauliSum([PauliStr(n_modes)], [E_nuclear])

H_total = H_electronic + H_nuclear

# 18. Inspect the results
@show countSites(H_total)
@show countTerms(H_total)
@show isHermitian(H_total)


# 19. Expectation value of a canonical PauliSum on occupation bitstring
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


# 20. Energy check on the encoded Hamiltonian

hf_bits = falses(n_modes)
hf_bits[1:10] .= true

E_encoded = occupation_energy(H_total, hf_bits)
E_refrence = sum(resRHF.energy)

@show abs(E_encoded - E_refrence)
@assert isapprox(E_encoded,E_refrence; atol=1e-8, rtol=0)


# 21. Constructing the MPO, using ITensors package with a function defined in pauli_mpo.jl file

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

# Preserve the uncompressed Hamiltonian
H_mpo_reference = deepcopy(H_mpo_total)

# Six compression cases
mpo_maxdims = [250, 190, 130, 100, 60, 30]

# 22. Checking the energy conversion in MPO representation
hf_states = [j <= 10 ? "1" : "0" for j in 1:n_sites]
psi_hf = ITensorMPS.MPS(sites_total, hf_states)

E_mpo = ITensorMPS.inner(psi_hf', H_mpo_total, psi_hf)
E_reference = sum(resRHF.energy)

@show norm(psi_hf)
@show E_encoded E_mpo E_reference
@show abs(E_mpo - E_encoded)

@show size(H_mpo_total[23])


#--------------------------DMRG Calculate-------------------

#  Fixing the Hamiltonian to be commutative with number operator
# 23. Specify the electron-number targets
alpha_sites = collect(1:2:n_sites)
beta_sites  = collect(2:2:n_sites)

N_alpha_target = 5
N_beta_target  = 5
penalty_strength = 10.0


# 24. Define the penalty-construction function

function number_penalty_mpo(
    sites,
    selected_sites,
    target;
    strength = 10.0,
)
    selected = collect(selected_sites)
    os = ITensorMPS.OpSum()

    # Constant contribution
    os += strength * target^2, "I", 1

    # Contributions involving one occupation operator
    for j in selected
        os += strength * (1 - 2 * target), "Proj1", j
    end

    # Contributions involving pairs of occupation operators
    for a in 1:(length(selected) - 1)
        for b in (a + 1):length(selected)
            j = selected[a]
            k = selected[b]

            os += 2 * strength, "Proj1", j, "Proj1", k
        end
    end

    return ITensorMPS.MPO(os, sites; cutoff = 1e-14)
end

# 25. Constructing the penalty
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

# 26. Checking the results after penalty
P_alpha_HF = ITensorMPS.inner(psi_hf', P_alpha, psi_hf)
P_beta_HF  = ITensorMPS.inner(psi_hf', P_beta, psi_hf)

@show P_alpha_HF P_beta_HF

@assert abs(P_alpha_HF) < 1e-10
@assert abs(P_beta_HF) < 1e-10

function relative_hermiticity_error(H)
    H_adjoint = ITensorMPS.MPO([
        ITensors.swapprime(
            ITensors.dag(H[j]),
            0 => 1;
            tags = "Site",
        )
        for j in 1:length(H)
    ])

    difference = ITensorMPS.add(
        H, -1.0 * H_adjoint;
        alg = "directsum",
    )

    ITensorMPS.orthogonalize!(difference, 1)

    return norm(difference) / norm(H)
end

let
    for mpo_maxdim in mpo_maxdims
        println("\n===== Starting MPO cap: $mpo_maxdim =====")
        flush(stdout)

        # Each case starts from the same uncompressed MPO
        H_mpo_total = deepcopy(H_mpo_reference)

        ITensorMPS.truncate!(
            H_mpo_total;
            maxdim = mpo_maxdim,
            cutoff = 0.0,
        )

        hermiticity_error = relative_hermiticity_error(H_mpo_total)

        @show mpo_maxdim
        @show ITensorMPS.maxlinkdim(H_mpo_total)
        @show hermiticity_error

        if !isfinite(hermiticity_error) || hermiticity_error > 1e-10
            @warn "Skipping this cap: compressed MPO failed Hermiticity check" mpo_maxdim
            continue
        end

        # HF energy for this compressed Hamiltonian
        E_mpo = ITensorMPS.inner(
            psi_hf', H_mpo_total, psi_hf,
        )

        @show E_mpo
        @show abs(E_mpo - E_encoded)

   

        # 27. Preparing the Hamiltonian for DMRG
        H_for_dmrg = [H_mpo_total, P_alpha, P_beta]


        # 28. Main Eenergy refrences
        E_HF_total = sum(resRHF.energy)
        E_FCI_paper = -76.11969704


        # 29. First DMRG Calculation
        dmrg_cutoff = 1e-8

        E_dmrg_opt, psi_dmrg = ITensorMPS.dmrg(
            H_for_dmrg,
            deepcopy(psi_hf);
            nsweeps = 15,
            maxdim = [32, 64, 128, 256, 512],
            cutoff = dmrg_cutoff,
            noise = [1e-6, 1e-7, 1e-8, 0.0],
            eigsolve_krylovdim = 8,
            eigsolve_maxiter = 4,
            outputlevel = 1,
        );


        #---------------------------Submit Job-----------------------------

        job_id = get(ENV, "SLURM_JOB_ID", "manual")
        result_file = joinpath(
            @__DIR__,
            "h2o_compression_$(job_id)_W$(mpo_maxdim).jls",
        )

        norm2 = real(ITensorMPS.inner(psi_dmrg, psi_dmrg))

        # Evaluate the same state with both Hamiltonians
        E_compressed = ITensorMPS.inner(
            psi_dmrg', H_mpo_total, psi_dmrg,
        ) / norm2

        E_original = ITensorMPS.inner(
            psi_dmrg', H_mpo_reference, psi_dmrg,
        ) / norm2

        P_alpha_final = ITensorMPS.inner(
            psi_dmrg', P_alpha, psi_dmrg,
        ) / norm2

        P_beta_final = ITensorMPS.inner(
            psi_dmrg', P_beta, psi_dmrg,
        ) / norm2

        mpo_actual = ITensorMPS.maxlinkdim(H_mpo_total)
        mps_actual = ITensorMPS.maxlinkdim(psi_dmrg)

        @show E_dmrg_opt E_compressed E_original
        @show mpo_actual mps_actual
        @show P_alpha_final P_beta_final

        serialize(result_file, (
            mpo_cap = mpo_maxdim,
            mpo_actual = mpo_actual,
            mps_actual = mps_actual,
            energy_penalized = E_dmrg_opt,
            energy_compressed = E_compressed,
            energy_original = E_original,
            alpha_penalty = P_alpha_final,
            beta_penalty = P_beta_final,
            hf_energy = E_HF_total,
            hf_compressed_energy = E_mpo,
            hf_encoded_energy = E_encoded,
            hermiticity_error = hermiticity_error,
            energy_error_Eh = real(E_original) - E_FCI_paper,
            alpha_number_mse = real(P_alpha_final) / penalty_strength,
            beta_number_mse = real(P_beta_final) / penalty_strength,
            nsweeps = 15,
            mps_cutoff = dmrg_cutoff,
            mps_cap = 512,
            state = psi_dmrg,
        ))

        println("Saved results to: ", result_file)

    end
end