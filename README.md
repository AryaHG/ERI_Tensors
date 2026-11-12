# ERI Tensors

Julia research code for representing molecular Hamiltonians as matrix
product operators (MPOs) and calculating ground-state energies using
the density matrix renormalization group (DMRG).

The workflow consists of:

1. Restricted Hartree–Fock (RHF).
2. Calculation of one- and two-electron integrals.
3. Transformation from atomic orbitals to molecular orbitals.
4. Jordan–Wigner encoding into Pauli operators.
5. MPO construction and validation.
6. DMRG with electron-number penalties.

The molecular calculations construct the full electronic Hamiltonian
plus nuclear repulsion.

## Project team

This project is carried out in the Whitfield Group at Dartmouth College.

- **Arya Haghighian** — Ph.D. student, Dartmouth College
- **Weishi Wang** — Postdoctoral researcher, Dartmouth College
- **Prof. James D. Whitfield** — Research advisor, Dartmouth College

## Installation

Install Julia, then clone or download this repository.

From the repository root—the directory containing `Project.toml`—run:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs the environment recorded in `Project.toml` and
`Manifest.toml`.

The main dependencies are Quiqbox, Paulimorphic, ITensors,
ITensorMPS, ITensorMPOConstruction, and Plots.

## Examples

Run each example in a separate Julia process from the repository root.

### Two-qubit MPO example

```bash
julia --project=. experiments/03_mpo_toy.jl
```

Constructs the Hamiltonian

```math
H = X \otimes X + \frac{1}{2} Z \otimes Z
```

using two approaches:

- Successive SVDs of a dense operator.
- Conversion of a Paulimorphic PauliSum into an ITensorMPS MPO.

The example reconstructs the dense operator from both representations
and checks agreement with the original matrix.

### Methane: RHF and MPO construction

```bash
julia --project=. experiments/02_ch4_ccpvdz.jl
```

Uses CH₄ in the cc-pVDZ basis. The script calculates molecular
integrals, constructs the Jordan–Wigner Hamiltonian and its MPO,
and compares their Hartree–Fock expectation values.

This example does not run DMRG.

### Water: ground-state DMRG

```bash
julia --project=. experiments/04_H2O_6-31g.jl
```

Uses H₂O in the 6-31G basis. After constructing and checking the MPO,
the script runs 12 DMRG sweeps with an MPS bond-dimension schedule
of `[32, 64, 128, 256, 512]` and truncation cutoff `1e-8`.

Separate penalties target five alpha and five beta electrons.
The script reports the physical energy and final penalty expectations.

Molecular calculations can require substantial memory and runtime.
Start with the two-qubit example.

## Conventions

- Input coordinates are in Ångström and are converted to Bohr.
- Energies are in Hartree.
- Spin orbitals use interleaved ordering:
  `1α, 1β, 2α, 2β, …`.
- Qubit state `"1"` denotes an occupied spin orbital;
  `"0"` denotes an empty one.
- Both molecular examples use neutral, closed-shell,
  ten-electron systems.

DMRG optimizes the molecular Hamiltonian with added penalties:

```math
H_{\mathrm{opt}}
=
H
+
\lambda(N_\alpha-5)^2
+
\lambda(N_\beta-5)^2,
\qquad \lambda=10.
```

These penalties favor the target electron sector. Their final
expectations must be checked to assess electron-number leakage.
The reported physical energy excludes the artificial penalties.

## Code organization

| Path | Purpose |
| --- | --- |
| `experiments/` | Molecular calculations and illustrative examples |
| `scripts/pauli_mpo.jl` | PauliSum-to-OpSum conversion |
| `scripts/mpo.jl` | Dense SVD-based MPO decomposition for small systems |
| `docs/` | Project notes and documentation |
| `results/` | Local calculation outputs |
| `setup.jl` | Script for adding the project dependencies |

## Development status

This repository contains research scripts under active development.
The cluster MPO-compression experiment is still being organized and
is not part of the examples documented above.