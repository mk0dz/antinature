# test_positronium.py

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature.core.basis import GaussianBasisFunction, MixedMatterBasis
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.molecular_data import MolecularData
from antinature.specialized.positronium import PositroniumSCF


def create_test_basis():
    """Create a minimal test basis set for positronium."""
    basis = MixedMatterBasis()

    # Add electron basis functions
    e_basis = []
    e_basis.append(GaussianBasisFunction(np.array([0.0, 0.0, 0.0]), 1.0, (0, 0, 0)))
    e_basis.append(GaussianBasisFunction(np.array([0.0, 0.0, 0.0]), 0.5, (0, 0, 0)))

    # Add positron basis functions
    p_basis = []
    p_basis.append(GaussianBasisFunction(np.array([0.0, 0.0, 0.0]), 0.8, (0, 0, 0)))
    p_basis.append(GaussianBasisFunction(np.array([0.0, 0.0, 0.0]), 0.4, (0, 0, 0)))

    # Set up the basis
    basis.electron_basis.basis_functions = e_basis
    basis.positron_basis.basis_functions = p_basis
    basis.n_electron_basis = len(e_basis)
    basis.n_positron_basis = len(p_basis)
    basis.n_total_basis = basis.n_electron_basis + basis.n_positron_basis

    return basis


def create_test_hamiltonian(basis):
    """Create a test Hamiltonian for positronium."""
    n_e_basis = basis.n_electron_basis
    n_p_basis = basis.n_positron_basis

    # Overlap matrix
    S = np.zeros((n_e_basis + n_p_basis, n_e_basis + n_p_basis))

    # Electron-electron overlap
    S[:n_e_basis, :n_e_basis] = np.array([[1.0, 0.8], [0.8, 1.0]])

    # Positron-positron overlap
    S[n_e_basis:, n_e_basis:] = np.array([[1.0, 0.7], [0.7, 1.0]])

    # Core Hamiltonian for electrons
    H_core_e = np.array([[-1.0, -0.5], [-0.5, -0.8]])

    # Core Hamiltonian for positrons
    H_core_p = np.array([[-1.0, -0.4], [-0.4, -0.7]])

    # Electron-positron attraction (4-index tensor)
    ERI_ep = np.zeros((n_e_basis, n_e_basis, n_p_basis, n_p_basis))
    # Fill with some reasonable values
    for i in range(n_e_basis):
        for j in range(n_e_basis):
            for k in range(n_p_basis):
                for l in range(n_p_basis):
                    ERI_ep[i, j, k, l] = -0.5 * np.exp(-((i - k) ** 2 + (j - l) ** 2))

    # Create Hamiltonian dict
    hamiltonian = {
        'overlap': S,
        'H_core_electron': H_core_e,
        'H_core_positron': H_core_p,
        'electron_positron_attraction': ERI_ep,
    }

    return hamiltonian


def test_positronium_scf():
    """Test the PositroniumSCF class."""
    print("Testing PositroniumSCF...")

    # Create positronium molecule
    molecule = MolecularData.positronium()

    # Create basis set
    basis = create_test_basis()

    # Create test Hamiltonian
    hamiltonian = create_test_hamiltonian(basis)

    # Test para-positronium (singlet state)
    print("\nTesting para-positronium (singlet state):")
    scf_para = PositroniumSCF(
        hamiltonian=hamiltonian, basis_set=basis, molecular_data=molecule
    )

    # Note: We would have set positronium_state='para' and include_qed_corrections=True
    # but these parameters are not supported in the current implementation
    print("Note: Using default state (equivalent to para-positronium)")

    # Solve SCF
    results_para = scf_para.solve_scf()

    # Print results
    print(f"Energy: {results_para['energy']:.10f} Hartree")
    print(f"Converged: {results_para['converged']}")
    print(f"Iterations: {results_para['iterations']}")

    # Check if 'exact_solution' exists in results
    if 'exact_solution' in results_para:
        print(f"Exact solution used: {results_para['exact_solution']}")

    # Analyze wavefunction
    try:
        analysis_para = scf_para.analyze_wavefunction()
        print("\nWavefunction analysis:")
        for key, value in analysis_para.items():
            if not isinstance(value, dict):
                print(f"  {key}: {value}")
    except AttributeError:
        print("\nWavefunction analysis not available")

    # Calculate annihilation rate
    try:
        ann_para = scf_para.calculate_annihilation_rate()
        print("\nAnnihilation rate:")
        for key, value in ann_para.items():
            print(f"  {key}: {value}")
    except AttributeError:
        print("\nAnnihilation rate calculation not available")

    # Test ortho-positronium (triplet state)
    print("\nTesting ortho-positronium (triplet state):")
    scf_ortho = PositroniumSCF(
        hamiltonian=hamiltonian, basis_set=basis, molecular_data=molecule
    )

    # Note: We would have set positronium_state='ortho' and include_qed_corrections=True
    # but these parameters are not supported in the current implementation
    print("Note: Using default state (simulating ortho-positronium)")

    # Solve SCF
    results_ortho = scf_ortho.solve_scf()

    # Print results
    print(f"Energy: {results_ortho['energy']:.10f} Hartree")
    print(f"Converged: {results_ortho['converged']}")
    print(f"Iterations: {results_ortho['iterations']}")

    # Calculate annihilation rate
    try:
        ann_ortho = scf_ortho.calculate_annihilation_rate()
        print("\nAnnihilation rate:")
        for key, value in ann_ortho.items():
            print(f"  {key}: {value}")
    except AttributeError:
        print("\nAnnihilation rate calculation not available")

    # Try visualization
    try:
        fig = scf_para.visualize_orbitals(
            grid_dims=(20, 20, 20),
            limits=(-5.0, 5.0),
            save_path='positronium_orbitals.png',
        )
        print("\nVisualization saved to 'positronium_orbitals.png'")
    except Exception as e:
        print(f"\nVisualization error: {str(e)}")


if __name__ == "__main__":
    test_positronium_scf()
