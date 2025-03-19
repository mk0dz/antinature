# test_relativistic.py

import os
import sys

import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature.core.basis import GaussianBasisFunction, MixedMatterBasis
from antinature.core.molecular_data import MolecularData
from antinature.specialized.relativistic import RelativisticCorrection


def create_test_hamiltonian():
    """Create a minimal test Hamiltonian."""
    n_basis = 2

    # Create simple one-electron matrices
    H_core = np.array([[-1.0, -0.1], [-0.1, -0.8]])
    S = np.array([[1.0, 0.1], [0.1, 1.0]])

    return {
        'H_core_electron': H_core.copy(),
        'H_core_positron': H_core.copy(),
        'overlap': S,
        'nuclear_attraction_e': H_core.copy(),
        'nuclear_attraction_p': H_core.copy(),
        'kinetic_e': np.eye(n_basis) * 0.5,
        'kinetic_p': np.eye(n_basis) * 0.5,
    }


def create_test_basis():
    """Create a minimal test basis set."""
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


def create_test_molecule():
    """Create a test molecular data object."""
    # Hydrogen atom at origin
    atoms = [('H', np.array([0.0, 0.0, 0.0]))]

    # Create molecular data
    molecule = MolecularData(
        atoms=atoms,
        n_electrons=1,
        n_positrons=1,
        charge=0,
        name="Test Positronium",
        is_positronium=True,
    )

    return molecule


def create_test_wavefunction():
    """Create a test wavefunction with density matrices."""
    n_basis = 2

    # Simple density matrices
    P_e = np.array([[0.8, 0.2], [0.2, 0.2]])
    P_p = np.array([[0.7, 0.3], [0.3, 0.3]])

    return {'P_electron': P_e, 'P_positron': P_p, 'n_electrons': 1, 'n_positrons': 1}


def test_relativistic_corrections():
    """Test the relativistic corrections module."""
    # Create test objects
    hamiltonian = create_test_hamiltonian()
    basis = create_test_basis()
    molecule = create_test_molecule()
    wavefunction = create_test_wavefunction()

    print("Testing RelativisticCorrection with different methods:")

    # Test with different correction types
    for correction_type in ['perturbative', 'zora', 'dkh1', 'dkh2']:
        print(f"\nTesting {correction_type} corrections:")

        # Create relativistic correction object
        rel_corr = RelativisticCorrection(
            hamiltonian=hamiltonian,
            basis_set=basis,
            molecular_data=molecule,
            correction_type=correction_type,
        )

        # Calculate integrals
        matrices = rel_corr.calculate_relativistic_integrals()
        print(f"  Mass-velocity matrix shape: {matrices['mass_velocity_e'].shape}")
        print(f"  Darwin matrix shape: {matrices['darwin_e'].shape}")
        if 'spin_orbit_e' in matrices:
            print(f"  Spin-orbit matrix shape: {matrices['spin_orbit_e'].shape}")

        # Apply corrections
        relativistic_terms = rel_corr.apply_relativistic_corrections()
        print(f"  Original H_core_e[0,0]: {hamiltonian['H_core_electron'][0,0]:.6f}")
        print(
            f"  Mass velocity correction[0,0]: {relativistic_terms['mass_velocity_e'][0,0]:.6f}"
        )
        print(f"  Darwin correction[0,0]: {relativistic_terms['darwin_e'][0,0]:.6f}")

        # Calculate corrected Hamiltonian manually (for display purposes)
        corrected_h_core = (
            hamiltonian['H_core_electron']
            + relativistic_terms['mass_velocity_e']
            + relativistic_terms['darwin_e']
        )
        print(f"  Corrected H_core_e[0,0]: {corrected_h_core[0,0]:.6f}")

        # Calculate energy corrections
        energy_corrections = rel_corr.calculate_relativistic_energy_correction(
            wavefunction
        )
        print(
            f"  Mass-velocity correction: {energy_corrections['mass_velocity']['total']:.8f}"
        )
        print(f"  Darwin correction: {energy_corrections['darwin']['total']:.8f}")
        if 'spin_orbit' in energy_corrections:
            print(
                f"  Spin-orbit correction: {energy_corrections['spin_orbit']['total']:.8f}"
            )
        print(f"  Total correction: {energy_corrections['total']:.8f}")

        # Check timing
        for key, value in rel_corr.timing.items():
            print(f"  Timing - {key}: {value:.4f} seconds")


if __name__ == "__main__":
    test_relativistic_corrections()
