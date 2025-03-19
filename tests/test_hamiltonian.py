# test_hamiltonian.py
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from antinature.core.basis import (
    BasisSet,
    GaussianBasisFunction,
    MixedMatterBasis,
    PositronBasis,
)
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.molecular_data import MolecularData  # We'll need this class


def test_hamiltonian_construction():
    """Test the construction of Hamiltonian matrices for a simple system."""
    print("Testing Hamiltonian construction for a simple H2 molecule...")

    # Create a minimal system - H2 molecule
    h2_data = MolecularData(
        atoms=[('H', np.array([0.0, 0.0, 0.0])), ('H', np.array([0.74, 0.0, 0.0]))],
        n_electrons=2,
        n_positrons=0,
        charge=0,
    )

    # Create a basis set
    basis = MixedMatterBasis()
    basis.create_for_molecule(
        atoms=h2_data.atoms, e_quality='minimal', p_quality='minimal'
    )

    # Create integral engine
    engine = AntinatureIntegralEngine()

    # Create Hamiltonian
    hamiltonian = AntinatureHamiltonian(
        molecular_data=h2_data,
        basis_set=basis,
        integral_engine=engine,
        include_relativistic=False,
    )

    # Build the Hamiltonian
    print("Building Hamiltonian matrices...")
    h_matrices = hamiltonian.build_hamiltonian()

    # Print key matrix dimensions
    print("\nMatrix dimensions:")
    for key, matrix in h_matrices.items():
        if matrix is not None and hasattr(matrix, 'shape'):
            print(f"  {key}: {matrix.shape}")

    # Check overlap matrix
    if 'overlap' in h_matrices:
        S = h_matrices['overlap']
        print("\nOverlap matrix:")
        print(S)

        # Verify diagonal elements are 1.0
        diag_ok = np.allclose(np.diag(S), 1.0)
        print(f"Diagonal elements check: {'PASSED' if diag_ok else 'FAILED'}")

    # Check core Hamiltonian
    if 'H_core_electron' in h_matrices:
        H_core = h_matrices['H_core_electron']
        print("\nCore Hamiltonian matrix:")
        print(H_core)

        # Verify symmetry
        symm_ok = np.allclose(H_core, H_core.T)
        print(f"Symmetry check: {'PASSED' if symm_ok else 'FAILED'}")

    # Get performance report
    report = hamiltonian.get_performance_report()
    print("\nPerformance Report:")
    for component, time_spent in report['times'].items():
        print(
            f"  {component}: {time_spent:.6f} seconds ({report['percentages'][component]:.1f}%)"
        )
    print(f"  Total time: {report['total_time']:.6f} seconds")

    # Test with positrons
    print("\n\nTesting with positronium system...")

    # Create positronium system
    ps_data = MolecularData(
        atoms=[('H', np.array([0.0, 0.0, 0.0]))],  # Dummy nucleus for positronium
        n_electrons=1,
        n_positrons=1,
        charge=0,
        is_positronium=True,
    )

    # Create a positronium basis
    ps_basis = MixedMatterBasis()
    ps_basis.create_positronium_basis(quality='minimal')

    # Create Hamiltonian for positronium
    ps_hamiltonian = AntinatureHamiltonian(
        molecular_data=ps_data,
        basis_set=ps_basis,
        integral_engine=engine,
        include_annihilation=True,
    )

    # Build the Hamiltonian
    ps_matrices = ps_hamiltonian.build_hamiltonian()

    # Check if annihilation operator is computed
    if 'annihilation' in ps_matrices and ps_matrices['annihilation'] is not None:
        ann = ps_matrices['annihilation']
        print("\nAnnihilation operator matrix:")
        print(ann)
        print(f"Shape: {ann.shape}")

    print("\nHamiltonian tests completed successfully!")


if __name__ == "__main__":
    test_hamiltonian_construction()
