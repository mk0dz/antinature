# test_scf.py
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from antinature.core.basis import MixedMatterBasis
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.molecular_data import MolecularData

# Force reload of the scf module to avoid caching issues
if 'antinature.core.scf' in sys.modules:
    importlib.reload(sys.modules['antinature.core.scf'])
import antinature.core.scf
from antinature.core.scf import AntinatureSCF


def test_scf_solver():
    """Test the SCF solver for various systems."""
    print("Testing SCF solver for H2 molecule...")

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

    # Build the Hamiltonian matrices
    h_matrices = hamiltonian.build_hamiltonian()

    # Create SCF solver
    scf = AntinatureSCF(
        hamiltonian=h_matrices,
        basis_set=basis,
        molecular_data=h2_data,
        max_iterations=50,
        convergence_threshold=1e-6,
        use_diis=True,
        damping_factor=0.5,
        level_shifting=0.0,
        diis_start=3,
        diis_dimension=6,
        print_level=1,
    )

    # Run SCF calculation
    results = scf.solve_scf()

    # Print results
    print("\nSCF Results for H2:")
    print(f"  Converged: {results['converged']}")
    print(f"  Energy: {results['energy']:.10f} Hartree")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Electron orbital energies: {results['E_electron']}")

    # Test positronium system
    print("\n\nTesting with positronium system...")

    # Create positronium system
    ps_data = MolecularData(
        atoms=[('H', np.array([0.0, 0.0, 0.0]))],  # Dummy nucleus
        n_electrons=1,
        n_positrons=1,
        charge=0,
        is_positronium=True,
    )

    # Create positronium basis
    ps_basis = MixedMatterBasis()
    ps_basis.create_positronium_basis(quality='minimal')

    # Create Hamiltonian
    ps_hamiltonian = AntinatureHamiltonian(
        molecular_data=ps_data,
        basis_set=ps_basis,
        integral_engine=engine,
        include_annihilation=True,
    )

    # Build Hamiltonian matrices
    ps_matrices = ps_hamiltonian.build_hamiltonian()

    # Create SCF solver for positronium
    ps_scf = AntinatureSCF(
        hamiltonian=ps_matrices,
        basis_set=ps_basis,
        molecular_data=ps_data,
        max_iterations=50,
        convergence_threshold=1e-5,
        use_diis=True,
        damping_factor=0.7,  # Higher damping for better stability
        level_shifting=0.0,
        diis_start=3,
        diis_dimension=6,
        print_level=1,
    )

    # Run SCF calculation
    ps_results = ps_scf.solve_scf()

    # Print results
    print("\nSCF Results for Positronium:")
    print(f"  Converged: {ps_results['converged']}")
    print(f"  Energy: {ps_results['energy']:.10f} Hartree")
    print(f"  Iterations: {ps_results['iterations']}")
    print(f"  Electron orbital energy: {ps_results['E_electron']}")
    print(f"  Positron orbital energy: {ps_results['E_positron']}")

    # Plot convergence history if available
    if 'energy_history' in ps_results and len(ps_results['energy_history']) > 1:
        try:
            plt.figure(figsize=(10, 6))
            plt.semilogy(
                range(1, len(ps_results['energy_history'])),
                [
                    abs(e - ps_results['energy_history'][-1])
                    for e in ps_results['energy_history'][:-1]
                ],
                'o-',
                label='Energy Error',
            )
            plt.grid(True)
            plt.xlabel('Iteration')
            plt.ylabel('|E - E_final|')
            plt.title('Positronium SCF Convergence')
            plt.legend()
            plt.savefig('positronium_convergence.png')
            print("\nConvergence plot saved to 'positronium_convergence.png'")
        except Exception as e:
            print(f"Couldn't create convergence plot: {e}")

    print("\nSCF tests completed successfully!")


if __name__ == "__main__":
    test_scf_solver()
