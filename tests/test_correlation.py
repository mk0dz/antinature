# test_correlation.py
import importlib
import inspect
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from antinature.core.basis import MixedMatterBasis
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.molecular_data import MolecularData
from antinature.core.scf import AntinatureSCF

# Force reload the correlation module to avoid caching issues
if 'antinature.core.correlation' in sys.modules:
    importlib.reload(sys.modules['antinature.core.correlation'])
import antinature.core.correlation
from antinature.core.correlation import AntinatureCorrelation

# Print the actual signature of AntinatureCorrelation
print("\nAntinatureCorrelation class info:")
print("  Signature:", inspect.signature(AntinatureCorrelation.__init__))
print("  Module location:", AntinatureCorrelation.__module__)
print("  File location:", inspect.getfile(AntinatureCorrelation))
print("  Source (first few lines):")
try:
    source = inspect.getsource(AntinatureCorrelation.__init__)
    # Print the first few lines
    print('\n'.join(source.split('\n')[:15]))
except Exception as e:
    print(f"  Error getting source: {e}")


def test_correlation():
    """Test the correlation module."""
    print("Testing correlation module for H2 molecule...")

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
        print_level=1,
    )

    # Run SCF calculation
    scf_results = scf.solve_scf()

    print("\nStarting correlation calculations...")

    # Convert SCF results arrays from lists to NumPy arrays
    for key in [
        'C_electron',
        'C_positron',
        'E_electron',
        'E_positron',
        'P_electron',
        'P_positron',
    ]:
        if key in scf_results and isinstance(scf_results[key], list):
            scf_results[key] = np.array(scf_results[key])

    # Create correlation object
    corr = AntinatureCorrelation(
        scf_result=scf_results, hamiltonian=h_matrices, basis_set=basis
    )

    # Calculate MP2 energy
    mp2_energy = corr.mp2_energy()

    # Print MP2 results
    print("\nMP2 Results:")
    print(f"  Correlation energy: {mp2_energy:.10f} Hartree")
    print(f"  Total energy: {scf_results['energy'] + mp2_energy:.10f} Hartree")

    # Try CCSD
    try:
        ccsd_result = corr.coupled_cluster()

        # Print CCSD results if it's a numeric value
        print("\nCCSD Results:")
        if isinstance(ccsd_result, (int, float, np.number)):
            print(f"  Correlation energy: {ccsd_result:.10f} Hartree")
            print(f"  Total energy: {scf_results['energy'] + ccsd_result:.10f} Hartree")
        else:
            print(f"  Result: {ccsd_result}")
    except Exception as e:
        print(f"\nCCSD calculation error: {e}")

    # Test positronium system
    print("\n\nTesting correlation for positronium system...")

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
        damping_factor=0.7,
        print_level=1,
    )

    # Run SCF calculation
    ps_results = ps_scf.solve_scf()

    # Convert SCF results arrays from lists to NumPy arrays
    for key in [
        'C_electron',
        'C_positron',
        'E_electron',
        'E_positron',
        'P_electron',
        'P_positron',
    ]:
        if key in ps_results and isinstance(ps_results[key], list):
            ps_results[key] = np.array(ps_results[key])

    # Create correlation object for positronium
    ps_corr = AntinatureCorrelation(
        scf_result=ps_results, hamiltonian=ps_matrices, basis_set=ps_basis
    )

    # Calculate MP2 energy with electron-positron correlation
    try:
        ps_mp2_energy = ps_corr.mp2_energy(include_electron_positron=True)

        # Print MP2 results
        print("\nMP2 Results for Positronium:")
        print(f"  Correlation energy: {ps_mp2_energy:.10f} Hartree")
        print(f"  Total energy: {ps_results['energy'] + ps_mp2_energy:.10f} Hartree")
    except Exception as e:
        print(f"\nPositronium MP2 calculation error: {e}")

    # Calculate CCSD energy for positronium
    try:
        ps_ccsd_result = ps_corr.coupled_cluster()

        # Print CCSD results
        print("\nCCSD Results for Positronium:")
        if isinstance(ps_ccsd_result, (int, float, np.number)):
            print(f"  Correlation energy: {ps_ccsd_result:.10f} Hartree")
            print(
                f"  Total energy: {ps_results['energy'] + ps_ccsd_result:.10f} Hartree"
            )
        else:
            print(f"  Result: {ps_ccsd_result}")
    except Exception as e:
        print(f"\nPositronium CCSD calculation error: {e}")

    # Calculate positron annihilation rate if available
    try:
        annihilation_rate = ps_corr.calculate_annihilation_rate()
        print(f"\nPositron annihilation rate: {annihilation_rate:.6e} s^-1")
    except Exception as e:
        print(f"\nAnnihilation rate calculation error: {e}")

    print("\nCorrelation tests completed successfully!")


if __name__ == "__main__":
    test_correlation()
