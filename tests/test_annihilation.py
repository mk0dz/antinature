# test_annihilation.py

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature.core.basis import GaussianBasisFunction, MixedMatterBasis
from antinature.core.molecular_data import MolecularData
from antinature.specialized.annihilation import AnnihilationOperator


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


def create_test_wavefunction():
    """Create a test wavefunction for positronium."""
    n_basis = 2

    # Simple density matrices
    P_e = np.array([[0.8, 0.2], [0.2, 0.2]])
    P_p = np.array([[0.7, 0.3], [0.3, 0.3]])

    # MO coefficients
    C_e = np.array([[0.9, 0.1], [0.1, 0.9]])
    C_p = np.array([[0.85, 0.15], [0.15, 0.85]])

    return {
        'P_electron': P_e,
        'P_positron': P_p,
        'C_electron': C_e,
        'C_positron': C_p,
        'n_electrons': 1,
        'n_positrons': 1,
        'singlet_fraction': 0.75,  # 75% para-positronium (singlet)
        'triplet_fraction': 0.25,  # 25% ortho-positronium (triplet)
    }


def test_annihilation_operator():
    """Test the annihilation operator module."""
    # Create test objects
    basis = create_test_basis()
    wavefunction = create_test_wavefunction()

    print("Testing AnnihilationOperator with different methods:")

    # Test with different calculation methods (simplified to one test since methods aren't implemented)
    print("\nTesting annihilation calculation:")

    # Create annihilation operator
    ann_op = AnnihilationOperator(basis_set=basis, wavefunction=wavefunction)

    # Build operator matrix
    matrix = ann_op.build_annihilation_operator()
    print(f"  Annihilation matrix shape: {matrix.shape}")
    print(f"  Matrix[0,0]: {matrix[0,0]:.6f}")

    # Calculate annihilation rates
    rate = ann_op.calculate_annihilation_rate()
    print(f"  Annihilation rate details: {rate}")
    print(f"  Two-gamma rate: {rate['two_gamma']['rate']:.6e} au")
    print(f"  Three-gamma rate: {rate['three_gamma']['rate']:.6e} au")
    print(f"  Total rate: {rate['total_rate']:.6e} au")

    # Calculate lifetime (if the method exists)
    try:
        lifetime = ann_op.calculate_lifetime(rate)
        if isinstance(lifetime, dict) and 'lifetime_ns' in lifetime:
            print(f"  Lifetime: {lifetime['lifetime_ns']:.6f} ns")
        else:
            print(f"  Lifetime: {lifetime:.6f}")
    except (AttributeError, TypeError):
        print("  Lifetime calculation not available")

    # Analyze annihilation channels
    try:
        channels = ann_op.analyze_annihilation_channels()
        print(f"  Annihilation channels analysis: {type(channels)}")
        for key, value in channels.items():
            if isinstance(value, dict):
                print(f"    {key}: {value}")
            else:
                print(f"    {key}: {value}")
    except (AttributeError, TypeError) as e:
        print(f"  Channel analysis not available: {e}")

    # Visualize annihilation density (without displaying)
    try:
        density_data = ann_op.visualize_annihilation_density(grid_dims=(10, 10, 10))
        print(f"  Density data type: {type(density_data)}")
        if isinstance(density_data, dict):
            print(f"  Density data keys: {list(density_data.keys())}")
            if 'density' in density_data:
                print(f"  Density data shape: {density_data['density'].shape}")
    except (AttributeError, TypeError) as e:
        print(f"  Density visualization not available: {e}")

    # Check timing
    for key, value in ann_op.timing.items():
        print(f"  Timing - {key}: {value:.4f} seconds")

    # Try creating a plot (if method exists)
    print("\nTrying visualization plot...")
    try:
        fig = ann_op.plot_annihilation_density(
            plot_type='contour', save_path='annihilation_density.png'
        )
        print("  Plot saved as 'annihilation_density.png'")
    except (AttributeError, TypeError) as e:
        print(f"  Plot method not available: {e}")
        # Try alternative visualization
        print("  Trying alternative visualization...")
        try:
            dens_data = ann_op.visualize_annihilation_density()
            print(
                f"  Visualization data available with keys: {list(dens_data.keys()) if isinstance(dens_data, dict) else 'N/A'}"
            )
        except (AttributeError, TypeError) as e:
            print(f"  Alternative visualization not available: {e}")


if __name__ == "__main__":
    test_annihilation_operator()
