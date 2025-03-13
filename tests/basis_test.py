import numpy as np
import matplotlib.pyplot as plt

import unittest
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from antimatter_qchem.core import *
def test_gaussian_basis_function():
    """Test the GaussianBasisFunction class."""
    print("Testing GaussianBasisFunction...")
    
    # Create a simple s-type Gaussian centered at origin
    center = np.array([0.0, 0.0, 0.0])
    exponent = 1.0
    s_function = GaussianBasisFunction(center, exponent, (0, 0, 0))
    
    # Create a p-type Gaussian (px)
    p_function = GaussianBasisFunction(center, exponent, (1, 0, 0))
    
    # Test evaluation at different points
    test_points = [
        np.array([0.0, 0.0, 0.0]),  # Center
        np.array([1.0, 0.0, 0.0]),  # 1 Bohr along x
        np.array([0.0, 1.0, 0.0]),  # 1 Bohr along y
        np.array([0.0, 0.0, 1.0])   # 1 Bohr along z
    ]
    
    print("\nS-type Gaussian evaluation:")
    for point in test_points:
        value = s_function.evaluate(point)
        dist = np.linalg.norm(point - center)
        expected = s_function.normalization * np.exp(-exponent * dist**2)
        print(f"At point {point}: {value:.6f} (Expected: {expected:.6f})")
    
    print("\nP-type (px) Gaussian evaluation:")
    for point in test_points:
        value = p_function.evaluate(point)
        print(f"At point {point}: {value:.6f}")
        
    # Visual check - create a 1D plot along x-axis
    x = np.linspace(-3, 3, 100)
    s_values = [s_function.evaluate(np.array([xi, 0, 0])) for xi in x]
    p_values = [p_function.evaluate(np.array([xi, 0, 0])) for xi in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, s_values, label='s-type')
    plt.plot(x, p_values, label='p-type (px)')
    plt.xlabel('x (Bohr)')
    plt.ylabel('Value')
    plt.title('Gaussian Basis Functions')
    plt.legend()
    plt.grid(True)
    plt.savefig('gaussian_functions.png')
    print("Plot saved as 'gaussian_functions.png'")

def test_basis_set():
    """Test the BasisSet class."""
    print("\nTesting BasisSet...")
    
    # Create a simple basis set for hydrogen
    basis = BasisSet()
    
    # Add STO-3G like functions for hydrogen
    center = np.array([0.0, 0.0, 0.0])
    exponents = [3.42525091, 0.62391373, 0.16885540]
    
    for exp in exponents:
        basis.add_function(GaussianBasisFunction(center, exp, (0, 0, 0)))
    
    print(f"Created basis set with {basis.n_basis} functions")
    
    # Test evaluation of all functions at a point
    test_point = np.array([0.5, 0.0, 0.0])
    values = basis.evaluate_all_at(test_point)
    
    print(f"Basis function values at {test_point}:")
    for i, value in enumerate(values):
        print(f"  Function {i}: {value:.6f}")

def test_positron_basis():
    """Test the PositronBasis class."""
    print("\nTesting PositronBasis...")
    
    # Create a positron basis for a simple molecule (H2)
    positron_basis = PositronBasis()
    
    # Add positron basis functions for H2
    h1_pos = np.array([0.0, 0.0, 0.0])
    h2_pos = np.array([0.0, 0.0, 0.74])  # Typical H2 bond length in angstroms (converted to Bohr)
    
    # Generate basis for each atom
    positron_basis.generate_positron_basis('H', h1_pos, 'standard')
    positron_basis.generate_positron_basis('H', h2_pos, 'standard')
    
    print(f"Created positron basis with {positron_basis.n_basis} functions")
    
    # Add bond functions
    bond_center = (h1_pos + h2_pos) / 2
    bond_length = np.linalg.norm(h2_pos - h1_pos)
    positron_basis.add_bond_functions([bond_center], [bond_length])
    
    print(f"After adding bond functions: {positron_basis.n_basis} functions")
    
    # Add annihilation functions
    positron_basis.add_annihilation_functions(bond_center, bond_length)
    
    print(f"After adding annihilation functions: {positron_basis.n_basis} functions")
    print(f"Number of annihilation functions: {len(positron_basis.annihilation_functions)}")

def test_mixed_matter_basis():
    """Test the MixedMatterBasis class."""
    print("\nTesting MixedMatterBasis...")
    
    # Create a mixed basis for a simple molecule (H2)
    mixed_basis = MixedMatterBasis()
    
    # Define H2 molecule
    h2_molecule = [
        ('H', np.array([0.0, 0.0, 0.0])),
        ('H', np.array([0.0, 0.0, 1.4]))  # Bond length in Bohr
    ]
    
    # Create basis sets for the molecule
    mixed_basis.create_for_molecule(h2_molecule, 'standard', 'extended')
    
    print(f"Created mixed basis with:")
    print(f"  {mixed_basis.n_electron_basis} electron basis functions")
    print(f"  {mixed_basis.n_positron_basis} positron basis functions")
    print(f"  {mixed_basis.n_total_basis} total basis functions")
    
    # Test calculating integrals
    if mixed_basis.n_electron_basis > 0 and mixed_basis.n_positron_basis > 0:
        # Calculate an overlap integral between first electron and first positron basis function
        e_func = mixed_basis.get_basis_function(0)  # First electron function
        p_func = mixed_basis.get_basis_function(mixed_basis.n_electron_basis)  # First positron function
        
        overlap = mixed_basis.integral_engine.overlap_integral(e_func, p_func)
        print(f"\nOverlap between first electron and positron basis functions: {overlap:.6f}")
        
        # Calculate an annihilation integral
        annihilation = mixed_basis.annihilation_integral(0, mixed_basis.n_electron_basis)
        print(f"Annihilation integral between first electron and positron basis functions: {annihilation:.6f}")

if __name__ == "__main__":
    print("=== Basis Functions and Sets Testing ===\n")
    
    test_gaussian_basis_function()
    test_basis_set()
    test_positron_basis()
    test_mixed_matter_basis()
    
    print("\nAll basis tests completed.")