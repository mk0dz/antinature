import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from antimatter_qchem.validation.validator import AntimatterValidator
from antimatter_qchem.validation.test_suite import TestSuite

import numpy as np
import matplotlib.pyplot as plt
from antimatter_qchem.core import GaussianBasisFunction, AntimatterIntegralEngine

def test_overlap_integrals():
    """Test overlap integral calculations."""
    print("Testing overlap integrals...")
    
    # Create integral engine
    integral_engine = AntimatterIntegralEngine(use_grid=False)
    
    # Create two Gaussian basis functions
    center1 = np.array([0.0, 0.0, 0.0])
    center2 = np.array([1.0, 0.0, 0.0])
    exponent = 1.0
    
    func1 = GaussianBasisFunction(center1, exponent, (0, 0, 0))
    func2 = GaussianBasisFunction(center2, exponent, (0, 0, 0))
    
    # Calculate overlap
    overlap = integral_engine.overlap_integral(func1, func2)
    
    # Analytical result for s-type Gaussians: (π/(α+β))^(3/2) * exp(-αβ/(α+β) * |R_A - R_B|^2)
    alpha = func1.exponent
    beta = func2.exponent
    R_diff = np.linalg.norm(func1.center - func2.center)
    expected = (np.pi / (alpha + beta))**1.5 * np.exp(-alpha * beta / (alpha + beta) * R_diff**2)
    expected *= func1.normalization * func2.normalization / ((2*alpha/np.pi)**0.75 * (2*beta/np.pi)**0.75)
    
    print(f"Overlap between two s-type Gaussians at distance 1.0 Bohr:")
    print(f"  Calculated: {overlap:.8f}")
    print(f"  Expected: {expected:.8f}")
    
    # Test varying distances
    distances = np.linspace(0.0, 3.0, 10)
    overlaps = []
    expecteds = []
    
    for dist in distances:
        center2 = np.array([dist, 0.0, 0.0])
        func2 = GaussianBasisFunction(center2, exponent, (0, 0, 0))
        
        overlap = integral_engine.overlap_integral(func1, func2)
        overlaps.append(overlap)
        
        R_diff = dist
        expected = (np.pi / (alpha + beta))**1.5 * np.exp(-alpha * beta / (alpha + beta) * R_diff**2)
        expected *= func1.normalization * func2.normalization / ((2*alpha/np.pi)**0.75 * (2*beta/np.pi)**0.75)
        expecteds.append(expected)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(distances, overlaps, 'o-', label='Calculated')
    plt.plot(distances, expecteds, 's--', label='Analytical')
    plt.xlabel('Distance (Bohr)')
    plt.ylabel('Overlap Integral')
    plt.title('Overlap Integral vs. Distance')
    plt.legend()
    plt.grid(True)
    plt.savefig('overlap_integrals.png')
    print("Plot saved as 'overlap_integrals.png'")

def test_kinetic_integrals():
    """Test kinetic energy integral calculations."""
    print("\nTesting kinetic energy integrals...")
    
    # Create integral engine
    integral_engine = AntimatterIntegralEngine(use_grid=False)
    
    # Create two Gaussian basis functions
    center = np.array([0.0, 0.0, 0.0])
    exponent1 = 1.0
    exponent2 = 0.5
    
    func1 = GaussianBasisFunction(center, exponent1, (0, 0, 0))
    func2 = GaussianBasisFunction(center, exponent2, (0, 0, 0))
    
    # Calculate kinetic energy integral
    kinetic = integral_engine.kinetic_integral(func1, func2)
    
    # Analytical result for s-type Gaussians at same center: 
    # K_ab = α*β/(α+β) * (3 - 2αβ/(α+β)|R_A-R_B|²) * S_ab
    alpha = func1.exponent
    beta = func2.exponent
    overlap = integral_engine.overlap_integral(func1, func2)
    expected = alpha * beta / (alpha + beta) * 3 * overlap
    
    print(f"Kinetic energy integral between two s-type Gaussians of different exponents:")
    print(f"  Calculated: {kinetic:.8f}")
    print(f"  Expected: {expected:.8f}")
    
    # Test with different angular momenta
    print("\nKinetic energy integrals with different angular momenta:")
    
    # p-type orbital (px)
    p_func = GaussianBasisFunction(center, exponent1, (1, 0, 0))
    
    # Calculate kinetic energy between s and p orbitals
    kinetic_sp = integral_engine.kinetic_integral(func1, p_func)
    print(f"  s-px: {kinetic_sp:.8f} (should be 0 by symmetry)")
    
    # Calculate kinetic energy between p and p orbitals
    kinetic_pp = integral_engine.kinetic_integral(p_func, p_func)
    print(f"  px-px: {kinetic_pp:.8f}")

def test_nuclear_attraction_integrals():
    """Test nuclear attraction integral calculations."""
    print("\nTesting nuclear attraction integrals...")
    
    # Create integral engine
    integral_engine = AntimatterIntegralEngine(use_grid=False)
    
    # Create a Gaussian basis function
    center = np.array([0.0, 0.0, 0.0])
    exponent = 1.0
    
    func = GaussianBasisFunction(center, exponent, (0, 0, 0))
    
    # Test nuclear attraction at various distances
    nuclear_pos = np.array([0.0, 0.0, 0.0])  # Nucleus at origin
    attraction = integral_engine.nuclear_attraction_integral(func, func, nuclear_pos)
    
    print(f"Nuclear attraction for s-type orbital with nucleus at same center:")
    print(f"  Value: {attraction:.8f}")
    
    # Test attraction with nucleus at various distances
    distances = np.linspace(0.01, 3.0, 10)  # Avoid r=0 (singularity)
    attractions = []
    
    for dist in distances:
        nuclear_pos = np.array([dist, 0.0, 0.0])
        attraction = integral_engine.nuclear_attraction_integral(func, func, nuclear_pos)
        attractions.append(attraction)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(distances, attractions, 'o-')
    plt.xlabel('Nuclear Distance (Bohr)')
    plt.ylabel('Nuclear Attraction Integral')
    plt.title('Nuclear Attraction Integral vs. Distance')
    plt.grid(True)
    plt.savefig('nuclear_attraction.png')
    print("Plot saved as 'nuclear_attraction.png'")

def test_electron_repulsion_integrals():
    """Test electron repulsion integral calculations."""
    print("\nTesting electron repulsion integrals...")
    
    # Create integral engine
    integral_engine = AntimatterIntegralEngine(use_grid=False)
    
    # Create Gaussian basis functions
    center1 = np.array([0.0, 0.0, 0.0])
    center2 = np.array([1.0, 0.0, 0.0])
    exponent = 1.0
    
    func1 = GaussianBasisFunction(center1, exponent, (0, 0, 0))
    func2 = GaussianBasisFunction(center2, exponent, (0, 0, 0))
    
    # Calculate various ERIs
    # (11|11): Coulomb self-interaction
    eri_1111 = integral_engine.electron_repulsion_integral(func1, func1, func1, func1)
    # (12|12): Coulomb interaction
    eri_1212 = integral_engine.electron_repulsion_integral(func1, func2, func1, func2)
    # (12|21): Exchange interaction
    eri_1221 = integral_engine.electron_repulsion_integral(func1, func2, func2, func1)
    
    print(f"Electron repulsion integrals:")
    print(f"  (11|11): {eri_1111:.8f}")
    print(f"  (12|12): {eri_1212:.8f}")
    print(f"  (12|21): {eri_1221:.8f}")
    
    # Test symmetry properties
    eri_2121 = integral_engine.electron_repulsion_integral(func2, func1, func2, func1)
    print("\nTesting ERI symmetry:")
    print(f"  (12|12) = {eri_1212:.8f}")
    print(f"  (21|21) = {eri_2121:.8f}")
    print(f"  Difference: {abs(eri_1212 - eri_2121):.8f} (should be close to 0)")

def test_annihilation_integrals():
    """Test annihilation integral calculations."""
    print("\nTesting annihilation integrals...")
    
    # Create integral engine
    integral_engine = AntimatterIntegralEngine(use_grid=False)
    
    # Create electron and positron basis functions
    center1 = np.array([0.0, 0.0, 0.0])
    center2 = np.array([0.0, 0.0, 0.0])  # Same center initially
    exponent_e = 1.0
    exponent_p = 0.5  # Positron orbitals are more diffuse
    
    electron_func = GaussianBasisFunction(center1, exponent_e, (0, 0, 0))
    positron_func = GaussianBasisFunction(center2, exponent_p, (0, 0, 0))
    
    # Calculate annihilation integral
    annihilation = integral_engine.annihilation_integral(electron_func, positron_func)
    
    # For same-center s-type Gaussians, this is related to the overlap
    overlap = integral_engine.overlap_integral(electron_func, positron_func)
    
    print(f"Annihilation integral between electron and positron at same center:")
    print(f"  Annihilation: {annihilation:.8f}")
    print(f"  Overlap: {overlap:.8f}")
    
    # Test annihilation integral vs. distance
    distances = np.linspace(0.0, 3.0, 10)
    annihilations = []
    
    for dist in distances:
        center2 = np.array([dist, 0.0, 0.0])
        positron_func = GaussianBasisFunction(center2, exponent_p, (0, 0, 0))
        
        annihilation = integral_engine.annihilation_integral(electron_func, positron_func)
        annihilations.append(annihilation)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(distances, annihilations, 'o-')
    plt.xlabel('Distance (Bohr)')
    plt.ylabel('Annihilation Integral')
    plt.title('Annihilation Integral vs. Distance')
    plt.grid(True)
    plt.savefig('annihilation_integrals.png')
    print("Plot saved as 'annihilation_integrals.png'")

def test_relativistic_integrals():
    """Test relativistic correction integral calculations."""
    print("\nTesting relativistic correction integrals...")
    
    # Create integral engine
    integral_engine = AntimatterIntegralEngine(use_grid=False)
    
    # Create Gaussian basis functions
    center = np.array([0.0, 0.0, 0.0])
    nuclear_pos = np.array([0.0, 0.0, 0.0])
    exponents = [0.5, 1.0, 2.0]
    
    print("Relativistic correction integrals for s-type orbitals of different exponents:")
    
    for exp in exponents:
        func = GaussianBasisFunction(center, exp, (0, 0, 0))
        
        # Calculate mass-velocity term
        mass_velocity = integral_engine.mass_velocity_integral(func, func)
        
        # Calculate Darwin term
        darwin = integral_engine.darwin_integral(func, func, nuclear_pos)
        
        print(f"\nExponent: {exp:.1f}")
        print(f"  Mass-velocity term: {mass_velocity:.8f}")
        print(f"  Darwin term: {darwin:.8f}")
        print(f"  Ratio (Darwin/Mass-velocity): {darwin/mass_velocity if mass_velocity != 0 else 'N/A':.4f}")
    
    # Test relativistic terms vs. nuclear charge
    charges = [1, 2, 3, 4]  # H, He, Li, Be
    darwins = []
    
    func = GaussianBasisFunction(center, 1.0, (0, 0, 0))
    
    for Z in charges:
        darwin = Z * integral_engine.darwin_integral(func, func, nuclear_pos)
        darwins.append(darwin)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(charges, darwins, 'o-')
    plt.xlabel('Nuclear Charge (Z)')
    plt.ylabel('Darwin Term (x Z)')
    plt.title('Darwin Term vs. Nuclear Charge')
    plt.grid(True)
    plt.savefig('darwin_term.png')
    print("Plot saved as 'darwin_term.png'")

if __name__ == "__main__":
    print("=== Integral Engine Testing ===\n")
    
    test_overlap_integrals()
    test_kinetic_integrals()
    test_nuclear_attraction_integrals()
    test_electron_repulsion_integrals()
    test_annihilation_integrals()
    test_relativistic_integrals()
    
    print("\nAll integral engine tests completed.")