#!/usr/bin/env python
"""
Example 5: Muonium and Antimuonium
==================================
This example demonstrates simulation of muonium (μ⁺e⁻) and antimuonium (μ⁻e⁺).
Muonium is an exotic atom similar to hydrogen but with a positive muon
instead of a proton.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature import MolecularData, AntinatureCalculator

def main():
    print("=" * 60)
    print("EXAMPLE 5: MUONIUM AND ANTIMUONIUM")
    print("=" * 60)
    
    # Muon mass is about 207 times electron mass
    # This affects the reduced mass and thus the energy levels
    
    # Create muonium (μ⁺e⁻)
    print("\n1. Creating muonium system (μ⁺e⁻):")
    
    muonium_data = MolecularData(
        atoms=[('Mu', np.array([0.0, 0.0, 0.0]))],  # Positive muon as nucleus
        n_electrons=1,
        n_positrons=0,
        charge=0,
        name='Muonium',
        description='Exotic atom with positive muon and electron'
    )
    
    print(f"  Name: {muonium_data.name}")
    print(f"  Formula: {muonium_data.get_formula()}")
    
    # Create antimuonium (μ⁻e⁺)
    print("\n2. Creating antimuonium system (μ⁻e⁺):")
    
    antimuonium_data = MolecularData(
        atoms=[('Mu-', np.array([0.0, 0.0, 0.0]))],  # Negative muon
        n_electrons=0,
        n_positrons=1,
        charge=0,
        name='Antimuonium',
        description='Exotic atom with negative muon and positron'
    )
    
    print(f"  Name: {antimuonium_data.name}")
    print(f"  Formula: {antimuonium_data.get_formula()}")
    
    # Calculate energies
    print("\n3. Calculating ground state energies:")
    calc = AntinatureCalculator(print_level=0)
    
    # Muonium calculation
    mu_result = calc.calculate_custom_system(
        atoms=muonium_data.atoms,
        n_electrons=muonium_data.n_electrons,
        n_positrons=muonium_data.n_positrons,
        accuracy='medium'
    )
    
    # Antimuonium calculation
    antimu_result = calc.calculate_custom_system(
        atoms=antimuonium_data.atoms,
        n_electrons=antimuonium_data.n_electrons,
        n_positrons=antimuonium_data.n_positrons,
        accuracy='medium'
    )
    
    print(f"\nResults:")
    print(f"  Muonium energy: {mu_result['energy']:.6f} Hartree")
    print(f"  Antimuonium energy: {antimu_result['energy']:.6f} Hartree")
    
    # Compare with hydrogen
    print("\n4. Comparison with hydrogen:")
    h_energy = -0.5  # Hartree
    
    # Reduced mass correction factor
    # μ_reduced(Mu) / μ_reduced(H) ≈ 0.995
    reduced_mass_factor = 0.995
    expected_mu_energy = h_energy * reduced_mass_factor
    
    print(f"  Hydrogen energy: {h_energy:.6f} Hartree")
    print(f"  Expected muonium (with reduced mass): {expected_mu_energy:.6f} Hartree")
    print(f"  Calculated muonium: {mu_result['energy']:.6f} Hartree")
    print(f"  Difference: {abs(mu_result['energy'] - expected_mu_energy):.6e} Hartree")
    
    # Hyperfine structure (simplified estimate)
    print("\n5. Hyperfine structure estimate:")
    
    # Muon magnetic moment is about 3.18 times proton magnetic moment
    muon_magnetic_factor = 3.18
    h_hyperfine = 1420.4  # MHz for hydrogen
    mu_hyperfine = h_hyperfine * muon_magnetic_factor * reduced_mass_factor
    
    print(f"  Hydrogen hyperfine splitting: {h_hyperfine:.1f} MHz")
    print(f"  Estimated muonium hyperfine: {mu_hyperfine:.1f} MHz")
    print(f"  (Experimental value: ~4463 MHz)")
    
    # Muon lifetime consideration
    print("\n6. Muon decay consideration:")
    muon_lifetime = 2.2e-6  # seconds
    
    print(f"  Muon lifetime: {muon_lifetime*1e6:.1f} μs")
    print(f"  This limits experimental observation time")
    print(f"  Number of Bohr periods before decay: ~{muon_lifetime/2.4e-17:.0e}")

if __name__ == "__main__":
    main()