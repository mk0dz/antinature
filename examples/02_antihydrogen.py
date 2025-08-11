#!/usr/bin/env python
"""
Example 2: Anti-Hydrogen Atom Simulation
========================================
This example shows how to simulate an anti-hydrogen atom, which consists
of an antiproton and a positron. Anti-hydrogen is the antimatter counterpart
of the simplest atom in the universe.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature import MolecularData, AntinatureCalculator

def main():
    print("=" * 60)
    print("EXAMPLE 2: ANTI-HYDROGEN ATOM SIMULATION")
    print("=" * 60)
    
    # Method 1: Using predefined anti-hydrogen
    print("\n1. Using predefined anti-hydrogen system:")
    ah_data = MolecularData.anti_hydrogen()
    print(f"  Name: {ah_data.name}")
    print(f"  Formula: {ah_data.get_formula()}")
    print(f"  Positrons: {ah_data.n_positrons}")
    print(f"  Electrons: {ah_data.n_electrons}")
    
    # Method 2: Calculate anti-hydrogen energy
    print("\n2. Calculating anti-hydrogen ground state:")
    from antinature import calculate_antihydrogen
    
    ah_result = calculate_antihydrogen(accuracy='medium')
    print(f"\nResults:")
    print(f"  Energy: {ah_result['energy']:.6f} Hartree")
    print(f"  Converged: {ah_result['converged']}")
    
    # Method 3: Custom anti-hydrogen with different basis
    print("\n3. Custom anti-hydrogen calculation:")
    calc = AntinatureCalculator(print_level=0)
    
    # Create custom anti-hydrogen
    custom_ah = MolecularData(
        atoms=[('H', np.array([0.0, 0.0, 0.0]))],  # Antiproton at origin
        n_electrons=0,
        n_positrons=1,
        charge=-1,  # Antiproton has -1 charge
        name='Custom Anti-H'
    )
    
    result = calc.calculate_custom_system(
        atoms=custom_ah.atoms,
        n_electrons=0,
        n_positrons=1,
        accuracy='high'
    )
    
    print(f"\nCustom Anti-H Results:")
    print(f"  Energy: {result['energy']:.6f} Hartree")
    print(f"  Time: {result.get('computation_time', 0):.2f} seconds")
    
    # Compare with regular hydrogen
    print("\n4. Comparison with regular hydrogen:")
    print(f"  Anti-hydrogen energy: {ah_result['energy']:.6f} Hartree")
    print(f"  Regular hydrogen (theoretical): -0.5 Hartree")
    print(f"  Difference: {abs(ah_result['energy'] + 0.5):.6e} Hartree")

if __name__ == "__main__":
    main()