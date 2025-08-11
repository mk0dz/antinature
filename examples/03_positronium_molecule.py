#!/usr/bin/env python
"""
Example 3: Positronium Molecule (Ps₂)
=====================================
This example demonstrates how to simulate a positronium molecule,
which consists of two positronium atoms bound together.
This is an exotic molecule made entirely of matter-antimatter pairs.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature import MolecularData, AntinatureCalculator

def main():
    print("=" * 60)
    print("EXAMPLE 3: POSITRONIUM MOLECULE (Ps₂)")
    print("=" * 60)
    
    # Create positronium molecule
    print("\n1. Creating positronium molecule system:")
    
    # Ps₂ has 2 electrons and 2 positrons
    ps2_data = MolecularData(
        atoms=[],  # No nuclei in positronium molecule
        n_electrons=2,
        n_positrons=2,
        charge=0,
        name='Ps2',
        description='Positronium molecule - bound state of two positronium atoms'
    )
    
    print(f"  Name: {ps2_data.name}")
    print(f"  Description: {ps2_data.description}")
    print(f"  Formula: {ps2_data.get_formula()}")
    print(f"  Electrons: {ps2_data.n_electrons}")
    print(f"  Positrons: {ps2_data.n_positrons}")
    
    # Calculate energy
    print("\n2. Calculating Ps₂ ground state energy:")
    calc = AntinatureCalculator(print_level=1)
    
    result = calc.calculate_custom_system(
        atoms=[],
        n_electrons=2,
        n_positrons=2,
        accuracy='medium'
    )
    
    print(f"\nResults:")
    print(f"  Total Energy: {result['energy']:.6f} Hartree")
    print(f"  Binding Energy: {result['energy'] + 0.5:.6f} Hartree")  # Relative to 2 Ps atoms
    
    # Analyze electron-positron interactions
    print("\n3. Analyzing particle interactions:")
    if 'orbital_analysis' in result:
        orb = result['orbital_analysis']
        print(f"  Electron HOMO: {orb.get('electron_homo', 'N/A')} Hartree")
        print(f"  Positron HOMO: {orb.get('positron_homo', 'N/A')} Hartree")
    
    # Compare with isolated positronium atoms
    print("\n4. Comparison with isolated Ps atoms:")
    single_ps_energy = -0.25  # Hartree
    two_ps_energy = 2 * single_ps_energy
    binding_energy = result['energy'] - two_ps_energy
    
    print(f"  Single Ps energy: {single_ps_energy:.6f} Hartree")
    print(f"  Two isolated Ps: {two_ps_energy:.6f} Hartree")
    print(f"  Ps₂ energy: {result['energy']:.6f} Hartree")
    print(f"  Ps₂ binding energy: {binding_energy:.6f} Hartree")
    
    if binding_energy < 0:
        print(f"  → Ps₂ is bound by {abs(binding_energy)*27.2114:.3f} eV")
    else:
        print(f"  → Ps₂ is unbound")

if __name__ == "__main__":
    main()