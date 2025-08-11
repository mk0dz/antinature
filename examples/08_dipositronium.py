#!/usr/bin/env python
"""
Example 8: Dipositronium and Tripositronium
===========================================
This example explores systems with multiple positrons:
- Dipositronium: e⁺e⁺ (unstable but interesting)
- Tripositronium: e⁺e⁻e⁺ (three-body system)
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature import MolecularData, AntinatureCalculator
from antinature.specialized.annihilation import AnnihilationOperator

def main():
    print("=" * 60)
    print("EXAMPLE 8: MULTI-POSITRON SYSTEMS")
    print("=" * 60)
    
    # 1. Dipositronium (e⁺e⁺) - Unbound system
    print("\n1. Dipositronium (e⁺e⁺) - Repulsive system:")
    
    dipositronium = MolecularData(
        atoms=[],
        n_electrons=0,
        n_positrons=2,
        charge=2,
        name='Dipositronium',
        description='Two positrons - repulsive system'
    )
    
    print(f"  Name: {dipositronium.name}")
    print(f"  Formula: {dipositronium.get_formula()}")
    print(f"  Charge: +{dipositronium.charge}")
    print(f"  Nature: Repulsive (both particles positive)")
    
    # Calculate energy
    calc = AntinatureCalculator(print_level=0)
    
    # This should give positive (repulsive) energy
    try:
        dips_result = calc.calculate_custom_system(
            atoms=[],
            n_electrons=0,
            n_positrons=2,
            accuracy='low'
        )
        print(f"  Energy: {dips_result['energy']:.6f} Hartree")
        if dips_result['energy'] > 0:
            print("  ✓ Correctly shows repulsive interaction")
    except Exception as e:
        print(f"  Calculation failed (expected for unbound system): {e}")
    
    # 2. Tripositronium (e⁺e⁻e⁺)
    print("\n2. Tripositronium (e⁺e⁻e⁺) - Three-body system:")
    
    tripositronium = MolecularData(
        atoms=[],
        n_electrons=1,
        n_positrons=2,
        charge=1,
        name='Tripositronium',
        description='Positron-electron-positron system'
    )
    
    print(f"  Name: {tripositronium.name}")
    print(f"  Formula: {tripositronium.get_formula()}")
    print(f"  Charge: +{tripositronium.charge}")
    
    trips_result = calc.calculate_custom_system(
        atoms=[],
        n_electrons=1,
        n_positrons=2,
        accuracy='medium'
    )
    
    print(f"\nResults:")
    print(f"  Total energy: {trips_result['energy']:.6f} Hartree")
    
    # Compare with Ps + e⁺
    ps_energy = -0.25  # Hartree
    print(f"\n  Reference energies:")
    print(f"    Ps + e⁺ (separated): {ps_energy:.6f} Hartree")
    print(f"    Tripositronium: {trips_result['energy']:.6f} Hartree")
    
    binding = trips_result['energy'] - ps_energy
    if binding < 0:
        print(f"    Binding energy: {binding:.6f} Hartree = {binding*27.2114:.3f} eV")
        print(f"    → System is bound")
    else:
        print(f"    → System is unbound or metastable")
    
    # 3. e⁻e⁺e⁻ system (electron-positron-electron)
    print("\n3. Electron-positron-electron (e⁻e⁺e⁻) system:")
    
    epe_system = MolecularData(
        atoms=[],
        n_electrons=2,
        n_positrons=1,
        charge=-1,
        name='EPE',
        description='Electron-positron-electron system'
    )
    
    print(f"  Name: {epe_system.name}")
    print(f"  Formula: {epe_system.get_formula()}")
    print(f"  Charge: {epe_system.charge}")
    
    epe_result = calc.calculate_custom_system(
        atoms=[],
        n_electrons=2,
        n_positrons=1,
        accuracy='medium'
    )
    
    print(f"\nResults:")
    print(f"  Total energy: {epe_result['energy']:.6f} Hartree")
    
    # Compare with Ps + e⁻
    print(f"\n  Reference energies:")
    print(f"    Ps + e⁻ (separated): {ps_energy:.6f} Hartree")
    print(f"    EPE system: {epe_result['energy']:.6f} Hartree")
    
    # 4. Annihilation dynamics
    print("\n4. Annihilation dynamics in multi-positron systems:")
    
    ann_op = AnnihilationOperator()
    
    # For tripositronium
    print("\n  Tripositronium (e⁺e⁻e⁺):")
    print("    - Central e⁻ can annihilate with either e⁺")
    print("    - After annihilation: free positron + photons")
    
    # Estimate annihilation rate (simplified)
    overlap = 0.3  # Estimated e⁺e⁻ overlap
    rate_trips = ann_op.calculate_annihilation_rate(
        electron_density=1.0,
        positron_density=2.0,  # Two positrons
        overlap=overlap
    )
    
    lifetime_trips = 1e9 / rate_trips if rate_trips > 0 else float('inf')
    print(f"    - Estimated lifetime: {lifetime_trips:.2f} ns")
    
    # 5. Stability analysis
    print("\n5. Stability analysis:")
    
    systems = [
        ("Positronium (Ps)", -0.25, "Stable bound state"),
        ("Dipositronium (e⁺e⁺)", float('inf'), "Unbound - repulsive"),
        ("Tripositronium (e⁺e⁻e⁺)", trips_result['energy'], "Metastable or unbound"),
        ("EPE (e⁻e⁺e⁻)", epe_result['energy'], "Metastable or unbound"),
        ("Ps₂", -0.516, "Weakly bound molecule")
    ]
    
    print("\n  System                    Energy (Ha)    Status")
    print("  " + "-" * 55)
    for name, energy, status in systems:
        if energy == float('inf'):
            energy_str = "    ∞    "
        else:
            energy_str = f"{energy:9.6f}"
        print(f"  {name:24s} {energy_str}  {status}")

if __name__ == "__main__":
    main()