#!/usr/bin/env python
"""
Example 4: Positronium Hydride (PsH)
====================================
This example shows how to simulate positronium hydride,
an exotic molecule consisting of a hydrogen atom bound to a positronium atom.
PsH = H + Ps = (proton + electron) + (electron + positron)
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature import MolecularData, AntinatureCalculator

def main():
    print("=" * 60)
    print("EXAMPLE 4: POSITRONIUM HYDRIDE (PsH)")
    print("=" * 60)
    
    # Create PsH molecule
    print("\n1. Creating PsH molecular system:")
    
    psh_data = MolecularData(
        atoms=[('H', np.array([0.0, 0.0, 0.0]))],  # Hydrogen nucleus
        n_electrons=2,  # Two electrons (one from H, one from Ps)
        n_positrons=1,  # One positron from Ps
        charge=0,  # Neutral molecule
        name='PsH',
        description='Positronium hydride - hydrogen bound to positronium'
    )
    
    print(f"  Name: {psh_data.name}")
    print(f"  Description: {psh_data.description}")
    print(f"  Formula: {psh_data.get_formula()}")
    print(f"  Nuclear repulsion: {psh_data.get_nuclear_repulsion_energy():.6f} Hartree")
    
    # Calculate energy at different accuracies
    print("\n2. Calculating PsH energy at different accuracy levels:")
    calc = AntinatureCalculator(print_level=0)
    
    accuracies = ['low', 'medium', 'high']
    results = {}
    
    for accuracy in accuracies:
        result = calc.calculate_custom_system(
            atoms=psh_data.atoms,
            n_electrons=psh_data.n_electrons,
            n_positrons=psh_data.n_positrons,
            accuracy=accuracy
        )
        results[accuracy] = result
        print(f"  {accuracy:6s}: E = {result['energy']:.6f} Hartree")
    
    # Analyze convergence
    print("\n3. Convergence analysis:")
    if len(results) > 1:
        energies = [results[acc]['energy'] for acc in accuracies]
        convergence = abs(energies[-1] - energies[-2])
        print(f"  Energy change (medium→high): {convergence:.6e} Hartree")
        
        if convergence < 1e-4:
            print("  ✓ Good convergence achieved")
        else:
            print("  ⚠ May need higher accuracy for converged results")
    
    # Dissociation analysis
    print("\n4. Dissociation energy analysis:")
    psh_energy = results['high']['energy']
    h_energy = -0.5  # Hartree (theoretical)
    ps_energy = -0.25  # Hartree (theoretical)
    
    dissociation_energy = psh_energy - (h_energy + ps_energy)
    
    print(f"  PsH total energy: {psh_energy:.6f} Hartree")
    print(f"  H atom energy: {h_energy:.6f} Hartree")
    print(f"  Ps atom energy: {ps_energy:.6f} Hartree")
    print(f"  H + Ps energy: {h_energy + ps_energy:.6f} Hartree")
    print(f"  Dissociation energy: {dissociation_energy:.6f} Hartree")
    print(f"                      = {dissociation_energy * 27.2114:.3f} eV")
    
    if dissociation_energy < 0:
        print(f"  → PsH is bound by {abs(dissociation_energy * 27.2114):.3f} eV")
    else:
        print(f"  → PsH is unbound (metastable)")
    
    # Annihilation lifetime estimate
    print("\n5. Annihilation lifetime estimate:")
    from antinature.specialized.annihilation import AnnihilationOperator
    
    ann_op = AnnihilationOperator()
    # Estimate based on overlap (simplified)
    estimated_rate = ann_op.calculate_annihilation_rate(
        electron_density=1.0,
        positron_density=0.5,
        overlap=0.3  # Estimated overlap
    )
    
    lifetime_ns = 1e9 / estimated_rate if estimated_rate > 0 else float('inf')
    print(f"  Estimated annihilation rate: {estimated_rate:.2e} s⁻¹")
    print(f"  Estimated lifetime: {lifetime_ns:.2f} ns")

if __name__ == "__main__":
    main()