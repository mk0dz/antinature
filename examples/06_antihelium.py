#!/usr/bin/env python
"""
Example 6: Anti-Helium Ion (Anti-He⁺)
=====================================
This example simulates an anti-helium ion, which consists of
an anti-helium nucleus (2 antiprotons, 2 antineutrons) with one positron.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature import MolecularData, AntinatureCalculator
from antinature.core.basis import MixedMatterBasis
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.integral_engine import AntinatureIntegralEngine

def main():
    print("=" * 60)
    print("EXAMPLE 6: ANTI-HELIUM ION (Anti-He⁺)")
    print("=" * 60)
    
    # Create anti-helium ion
    print("\n1. Creating anti-helium ion system:")
    
    antihe_data = MolecularData(
        atoms=[('He', np.array([0.0, 0.0, 0.0]))],  # Anti-He nucleus
        n_electrons=0,
        n_positrons=1,  # One positron (Anti-He⁺)
        charge=-1,  # Ion charge (anti-nucleus has -2, positron has +1)
        name='Anti-He+',
        description='Anti-helium ion with one positron'
    )
    
    print(f"  Name: {antihe_data.name}")
    print(f"  Formula: {antihe_data.get_formula()}")
    print(f"  Nuclear charge: -2 (anti-helium)")
    print(f"  Positrons: {antihe_data.n_positrons}")
    print(f"  Total charge: {antihe_data.charge}")
    
    # Calculate with different methods
    print("\n2. Calculating ground state energy:")
    
    # Method 1: High-level calculator
    calc = AntinatureCalculator(print_level=0)
    result_simple = calc.calculate_custom_system(
        atoms=antihe_data.atoms,
        n_electrons=0,
        n_positrons=1,
        accuracy='high'
    )
    
    print(f"  Simple calculation: {result_simple['energy']:.6f} Hartree")
    
    # Method 2: Direct SCF with custom basis
    print("\n3. Advanced calculation with custom basis:")
    
    # Create custom basis set
    basis = MixedMatterBasis()
    
    # Add positron basis functions centered on nucleus
    from antinature.core.basis import GaussianBasisFunction, PositronBasis
    
    pos_basis = PositronBasis()
    # Add s-type functions with different exponents
    for exp in [0.5, 1.0, 2.0, 4.0, 8.0]:
        pos_basis.add_function(
            GaussianBasisFunction(
                center=np.array([0.0, 0.0, 0.0]),
                exponent=exp,
                angular_momentum=(0, 0, 0)
            )
        )
    
    # Add p-type functions
    for exp in [0.8, 1.6, 3.2]:
        for am in [(1,0,0), (0,1,0), (0,0,1)]:
            pos_basis.add_function(
                GaussianBasisFunction(
                    center=np.array([0.0, 0.0, 0.0]),
                    exponent=exp,
                    angular_momentum=am
                )
            )
    
    basis.positron_basis = pos_basis
    basis.n_positron_basis = pos_basis.n_basis
    basis.n_total_basis = pos_basis.n_basis
    
    print(f"  Basis functions: {basis.n_total_basis}")
    print(f"    S-type: 5")
    print(f"    P-type: 9")
    
    # Build and solve Hamiltonian
    engine = AntinatureIntegralEngine()
    basis.set_integral_engine(engine)
    
    ham = AntinatureHamiltonian(antihe_data, basis, engine)
    ham_matrices = ham.build_hamiltonian()
    
    from antinature.core.scf import AntinatureSCF
    
    scf = AntinatureSCF(
        hamiltonian=ham_matrices,
        basis_set=basis,
        molecular_data=antihe_data,
        max_iterations=50,
        convergence_threshold=1e-8
    )
    
    scf_result = scf.run()
    
    print(f"\nSCF Results:")
    print(f"  Energy: {scf_result['energy']:.8f} Hartree")
    print(f"  Converged: {scf_result['converged']}")
    print(f"  Iterations: {scf_result.get('iterations', 'N/A')}")
    
    # Compare with He⁺ ion
    print("\n4. Comparison with regular He⁺:")
    he_plus_energy = -2.0  # Hartree (approximate)
    
    print(f"  He⁺ energy: {he_plus_energy:.6f} Hartree")
    print(f"  Anti-He⁺ energy: {scf_result['energy']:.6f} Hartree")
    print(f"  CPT symmetry check: {abs(scf_result['energy'] - he_plus_energy):.6e} Hartree")
    
    if abs(scf_result['energy'] - he_plus_energy) < 1e-3:
        print("  ✓ CPT symmetry approximately satisfied")
    else:
        print("  ⚠ Significant CPT violation detected (likely numerical)")
    
    # Analyze orbital structure
    print("\n5. Orbital analysis:")
    if 'orbital_energies' in scf_result:
        orb_e = scf_result['orbital_energies']
        if 'positron' in orb_e and len(orb_e['positron']) > 0:
            print(f"  Positron orbital energies (Hartree):")
            for i, e in enumerate(orb_e['positron'][:5]):
                print(f"    Orbital {i+1}: {e:.6f}")

if __name__ == "__main__":
    main()