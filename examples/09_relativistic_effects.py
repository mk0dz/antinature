#!/usr/bin/env python
"""
Example 9: Relativistic Effects in Antimatter
=============================================
This example demonstrates relativistic corrections in antimatter systems,
including spin-orbit coupling, Darwin terms, and mass-velocity corrections.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature import MolecularData, AntinatureCalculator
from antinature.specialized.relativistic import RelativisticCorrection

def main():
    print("=" * 60)
    print("EXAMPLE 9: RELATIVISTIC EFFECTS IN ANTIMATTER")
    print("=" * 60)
    
    # Fine structure constant
    alpha = 1/137.035999  # Fine structure constant
    c = 137.035999  # Speed of light in atomic units
    
    print(f"\nPhysical constants:")
    print(f"  Fine structure constant (α): {alpha:.6f}")
    print(f"  Speed of light (c): {c:.1f} a.u.")
    
    # Create relativistic correction calculator
    rel_corr = RelativisticCorrection(fine_structure_constant=alpha)
    
    # 1. Positronium with relativistic corrections
    print("\n1. Positronium with relativistic corrections:")
    
    ps_data = MolecularData.positronium()
    calc = AntinatureCalculator(print_level=0)
    
    # Non-relativistic calculation
    ps_result = calc.calculate_positronium(accuracy='high')
    e_nonrel = ps_result['energy']
    
    print(f"  Non-relativistic energy: {e_nonrel:.8f} Hartree")
    
    # Relativistic corrections for positronium
    print("\n  Relativistic corrections:")
    
    # Mass-velocity correction: -(α²/8) * E₀
    mass_velocity = -(alpha**2 / 8) * abs(e_nonrel)
    print(f"    Mass-velocity: {mass_velocity:.8f} Hartree")
    
    # Darwin term (s-orbitals only): (α²/2) * E₀
    darwin = (alpha**2 / 2) * abs(e_nonrel)
    print(f"    Darwin term: {darwin:.8f} Hartree")
    
    # Spin-orbit coupling (for p-orbitals and higher)
    # For s-state positronium, this is zero
    spin_orbit = 0.0
    print(f"    Spin-orbit: {spin_orbit:.8f} Hartree")
    
    # Total relativistic correction
    total_rel = mass_velocity + darwin + spin_orbit
    print(f"    Total correction: {total_rel:.8f} Hartree")
    
    # Corrected energy
    e_rel = e_nonrel + total_rel
    print(f"\n  Relativistic energy: {e_rel:.8f} Hartree")
    print(f"  Correction: {100*total_rel/abs(e_nonrel):.4f}%")
    
    # 2. Anti-hydrogen with relativistic effects
    print("\n2. Anti-hydrogen relativistic corrections:")
    
    ah_data = MolecularData.anti_hydrogen()
    ah_result = calc.calculate_custom_system(
        atoms=ah_data.atoms,
        n_electrons=ah_data.n_electrons,
        n_positrons=ah_data.n_positrons,
        accuracy='high'
    )
    
    e_ah_nonrel = ah_result['energy']
    print(f"  Non-relativistic energy: {e_ah_nonrel:.8f} Hartree")
    
    # For hydrogen-like atoms, relativistic correction scales as Z⁴
    Z = 1  # Anti-hydrogen
    rel_correction_ah = (alpha**2 / 2) * Z**4 * abs(e_ah_nonrel)
    
    print(f"  Relativistic correction: {rel_correction_ah:.8f} Hartree")
    print(f"  Corrected energy: {e_ah_nonrel + rel_correction_ah:.8f} Hartree")
    
    # 3. Heavy antimatter (Anti-helium)
    print("\n3. Anti-helium ion (Anti-He⁺) - stronger relativistic effects:")
    
    antihe_data = MolecularData(
        atoms=[('He', np.array([0.0, 0.0, 0.0]))],
        n_electrons=0,
        n_positrons=1,
        charge=-1,
        name='Anti-He+'
    )
    
    antihe_result = calc.calculate_custom_system(
        atoms=antihe_data.atoms,
        n_electrons=0,
        n_positrons=1,
        accuracy='medium'
    )
    
    e_antihe_nonrel = antihe_result['energy']
    print(f"  Non-relativistic energy: {e_antihe_nonrel:.6f} Hartree")
    
    # For He⁺-like ions, Z=2
    Z = 2
    if abs(e_antihe_nonrel) > 1e-10:  # Check for non-zero energy
        rel_correction_he = (alpha**2 / 2) * Z**4 * abs(e_antihe_nonrel)
        print(f"  Relativistic correction (Z⁴ scaling): {rel_correction_he:.6f} Hartree")
        print(f"  Corrected energy: {e_antihe_nonrel + rel_correction_he:.6f} Hartree")
        print(f"  Correction: {100*rel_correction_he/abs(e_antihe_nonrel):.3f}%")
    else:
        print(f"  Warning: Non-relativistic energy is zero - calculation may have failed")
        print(f"  Skipping relativistic corrections")
    
    # 4. Velocity distribution analysis
    print("\n4. Particle velocities in antimatter systems:")
    
    # Average velocity scales as Z/n for hydrogen-like systems
    # v/c ≈ Z·α/n
    
    systems = [
        ("Positronium (n=1)", 0.5 * alpha, 0.5),  # Z_eff = 0.5 for Ps
        ("Anti-H (n=1)", 1 * alpha, 1),
        ("Anti-He⁺ (n=1)", 2 * alpha, 2),
        ("Anti-C⁵⁺ (n=1)", 6 * alpha, 6),  # Hypothetical
    ]
    
    print("\n  System               v/c        v (a.u.)   Relativistic?")
    print("  " + "-" * 58)
    for name, v_over_c, Z in systems:
        v_au = v_over_c * c
        is_rel = "Yes" if v_over_c > 0.01 else "No"
        print(f"  {name:20s} {v_over_c:.4f}    {v_au:7.3f}    {is_rel}")
    
    # 5. Lamb shift estimate
    print("\n5. Lamb shift in antimatter (QED effects):")
    
    # Lamb shift ≈ (α⁵/π) × m·c² × Z⁴ × f(n,l)
    # For 2S-2P splitting in hydrogen: ~1057 MHz
    
    lamb_shift_h = 1057.8  # MHz for hydrogen 2S-2P
    lamb_shift_ps = lamb_shift_h * 0.5**4  # Scales as Z⁴
    lamb_shift_antihe = lamb_shift_h * 2**4  # Anti-He⁺
    
    print(f"\n  2S-2P Lamb shift:")
    print(f"    Hydrogen: {lamb_shift_h:.1f} MHz")
    print(f"    Positronium: {lamb_shift_ps:.1f} MHz")
    print(f"    Anti-He⁺: {lamb_shift_antihe:.1f} MHz")
    
    # 6. Hyperfine structure
    print("\n6. Hyperfine structure (magnetic interactions):")
    
    # Using the RelativisticCorrection class
    print("\n  Spin-orbit coupling estimates:")
    
    for L in [0, 1, 2]:
        for S in [0.5, 1.0]:
            so_coupling = rel_corr.calculate_spin_orbit_coupling(L, S)
            if L > 0:  # No coupling for s-orbitals
                print(f"    L={L}, S={S}: {so_coupling:.6f} Hartree")

if __name__ == "__main__":
    main()