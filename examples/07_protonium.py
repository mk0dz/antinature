#!/usr/bin/env python
"""
Example 7: Protonium (Antiproton-Proton System)
===============================================
This example simulates protonium, an exotic atom consisting of
a proton and an antiproton orbiting their common center of mass.
This system can form before annihilation occurs.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature import MolecularData, AntinatureCalculator
from antinature.specialized.annihilation import AnnihilationOperator

def main():
    print("=" * 60)
    print("EXAMPLE 7: PROTONIUM (p̄p SYSTEM)")
    print("=" * 60)
    
    # Create protonium system
    print("\n1. Creating protonium system:")
    
    # Place proton and antiproton at typical separation
    bond_length = 2.0  # Bohr radii
    
    protonium_data = MolecularData(
        atoms=[
            ('H+', np.array([0.0, 0.0, -bond_length/2])),  # Proton
            ('H-', np.array([0.0, 0.0, bond_length/2]))    # Antiproton
        ],
        n_electrons=0,  # No electrons
        n_positrons=0,  # No positrons
        charge=0,
        name='Protonium',
        description='Proton-antiproton bound system'
    )
    
    print(f"  Name: {protonium_data.name}")
    print(f"  Description: {protonium_data.description}")
    print(f"  Separation: {bond_length:.2f} Bohr")
    
    # Note: Pure hadronic system - simplified treatment
    print("\n2. Energy estimation (simplified model):")
    
    # Protonium is primarily a hadronic system
    # We use a simplified Coulomb-like potential
    
    # Reduced mass of proton-antiproton system
    proton_mass = 1836.15  # electron masses
    reduced_mass = proton_mass / 2
    
    # Energy levels similar to positronium but scaled by mass
    n = 1  # Ground state
    rydberg = 13.6057  # eV
    energy_ev = -rydberg * (reduced_mass / 1) / (n**2)
    energy_hartree = energy_ev / 27.2114
    
    print(f"  Reduced mass: {reduced_mass:.1f} electron masses")
    print(f"  Estimated ground state: {energy_hartree:.3f} Hartree")
    print(f"                        = {energy_ev:.1f} eV")
    
    # Strong force effects (qualitative)
    print("\n3. Strong force considerations:")
    print("  Note: At short distances, strong force dominates")
    print("  This calculation uses electromagnetic interaction only")
    print("  Real protonium involves complex QCD dynamics")
    
    # Annihilation analysis
    print("\n4. Annihilation characteristics:")
    
    ann_op = AnnihilationOperator()
    
    # Proton-antiproton annihilation cross section
    # Much larger than e⁺e⁻ due to strong interaction
    energy_cm = 2 * 938.3  # MeV (rest mass energy)
    
    # Simplified cross section estimate
    cross_section_mb = 50.0  # millibarns (typical for low energy)
    cross_section_m2 = cross_section_mb * 1e-31
    
    print(f"  Center-of-mass energy: {energy_cm:.1f} MeV")
    print(f"  Annihilation cross section: {cross_section_mb:.1f} mb")
    
    # Lifetime estimate
    # τ ~ 1/(n·σ·v) where n is density, σ is cross section, v is velocity
    
    # Orbital velocity estimate
    velocity = 2.2e6 / reduced_mass  # m/s (scaled from Bohr velocity)
    
    # Wavefunction overlap at origin (simplified)
    overlap_density = 1e30  # m⁻³ (order of magnitude)
    
    lifetime_s = 1 / (overlap_density * cross_section_m2 * velocity)
    
    print(f"\n  Estimated orbital velocity: {velocity:.2e} m/s")
    print(f"  Estimated lifetime: {lifetime_s*1e12:.1f} ps")
    
    # Annihilation products
    print("\n5. Annihilation products:")
    print("  Typical p̄p annihilation produces:")
    print("    - 3-7 pions (average ~5)")
    print("    - π⁺, π⁻, π⁰ in roughly equal numbers")
    print("    - Occasional kaons or other mesons")
    print("    - Total energy: 2 × 938.3 MeV = 1.88 GeV")
    
    # Spectroscopy
    print("\n6. Spectroscopic properties:")
    
    # Transition energies scale with reduced mass
    lyman_alpha_h = 10.2  # eV for hydrogen
    lyman_alpha_protonium = lyman_alpha_h * reduced_mass
    
    print(f"  Lyman-α transition:")
    print(f"    Hydrogen: {lyman_alpha_h:.1f} eV")
    print(f"    Protonium: {lyman_alpha_protonium:.1f} eV = {lyman_alpha_protonium/1000:.2f} keV")
    
    # X-ray regime
    print(f"\n  → Protonium transitions are in the X-ray regime")
    print(f"  → Can be detected with X-ray spectroscopy")

if __name__ == "__main__":
    main()