#!/usr/bin/env python
"""
Example 10: Annihilation Dynamics and Lifetime Analysis
=======================================================
This example explores electron-positron annihilation processes,
cross sections, decay channels, and lifetime calculations.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature import MolecularData, AntinatureCalculator
from antinature.specialized.annihilation import AnnihilationOperator
from antinature.specialized.positronium import PositroniumSCF

def main():
    print("=" * 60)
    print("EXAMPLE 10: ANNIHILATION DYNAMICS")
    print("=" * 60)
    
    # Create annihilation operator
    ann_op = AnnihilationOperator()
    
    # 1. Positronium annihilation channels
    print("\n1. Positronium annihilation channels:")
    
    ps_scf = PositroniumSCF()
    
    # Para-positronium (singlet, S=0)
    para_result = ps_scf.solve(state='para')
    print(f"\n  Para-positronium (¹S₀):")
    print(f"    Energy: {para_result['energy']:.6f} Hartree")
    print(f"    Spin: S=0 (singlet)")
    print(f"    Decay: 2γ (two photons)")
    print(f"    Branching ratio: ~100%")
    
    # Calculate 2γ decay rate
    # τ(2γ) = 1/(2 × α⁵ × m_e × c²) ≈ 125 ps
    alpha = 1/137.036
    tau_2gamma = 125e-12  # seconds
    rate_2gamma = 1/tau_2gamma
    
    print(f"    Lifetime: {tau_2gamma*1e12:.1f} ps")
    print(f"    Decay rate: {rate_2gamma:.2e} s⁻¹")
    
    # Ortho-positronium (triplet, S=1)
    ortho_result = ps_scf.solve(state='ortho')
    print(f"\n  Ortho-positronium (³S₁):")
    print(f"    Energy: {ortho_result['energy']:.6f} Hartree")
    print(f"    Spin: S=1 (triplet)")
    print(f"    Decay: 3γ (three photons)")
    print(f"    Branching ratio: ~100%")
    
    # τ(3γ) = 1.42 × 10⁻⁷ s (much longer than 2γ)
    tau_3gamma = 142e-9  # seconds
    rate_3gamma = 1/tau_3gamma
    
    print(f"    Lifetime: {tau_3gamma*1e9:.1f} ns")
    print(f"    Decay rate: {rate_3gamma:.2e} s⁻¹")
    
    # 2. Energy-dependent cross sections
    print("\n2. Annihilation cross section vs energy:")
    
    energies = np.logspace(-3, 3, 50)  # MeV
    cross_sections = []
    
    for E in energies:
        sigma = ann_op.calculate_cross_section(E)
        cross_sections.append(sigma)
    
    # Find resonance peaks
    max_idx = np.argmax(cross_sections)
    print(f"\n  Peak cross section at E = {energies[max_idx]:.3f} MeV")
    print(f"  σ_max = {cross_sections[max_idx]:.2e} cm²")
    
    # Low energy limit (Rydberg formula)
    E_low = 0.001  # MeV
    sigma_low = ann_op.calculate_cross_section(E_low)
    print(f"\n  Low energy (E = {E_low} MeV):")
    print(f"    σ = {sigma_low:.2e} cm²")
    
    # High energy limit
    E_high = 1000  # MeV
    sigma_high = ann_op.calculate_cross_section(E_high)
    print(f"\n  High energy (E = {E_high} MeV):")
    print(f"    σ = {sigma_high:.2e} cm²")
    
    # 3. Annihilation in different systems
    print("\n3. Annihilation rates in various systems:")
    
    systems = [
        ("Free e⁺e⁻ (thermal)", 1e10, 1e10, 0.01),
        ("Positronium (ground)", 1e30, 1e30, 1.0),
        ("Ps in solid", 1e28, 1e28, 0.5),
        ("Ps in gas", 1e20, 1e20, 0.1),
        ("PsH molecule", 1e25, 1e24, 0.3),
    ]
    
    print("\n  System                e⁻ density   e⁺ density   Overlap   Rate (s⁻¹)   Lifetime")
    print("  " + "-" * 80)
    
    for name, n_e, n_p, overlap in systems:
        rate = ann_op.calculate_annihilation_rate(n_e, n_p, overlap)
        lifetime = 1/rate if rate > 0 else float('inf')
        
        if lifetime < 1e-9:
            lifetime_str = f"{lifetime*1e12:.1f} ps"
        elif lifetime < 1e-6:
            lifetime_str = f"{lifetime*1e9:.1f} ns"
        elif lifetime < 1e-3:
            lifetime_str = f"{lifetime*1e6:.1f} μs"
        else:
            lifetime_str = f"{lifetime:.2e} s"
        
        print(f"  {name:20s}  {n_e:.1e}    {n_p:.1e}    {overlap:.2f}     {rate:.2e}    {lifetime_str}")
    
    # 4. Pick-off annihilation in matter
    print("\n4. Pick-off annihilation (Ps in matter):")
    
    print("\n  When positronium enters matter:")
    print("    - Positron can annihilate with medium electrons")
    print("    - This shortens ortho-Ps lifetime dramatically")
    
    # Normal ortho-Ps lifetime
    tau_vacuum = 142e-9  # seconds
    
    # In different media
    media = [
        ("Vacuum", tau_vacuum, 1.0),
        ("Low density gas", 100e-9, 0.70),
        ("Liquid", 3e-9, 0.02),
        ("Solid metal", 0.5e-9, 0.003),
    ]
    
    print("\n  Medium           Lifetime    Relative to vacuum")
    print("  " + "-" * 45)
    for medium, tau, relative in media:
        print(f"  {medium:15s}  {tau*1e9:6.1f} ns    {relative:.3f}")
    
    # 5. Photon spectrum from annihilation
    print("\n5. Annihilation photon spectrum:")
    
    print("\n  2γ annihilation (para-Ps):")
    print("    - Each photon: 511 keV (electron rest mass)")
    print("    - Back-to-back emission (momentum conservation)")
    print("    - Sharp line in spectrum")
    
    print("\n  3γ annihilation (ortho-Ps):")
    print("    - Total energy: 1022 keV")
    print("    - Continuous spectrum for each photon")
    print("    - Maximum photon energy: 511 keV")
    print("    - Planar emission (momentum conservation)")
    
    # 6. Annihilation in flight
    print("\n6. Annihilation in flight (moving positrons):")
    
    # Doppler broadening
    T = 300  # K (room temperature)
    k_B = 8.617e-5  # eV/K
    m_e = 511e3  # eV/c²
    
    # Thermal velocity
    v_thermal = np.sqrt(2 * k_B * T / m_e)  # in units of c
    
    # Doppler broadening
    delta_E = 511 * v_thermal  # keV
    
    print(f"\n  Room temperature ({T} K):")
    print(f"    Thermal velocity: {v_thermal:.4f} c")
    print(f"    Doppler broadening: ±{delta_E:.2f} keV")
    print(f"    Line width: {2*delta_E:.2f} keV FWHM")
    
    # 7. Create visualization
    print("\n7. Creating annihilation visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Cross section vs energy
    ax1 = axes[0, 0]
    ax1.loglog(energies, cross_sections, 'b-', linewidth=2)
    ax1.set_xlabel('Energy (MeV)')
    ax1.set_ylabel('Cross Section (cm²)')
    ax1.set_title('e⁺e⁻ Annihilation Cross Section')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Lifetime comparison
    ax2 = axes[0, 1]
    systems_names = ['Para-Ps', 'Ortho-Ps', 'Ps in gas', 'Ps in solid']
    lifetimes = [125e-12, 142e-9, 50e-9, 1e-9]
    colors = ['red', 'blue', 'green', 'orange']
    bars = ax2.bar(systems_names, np.log10(np.array(lifetimes)*1e9), color=colors)
    ax2.set_ylabel('log₁₀(Lifetime in ns)')
    ax2.set_title('Positronium Lifetimes')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Photon spectrum (schematic)
    ax3 = axes[1, 0]
    E_gamma = np.linspace(0, 511, 1000)
    
    # 2γ spectrum (delta function at 511 keV)
    spectrum_2g = np.zeros_like(E_gamma)
    spectrum_2g[np.argmin(np.abs(E_gamma - 511))] = 1.0
    
    # 3γ spectrum (continuous)
    spectrum_3g = np.sqrt(np.maximum(0, 1 - (E_gamma/511)**2))
    spectrum_3g /= np.max(spectrum_3g)
    
    ax3.plot(E_gamma, spectrum_2g, 'r-', label='2γ (para-Ps)', linewidth=2)
    ax3.plot(E_gamma, 0.3*spectrum_3g, 'b-', label='3γ (ortho-Ps)', linewidth=2)
    ax3.set_xlabel('Photon Energy (keV)')
    ax3.set_ylabel('Intensity (a.u.)')
    ax3.set_title('Annihilation Photon Spectra')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Doppler broadening
    ax4 = axes[1, 1]
    E_center = 511  # keV
    E_range = np.linspace(E_center - 5, E_center + 5, 1000)
    
    for T in [77, 300, 1000]:  # K
        v_th = np.sqrt(2 * k_B * T / m_e)
        sigma = E_center * v_th
        gaussian = np.exp(-(E_range - E_center)**2 / (2*sigma**2))
        ax4.plot(E_range, gaussian/np.max(gaussian), label=f'T = {T} K', linewidth=2)
    
    ax4.set_xlabel('Photon Energy (keV)')
    ax4.set_ylabel('Intensity (normalized)')
    ax4.set_title('Doppler Broadening of 511 keV Line')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('annihilation_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'annihilation_analysis.png'")
    
    plt.show()

if __name__ == "__main__":
    main()