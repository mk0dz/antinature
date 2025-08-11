#!/usr/bin/env python
"""
Example 11: Complex Antimatter Molecules
========================================
This example explores various antimatter molecular systems including
anti-water, anti-methane, and mixed matter-antimatter molecules.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature import MolecularData, AntinatureCalculator

def main():
    print("=" * 60)
    print("EXAMPLE 11: COMPLEX ANTIMATTER MOLECULES")
    print("=" * 60)
    
    calc = AntinatureCalculator(print_level=0)
    
    # 1. Anti-water (Anti-H₂O)
    print("\n1. Anti-water molecule (Anti-H₂O):")
    
    # O-H bond length ~0.96 Å = 1.81 Bohr
    # H-O-H angle ~104.5°
    bond_length = 1.81
    angle = 104.5 * np.pi / 180
    
    antiwater = MolecularData(
        atoms=[
            ('O', np.array([0.0, 0.0, 0.0])),  # Anti-oxygen
            ('H', np.array([bond_length, 0.0, 0.0])),  # Anti-H1
            ('H', np.array([
                bond_length * np.cos(angle),
                bond_length * np.sin(angle),
                0.0
            ]))  # Anti-H2
        ],
        n_electrons=0,
        n_positrons=10,  # 8 from O + 1 from each H
        charge=-2,  # Anti-nuclei charge
        name='Anti-H2O',
        description='Anti-water molecule'
    )
    
    print(f"  Name: {antiwater.name}")
    print(f"  Formula: {antiwater.get_formula()}")
    print(f"  Positrons: {antiwater.n_positrons}")
    print(f"  Geometry: Bent (like regular water)")
    print(f"  Bond angle: {angle*180/np.pi:.1f}°")
    
    # Calculate energy
    antiwater_result = calc.calculate_custom_system(
        atoms=antiwater.atoms,
        n_electrons=0,
        n_positrons=10,
        accuracy='low'  # Use low for speed
    )
    
    print(f"  Energy: {antiwater_result['energy']:.4f} Hartree")
    
    # 2. Anti-ammonia (Anti-NH₃)
    print("\n2. Anti-ammonia molecule (Anti-NH₃):")
    
    # Tetrahedral geometry
    N_pos = np.array([0.0, 0.0, 0.0])
    bond_len_NH = 1.91  # Bohr
    
    # Tetrahedral angles
    theta = np.arccos(-1/3)  # ~109.47°
    
    antiammonia = MolecularData(
        atoms=[
            ('N', N_pos),  # Anti-nitrogen
            ('H', N_pos + bond_len_NH * np.array([0, 0, 1])),
            ('H', N_pos + bond_len_NH * np.array([
                np.sin(theta) * np.cos(0),
                np.sin(theta) * np.sin(0),
                np.cos(theta)
            ])),
            ('H', N_pos + bond_len_NH * np.array([
                np.sin(theta) * np.cos(2*np.pi/3),
                np.sin(theta) * np.sin(2*np.pi/3),
                np.cos(theta)
            ]))
        ],
        n_electrons=0,
        n_positrons=10,  # 7 from N + 1 from each H
        charge=-2,
        name='Anti-NH3',
        description='Anti-ammonia molecule'
    )
    
    print(f"  Name: {antiammonia.name}")
    print(f"  Formula: {antiammonia.get_formula()}")
    print(f"  Geometry: Pyramidal")
    print(f"  Bond angle: {theta*180/np.pi:.1f}°")
    
    # 3. Mixed matter-antimatter: HePs (Helium-Positronium)
    print("\n3. Helium-Positronium molecule (HePs):")
    
    heps = MolecularData(
        atoms=[('He', np.array([0.0, 0.0, 0.0]))],  # Regular helium
        n_electrons=3,  # 2 from He + 1 from Ps
        n_positrons=1,  # 1 from Ps
        charge=0,
        name='HePs',
        description='Helium bound to positronium'
    )
    
    print(f"  Name: {heps.name}")
    print(f"  Description: {heps.description}")
    print(f"  Matter part: He (2e⁻)")
    print(f"  Antimatter part: Ps (1e⁻ + 1e⁺)")
    
    heps_result = calc.calculate_custom_system(
        atoms=heps.atoms,
        n_electrons=3,
        n_positrons=1,
        accuracy='medium'
    )
    
    print(f"  Energy: {heps_result['energy']:.6f} Hartree")
    
    # Binding energy analysis
    he_energy = -2.9  # Hartree (approximate)
    ps_energy = -0.25  # Hartree
    binding = heps_result['energy'] - (he_energy + ps_energy)
    
    print(f"\n  Binding analysis:")
    print(f"    He energy: {he_energy:.2f} Hartree")
    print(f"    Ps energy: {ps_energy:.2f} Hartree")
    print(f"    HePs energy: {heps_result['energy']:.6f} Hartree")
    print(f"    Binding energy: {binding:.6f} Hartree = {binding*27.2114:.3f} eV")
    
    # 4. Positronic molecules: e⁺[H₂]
    print("\n4. Positronic hydrogen molecule e⁺[H₂]:")
    
    # H₂ with an extra positron
    h2_bond = 1.4  # Bohr
    
    pos_h2 = MolecularData(
        atoms=[
            ('H', np.array([-h2_bond/2, 0, 0])),
            ('H', np.array([h2_bond/2, 0, 0]))
        ],
        n_electrons=2,
        n_positrons=1,
        charge=1,
        name='e+[H2]',
        description='Hydrogen molecule with bound positron'
    )
    
    print(f"  Name: {pos_h2.name}")
    print(f"  Description: {pos_h2.description}")
    print(f"  Configuration: H-H + e⁺")
    
    pos_h2_result = calc.calculate_custom_system(
        atoms=pos_h2.atoms,
        n_electrons=2,
        n_positrons=1,
        accuracy='medium'
    )
    
    print(f"  Energy: {pos_h2_result['energy']:.6f} Hartree")
    
    # 5. Anti-benzene (C₆H₆ antimatter version)
    print("\n5. Anti-benzene ring (Anti-C₆H₆):")
    
    # Hexagonal ring geometry
    radius = 2.64  # Bohr (C-C distance ~1.4 Å)
    
    carbon_positions = []
    hydrogen_positions = []
    
    for i in range(6):
        angle = i * np.pi / 3
        # Carbon positions
        c_pos = radius * np.array([np.cos(angle), np.sin(angle), 0])
        carbon_positions.append(('C', c_pos))
        
        # Hydrogen positions (radially outward)
        h_pos = 1.4 * c_pos  # C-H bond
        hydrogen_positions.append(('H', h_pos))
    
    antibenzene = MolecularData(
        atoms=carbon_positions + hydrogen_positions,
        n_electrons=0,
        n_positrons=42,  # 6×6 + 6×1 = 42 positrons
        charge=-6,  # Net nuclear charge
        name='Anti-C6H6',
        description='Anti-benzene molecule'
    )
    
    print(f"  Name: {antibenzene.name}")
    print(f"  Formula: {antibenzene.get_formula()}")
    print(f"  Positrons: {antibenzene.n_positrons}")
    print(f"  Structure: Planar hexagonal ring")
    print(f"  Aromatic character: Preserved (by CPT symmetry)")
    
    # 6. Summary table
    print("\n6. Summary of antimatter molecules:")
    print("\n  Molecule        Positrons  Electrons  Type")
    print("  " + "-" * 50)
    
    molecules = [
        ("Anti-H₂O", 10, 0, "Pure antimatter"),
        ("Anti-NH₃", 10, 0, "Pure antimatter"),
        ("Anti-C₆H₆", 42, 0, "Pure antimatter"),
        ("HePs", 1, 3, "Mixed matter-antimatter"),
        ("e⁺[H₂]", 1, 2, "Positronic molecule"),
        ("PsH", 1, 2, "Positronium compound"),
        ("Ps₂", 2, 2, "Pure leptonic"),
    ]
    
    for name, n_pos, n_elec, mol_type in molecules:
        print(f"  {name:12s}    {n_pos:3d}        {n_elec:3d}      {mol_type}")
    
    # 7. Stability considerations
    print("\n7. Stability and experimental feasibility:")
    
    print("\n  Challenges for antimatter molecules:")
    print("    • Annihilation with matter (requires ultra-high vacuum)")
    print("    • Magnetic/electric trapping needed")
    print("    • Short lifetimes (ns to μs range)")
    print("    • Production requires antiproton beams")
    
    print("\n  Experimentally achieved:")
    print("    ✓ Anti-hydrogen (CERN ALPHA, ATRAP)")
    print("    ✓ Anti-helium nuclei (brief)")
    print("    ✓ Positronium and Ps₂")
    print("    ✓ Positronic molecules (e⁺ bound to molecules)")
    
    print("\n  Theoretical predictions:")
    print("    • Anti-water: Stable in isolation")
    print("    • Mixed molecules: Metastable, short-lived")
    print("    • Large antimolecules: Possible but challenging")

if __name__ == "__main__":
    main()