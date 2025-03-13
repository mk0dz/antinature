import numpy as np
import matplotlib.pyplot as plt

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from antimatter_qchem.core import * 
import numpy as np
import matplotlib.pyplot as plt

def test_hamiltonian_construction():
    """Test the construction of Hamiltonian for a simple system."""
    print("Testing Hamiltonian construction...")
    
    # Create a simple system: positronium (electron + positron)
    nuclei = []  # No nuclei for positronium
    n_electrons = 1
    n_positrons = 1
    
    # Create a Hamiltonian object
    hamiltonian = AntimatterHamiltonian(
        nuclei, 
        n_electrons, 
        n_positrons,
        include_annihilation=True,
        include_relativistic=True
    )
    
    print(f"Created Hamiltonian for positronium:")
    print(f"  Electrons: {n_electrons}")
    print(f"  Positrons: {n_positrons}")
    print(f"  Include annihilation: {hamiltonian.include_annihilation}")
    print(f"  Include relativistic: {hamiltonian.include_relativistic}")
    
    # Create a simple basis set
    electron_basis = BasisSet()
    positron_basis = PositronBasis()
    
    # Add basis functions for electron and positron
    center = np.array([0.0, 0.0, 0.0])
    
    # For electron: add a few s-type functions with different exponents
    electron_exponents = [5.0, 1.2, 0.3]
    for exp in electron_exponents:
        electron_basis.add_function(GaussianBasisFunction(center, exp, (0, 0, 0)))
    
    # For positron: add more diffuse functions
    positron_exponents = [2.0, 0.6, 0.15]
    for exp in positron_exponents:
        positron_basis.add_function(GaussianBasisFunction(center, exp, (0, 0, 0)))
    
    # Create mixed basis
    mixed_basis = MixedMatterBasis(electron_basis, positron_basis)
    
    print(f"\nBasis set information:")
    print(f"  Electron basis functions: {mixed_basis.n_electron_basis}")
    print(f"  Positron basis functions: {mixed_basis.n_positron_basis}")
    print(f"  Total basis functions: {mixed_basis.n_total_basis}")
    
    # Compute integrals
    print("\nComputing integrals...")
    hamiltonian.compute_integrals(mixed_basis)
    
    # Check the computed matrices
    print("\nComputed matrix dimensions:")
    print(f"  Overlap: {hamiltonian.overlap.shape}")
    print(f"  Kinetic: {hamiltonian.kinetic.shape}")
    print(f"  Nuclear attraction: {hamiltonian.nuclear_attraction.shape}")
    print(f"  Electron repulsion: {hamiltonian.electron_repulsion.shape}")
    
    if n_positrons > 0:
        print(f"  Positron kinetic: {hamiltonian.positron_kinetic.shape}")
        print(f"  Positron nuclear: {hamiltonian.positron_nuclear.shape}")
        print(f"  Positron repulsion: {hamiltonian.positron_repulsion.shape}")
        print(f"  Electron-positron attraction: {hamiltonian.electron_positron_attraction.shape}")
        
        if hamiltonian.include_annihilation:
            print(f"  Annihilation: {hamiltonian.annihilation.shape}")
    
    # Check for relativistic corrections
    if hamiltonian.include_relativistic:
        print("\nRelativistic corrections:")
        print(f"  Mass-velocity: {hamiltonian.mass_velocity.shape}")
        print(f"  Darwin: {hamiltonian.darwin.shape}")
    
    # Build the Hamiltonian
    print("\nBuilding complete Hamiltonian...")
    hamiltonian_dict = hamiltonian.build_hamiltonian()
    
    print("\nHamiltonian components:")
    for key, value in hamiltonian_dict.items():
        if value is not None:
            shape_str = f"{value.shape}" if hasattr(value, 'shape') else "N/A"
            print(f"  {key}: {shape_str}")
    
    return hamiltonian_dict, mixed_basis

def test_h2_molecule_hamiltonian():
    """Test Hamiltonian construction for H2 molecule."""
    print("\nTesting Hamiltonian for H2 molecule...")
    
    # Define H2 molecule
    bond_length = 1.4  # Bohr
    h1_pos = np.array([0.0, 0.0, 0.0])
    h2_pos = np.array([0.0, 0.0, bond_length])
    
    nuclei = [
        ('H', 1.0, h1_pos),
        ('H', 1.0, h2_pos)
    ]
    
    n_electrons = 2
    n_positrons = 1  # Add one positron
    
    # Create a Hamiltonian object
    hamiltonian = AntimatterHamiltonian(
        nuclei, 
        n_electrons, 
        n_positrons,
        include_annihilation=True,
        include_relativistic=False  # Turn off relativistic effects for simplicity
    )
    
    # Create a mixed basis set for H2
    mixed_basis = MixedMatterBasis()
    
    # Define H2 molecule for basis set creation
    h2_molecule = [
        ('H', h1_pos),
        ('H', h2_pos)
    ]
    
    # Create basis sets for the molecule
    mixed_basis.create_for_molecule(h2_molecule, 'standard', 'standard')
    
    print(f"\nBasis set information:")
    print(f"  Electron basis functions: {mixed_basis.n_electron_basis}")
    print(f"  Positron basis functions: {mixed_basis.n_positron_basis}")
    print(f"  Total basis functions: {mixed_basis.n_total_basis}")
    
    # Compute integrals
    print("\nComputing integrals...")
    hamiltonian.compute_integrals(mixed_basis)
    
    # Build the Hamiltonian
    print("\nBuilding complete Hamiltonian...")
    hamiltonian_dict = hamiltonian.build_hamiltonian()
    
    # Analyze Hamiltonian components
    print("\nAnalyzing H2 Hamiltonian components...")
    
    # Check core Hamiltonian eigenvalues (electron part)
    if 'H_core_electron' in hamiltonian_dict and hamiltonian_dict['H_core_electron'] is not None:
        H_core_e = hamiltonian_dict['H_core_electron']
        eigenvalues, _ = np.linalg.eigh(H_core_e)
        
        print("\nCore Hamiltonian eigenvalues (electronic part):")
        for i, ev in enumerate(eigenvalues):
            print(f"  Eigenvalue {i}: {ev:.6f} Hartree")
        
        # Plot eigenvalue spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(eigenvalues)), eigenvalues, 'o-')
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Energy (Hartree)')
        plt.title('H2 Electronic Core Hamiltonian Eigenvalue Spectrum')
        plt.grid(True)
        plt.savefig('h2_core_hamiltonian_spectrum.png')
        print("Plot saved as 'h2_core_hamiltonian_spectrum.png'")
    
    # Check nuclear repulsion
    nuclear_repulsion = 0.0
    for i, (elem1, charge1, pos1) in enumerate(nuclei):
        for j, (elem2, charge2, pos2) in enumerate(nuclei):
            if i < j:  # Avoid double counting
                distance = np.linalg.norm(pos1 - pos2)
                nuclear_repulsion += (charge1 * charge2) / distance
    
    print(f"\nNuclear repulsion energy: {nuclear_repulsion:.6f} Hartree")
    
    # Check electron-positron attraction
    if 'electron_positron_attraction' in hamiltonian_dict and hamiltonian_dict['electron_positron_attraction'] is not None:
        ep_attraction = hamiltonian_dict['electron_positron_attraction']
        print(f"\nElectron-positron attraction tensor shape: {ep_attraction.shape}")
        
        # Check if tensor elements have expected signs (negative for attraction)
        negative_count = np.sum(ep_attraction < 0)
        total_elements = np.prod(ep_attraction.shape)
        print(f"  Negative elements (attractive): {negative_count}/{total_elements} ({100*negative_count/total_elements:.1f}%)")
        print(f"  Min value: {np.min(ep_attraction):.6f}")
        print(f"  Max value: {np.max(ep_attraction):.6f}")
    
    # Check annihilation terms
    if 'annihilation' in hamiltonian_dict and hamiltonian_dict['annihilation'] is not None:
        annihilation = hamiltonian_dict['annihilation']
        print(f"\nAnnihilation matrix shape: {annihilation.shape}")
        print(f"  Max annihilation element: {np.max(np.abs(annihilation)):.6f}")
        
        # Calculate overlap between electron and positron basis functions
        # to compare with annihilation matrix elements
        if 'overlap' in hamiltonian_dict and hamiltonian_dict['overlap'] is not None:
            overlap = hamiltonian_dict['overlap']
            n_electron_basis = mixed_basis.n_electron_basis
            n_positron_basis = mixed_basis.n_positron_basis
            
            # Extract the electron-positron block of the overlap matrix
            if n_electron_basis > 0 and n_positron_basis > 0:
                ep_overlap = overlap[:n_electron_basis, n_electron_basis:]
                
                print(f"  Comparison of e-p overlap and annihilation matrix:")
                print(f"    Overlap max: {np.max(np.abs(ep_overlap)):.6f}")
                if annihilation.shape == ep_overlap.shape:
                    correlation = np.corrcoef(annihilation.flatten(), ep_overlap.flatten())[0, 1]
                    print(f"    Correlation coefficient: {correlation:.6f}")
    
    return hamiltonian_dict, mixed_basis

def analyze_hamiltonian(hamiltonian_dict, basis):
    """Analyze Hamiltonian components for physical insights."""
    print("\nDetailed Hamiltonian Analysis:")
    
    # Check if we have all necessary components
    if ('H_core_electron' not in hamiltonian_dict or 
        'electron_repulsion' not in hamiltonian_dict or
        hamiltonian_dict['H_core_electron'] is None or
        hamiltonian_dict['electron_repulsion'] is None):
        print("  Missing essential Hamiltonian components for analysis")
        return
    
    # Get dimensions
    n_basis = hamiltonian_dict['H_core_electron'].shape[0]
    
    # 1. Analyze one-electron terms
    H_core = hamiltonian_dict['H_core_electron']
    print(f"\nOne-electron Hamiltonian statistics:")
    print(f"  Average diagonal element: {np.mean(np.diag(H_core)):.6f} Hartree")
    print(f"  Diagonal element range: [{np.min(np.diag(H_core)):.6f}, {np.max(np.diag(H_core)):.6f}] Hartree")
    print(f"  Average off-diagonal element: {np.mean(H_core - np.diag(np.diag(H_core))):.6f} Hartree")
    
    # 2. Analyze two-electron terms (electron repulsion integrals)
    ERI = hamiltonian_dict['electron_repulsion']
    
    # Coulomb integrals (ii|jj)
    coulomb_values = []
    for i in range(n_basis):
        for j in range(n_basis):
            coulomb_values.append(ERI[i, i, j, j])
    
    # Exchange integrals (ij|ji)
    exchange_values = []
    for i in range(n_basis):
        for j in range(n_basis):
            exchange_values.append(ERI[i, j, j, i])
    
    print(f"\nTwo-electron integral statistics:")
    print(f"  Average Coulomb integral: {np.mean(coulomb_values):.6f} Hartree")
    print(f"  Coulomb integral range: [{np.min(coulomb_values):.6f}, {np.max(coulomb_values):.6f}] Hartree")
    print(f"  Average exchange integral: {np.mean(exchange_values):.6f} Hartree")
    print(f"  Exchange integral range: [{np.min(exchange_values):.6f}, {np.max(exchange_values):.6f}] Hartree")
    
    # 3. Compare electron-positron terms if available
    if ('H_core_positron' in hamiltonian_dict and 
        'electron_positron_attraction' in hamiltonian_dict and
        hamiltonian_dict['H_core_positron'] is not None and
        hamiltonian_dict['electron_positron_attraction'] is not None):
        
        H_core_p = hamiltonian_dict['H_core_positron']
        ep_attraction = hamiltonian_dict['electron_positron_attraction']
        
        print(f"\nElectron-positron interaction statistics:")
        print(f"  Average positron core H element: {np.mean(H_core_p):.6f} Hartree")
        print(f"  Average e-p attraction element: {np.mean(ep_attraction):.6f} Hartree")
        
        # 4. Analyze annihilation terms if available
        if ('annihilation' in hamiltonian_dict and 
            hamiltonian_dict['annihilation'] is not None):
            
            ann = hamiltonian_dict['annihilation']
            print(f"  Average annihilation element: {np.mean(ann):.6f}")
            print(f"  Max annihilation element: {np.max(ann):.6f}")
            print(f"  Annihilation element standard deviation: {np.std(ann):.6f}")

if __name__ == "__main__":
    print("=== Hamiltonian Construction Testing ===\n")
    
    # Test positronium Hamiltonian
    positronium_hamiltonian, positronium_basis = test_hamiltonian_construction()
    
    # Test H2 molecule Hamiltonian
    h2_hamiltonian, h2_basis = test_h2_molecule_hamiltonian()
    
    # Detailed analysis of H2 Hamiltonian
    analyze_hamiltonian(h2_hamiltonian, h2_basis)
    
    print("\nAll Hamiltonian tests completed.")