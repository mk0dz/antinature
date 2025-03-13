import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from antimatter_qchem import *



def test_annihilation_operator():
    """Test the annihilation operator functionality."""
    print("Testing annihilation operator...")
    
    # Create simple electron and positron basis sets
    electron_basis = BasisSet()
    positron_basis = PositronBasis()
    
    # Add basis functions for electron and positron
    center = np.array([0.0, 0.0, 0.0])
    
    # For electron: add a few s-type functions with different exponents
    electron_exponents = [5.0, 1.2, 0.3]
    for exp in electron_exponents:
        electron_basis.add_function(GaussianBasisFunction(center, exp, (0, 0, 0)))
    
    # For positron: add more diffuse functions
    positron_exponents = [2.5, 0.6, 0.15]
    for exp in positron_exponents:
        positron_basis.add_function(GaussianBasisFunction(center, exp, (0, 0, 0)))
    
    print(f"\nBasis set information:")
    print(f"  Electron basis functions: {electron_basis.n_basis}")
    print(f"  Positron basis functions: {positron_basis.n_basis}")
    
    # Create annihilation operator
    annihilation_op = AnnihilationOperator(electron_basis, positron_basis)
    
    # Build annihilation operator
    print("\nBuilding annihilation operator...")
    ann_matrix = annihilation_op.build_annihilation_operator()
    
    print(f"Annihilation matrix shape: {ann_matrix.shape}")
    print(f"Average annihilation element: {np.mean(ann_matrix):.6e}")
    print(f"Maximum annihilation element: {np.max(ann_matrix):.6e}")
    
    # Test annihilation rate calculation
    # Create simple density matrices (assuming ground state occupation)
    P_e = np.zeros((electron_basis.n_basis, electron_basis.n_basis))
    P_p = np.zeros((positron_basis.n_basis, positron_basis.n_basis))
    
    # Occupy lowest orbital (assume coefficient 1.0 for simplicity)
    P_e[0, 0] = 1.0
    P_p[0, 0] = 1.0
    
    # Calculate annihilation rate
    rate = annihilation_op.calculate_annihilation_rate(P_e, P_p)
    print(f"\nAnnihilation rate for ground state density: {rate:.6e}")
    
    # Test with different electron-positron separations
    separations = np.linspace(0.0, 5.0, 10)
    rates = []
    
    for sep in separations:
        # Create new basis sets with separated centers
        e_basis = BasisSet()
        p_basis = PositronBasis()
        
        e_center = np.array([-sep/2, 0.0, 0.0])
        p_center = np.array([sep/2, 0.0, 0.0])
        
        for exp in electron_exponents:
            e_basis.add_function(GaussianBasisFunction(e_center, exp, (0, 0, 0)))
        
        for exp in positron_exponents:
            p_basis.add_function(GaussianBasisFunction(p_center, exp, (0, 0, 0)))
        
        # Create new annihilation operator
        ann_op = AnnihilationOperator(e_basis, p_basis)
        ann_matrix = ann_op.build_annihilation_operator()
        
        # Create density matrices
        P_e = np.zeros((e_basis.n_basis, e_basis.n_basis))
        P_p = np.zeros((p_basis.n_basis, p_basis.n_basis))
        P_e[0, 0] = 1.0
        P_p[0, 0] = 1.0
        
        # Calculate rate
        rate = ann_op.calculate_annihilation_rate(P_e, P_p)
        rates.append(rate)
    
    # Plot annihilation rate vs. separation
    plt.figure(figsize=(10, 6))
    plt.plot(separations, rates, 'o-')
    plt.xlabel('Electron-Positron Separation (Bohr)')
    plt.ylabel('Annihilation Rate')
    plt.title('Annihilation Rate vs. Separation')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('annihilation_rate_vs_separation.png')
    print("Plot saved as 'annihilation_rate_vs_separation.png'")
    
    # Analyze annihilation channels
    print("\nAnalyzing annihilation channels...")
    
    # Create a simple wavefunction representation
    wavefunction = {
        'C_electron': np.eye(electron_basis.n_basis),  # Identity for simplicity
        'C_positron': np.eye(positron_basis.n_basis),
        'n_electrons': 1,
        'n_positrons': 1
    }
    
    # Analyze channels
    channels = annihilation_op.analyze_annihilation_channels(wavefunction)
    
    print(f"Annihilation channels analysis:")
    print(f"  2γ rate: {channels['two_gamma']:.6e}")
    print(f"  3γ rate: {channels['three_gamma']:.6e}")
    print(f"  Total rate: {channels['total']:.6e}")
    print(f"  2γ/3γ ratio: {channels['ratio_2g_3g']:.6f}")
    
    # Calculate lifetime
    lifetime = annihilation_op.calculate_lifetime(channels['total'])
    print(f"  Estimated lifetime: {lifetime:.6e} seconds")
    
    # Test annihilation density visualization
    print("\nVisualizing annihilation density...")
    density_data = annihilation_op.visualize_annihilation_density(wavefunction, grid_points=20)
    
    if density_data is not None:
        # Create a 2D slice of the 3D density for visualization
        slice_z = density_data['density'].shape[2] // 2
        density_slice = density_data['density'][:, :, slice_z]
        
        # Plot the density slice
        plt.figure(figsize=(8, 8))
        plt.imshow(density_slice, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis')
        plt.colorbar(label='Annihilation Density')
        plt.xlabel('x (Bohr)')
        plt.ylabel('y (Bohr)')
        plt.title('Annihilation Density (z=0 slice)')
        plt.savefig('annihilation_density.png')
        print("Plot saved as 'annihilation_density.png'")
    
    return annihilation_op

def test_relativistic_corrections():
    """Test relativistic correction functionality."""
    print("\nTesting relativistic corrections...")
    
    # Create a system with a heavy nucleus (e.g., carbon)
    nuclei = [('C', 6.0, np.array([0.0, 0.0, 0.0]))]
    n_electrons = 6
    n_positrons = 0
    
    # Create a Hamiltonian object with relativistic corrections
    hamiltonian = AntimatterHamiltonian(
        nuclei, 
        n_electrons, 
        n_positrons,
        include_annihilation=False,
        include_relativistic=True
    )
    
    # Create a basis set
    mixed_basis = MixedMatterBasis()
    molecule = [('C', np.array([0.0, 0.0, 0.0]))]
    mixed_basis.create_for_molecule(molecule, 'standard', 'minimal')
    
    print(f"\nBasis set information:")
    print(f"  Electron basis functions: {mixed_basis.n_electron_basis}")
    print(f"  Positron basis functions: {mixed_basis.n_positron_basis}")
    
    # Compute integrals (including relativistic)
    print("\nComputing integrals (including relativistic)...")
    hamiltonian.compute_integrals(mixed_basis)
    
    # Create a relativistic correction object
    rel_correction = RelativisticCorrection(
        hamiltonian.build_hamiltonian(),
        mixed_basis,
        nuclei,
        is_positronic=False
    )
    
    # Calculate scalar relativistic corrections
    print("\nCalculating scalar relativistic corrections...")
    
    # Create a simple wavefunction representation (mock SCF result)
    # Just use identity matrix for density to check functionality
    wavefunction = {
        'P_electron': np.eye(mixed_basis.n_electron_basis) * (n_electrons / mixed_basis.n_electron_basis)
    }
    
    corrections = rel_correction.scalar_relativistic_correction(wavefunction)
    
    print(f"Relativistic energy corrections:")
    print(f"  Mass-velocity: {corrections['mass_velocity']:.8f} Hartree")
    print(f"  Darwin: {corrections['darwin']:.8f} Hartree")
    print(f"  Total correction: {corrections['total']:.8f} Hartree")
    
    # Compare corrections for different elements
    elements = ['H', 'C', 'O', 'Si']
    Z_values = [1, 6, 8, 14]
    mv_corrections = []
    darwin_corrections = []
    
    print("\nRelativistic corrections for different elements:")
    
    for element, Z in zip(elements, Z_values):
        # Create a system with this element
        nuclei = [(element, float(Z), np.array([0.0, 0.0, 0.0]))]
        
        # Create a Hamiltonian object with relativistic corrections
        hamiltonian = AntimatterHamiltonian(
            nuclei, 
            Z,  # Number of electrons = Z for neutral atom
            0,
            include_annihilation=False,
            include_relativistic=True
        )
        
        # Create a basis set
        temp_basis = MixedMatterBasis()
        molecule = [(element, np.array([0.0, 0.0, 0.0]))]
        temp_basis.create_for_molecule(molecule, 'standard', 'minimal')
        
        # Compute integrals
        hamiltonian.compute_integrals(temp_basis)
        
        # Create a relativistic correction object
        temp_rel = RelativisticCorrection(
            hamiltonian.build_hamiltonian(),
            temp_basis,
            nuclei,
            is_positronic=False
        )
        
        # Calculate corrections using a simple density matrix
        P = np.eye(temp_basis.n_electron_basis) * (Z / temp_basis.n_electron_basis)
        corr = temp_rel.scalar_relativistic_correction({'P_electron': P})
        
        print(f"  {element} (Z={Z}):")
        print(f"    Mass-velocity: {corr['mass_velocity']:.8f} Hartree")
        print(f"    Darwin: {corr['darwin']:.8f} Hartree")
        print(f"    Total: {corr['total']:.8f} Hartree")
        
        mv_corrections.append(corr['mass_velocity'])
        darwin_corrections.append(corr['darwin'])
    
    # Plot corrections vs. Z
    plt.figure(figsize=(10, 6))
    plt.plot(Z_values, mv_corrections, 'o-', label='Mass-velocity')
    plt.plot(Z_values, darwin_corrections, 's-', label='Darwin')
    plt.plot(Z_values, [mv + darwin for mv, darwin in zip(mv_corrections, darwin_corrections)], '^-', label='Total')
    plt.xlabel('Nuclear Charge (Z)')
    plt.ylabel('Energy Correction (Hartree)')
    plt.title('Relativistic Corrections vs. Nuclear Charge')
    plt.legend()
    plt.grid(True)
    plt.savefig('relativistic_corrections_vs_Z.png')
    print("Plot saved as 'relativistic_corrections_vs_Z.png'")
    
    # Test ZORA implementation
    print("\nTesting ZORA implementation...")
    
    # Get Hamiltonian components
    h_dict = hamiltonian.build_hamiltonian()
    
    # Apply ZORA corrections
    zora_h = rel_correction.apply_relativistic_corrections(h_dict, method='zora')
    
    # Compare original and ZORA kinetic energy matrices
    if 'kinetic' in h_dict and 'kinetic' in zora_h:
        T_orig = h_dict['kinetic']
        T_zora = zora_h['kinetic']
        
        print(f"Comparison of original and ZORA kinetic energy matrices:")
        print(f"  Original diagonal mean: {np.mean(np.diag(T_orig)):.8f}")
        print(f"  ZORA diagonal mean: {np.mean(np.diag(T_zora)):.8f}")
        print(f"  Ratio (ZORA/Original): {np.mean(np.diag(T_zora))/np.mean(np.diag(T_orig)):.8f}")
    
    return rel_correction

if __name__ == "__main__":
    print("=== Specialized Physics Modules Testing ===\n")
    
    # Test annihilation operator
    annihilation_op = test_annihilation_operator()
    
    # Test relativistic corrections
    rel_correction = test_relativistic_corrections()
    
    print("\nAll specialized physics tests completed.")