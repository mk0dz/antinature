import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from antimatter_qchem import * 

def test_scf_hydrogen_atom():
    """Test SCF procedure for hydrogen atom."""
    print("Testing SCF procedure for hydrogen atom...")
    
    # Create a system with one proton and one electron
    nuclei = [('H', 1.0, np.array([0.0, 0.0, 0.0]))]
    n_electrons = 1
    n_positrons = 0
    
    # Create a Hamiltonian object
    hamiltonian = AntimatterHamiltonian(
        nuclei, 
        n_electrons, 
        n_positrons,
        include_annihilation=False,
        include_relativistic=False
    )
    
    # Create a basis set for H atom
    mixed_basis = MixedMatterBasis()
    h_atom = [('H', np.array([0.0, 0.0, 0.0]))]
    mixed_basis.create_for_molecule(h_atom, 'standard', 'minimal')
    
    print(f"\nBasis set information:")
    print(f"  Electron basis functions: {mixed_basis.n_electron_basis}")
    print(f"  Positron basis functions: {mixed_basis.n_positron_basis}")
    
    # Compute integrals
    print("\nComputing integrals...")
    hamiltonian.compute_integrals(mixed_basis)
    
    # Build the Hamiltonian
    print("\nBuilding Hamiltonian...")
    hamiltonian_dict = hamiltonian.build_hamiltonian()
    
    # Create SCF object
    scf = AntimatterSCF(
        hamiltonian_dict,
        mixed_basis,
        n_electrons,
        n_positrons
    )
    
    # Run SCF calculation
    print("\nRunning SCF calculation...")
    scf_result = scf.solve_scf(max_iterations=50, convergence_threshold=1e-6)
    
    # Print SCF results
    print("\nSCF Results:")
    print(f"  Converged: {scf_result['converged']}")
    print(f"  Iterations: {scf_result['iterations']}")
    print(f"  Energy: {scf_result['energy']:.8f} Hartree")
    
    # Compare with exact energy for hydrogen atom (-0.5 Hartree)
    exact_energy = -0.5
    error = (scf_result['energy'] - exact_energy)
    print(f"  Exact energy: {exact_energy:.8f} Hartree")
    print(f"  Error: {error:.8f} Hartree ({error/exact_energy*100:.2f}%)")
    
    return scf_result

def test_scf_h2_molecule():
    """Test SCF procedure for H2 molecule."""
    print("\nTesting SCF procedure for H2 molecule...")
    
    # Create H2 molecule (bond length in Bohr)
    bond_length = 1.4  # Typical H2 bond length
    nuclei = [
        ('H', 1.0, np.array([0.0, 0.0, -bond_length/2])),
        ('H', 1.0, np.array([0.0, 0.0, bond_length/2]))
    ]
    n_electrons = 2
    n_positrons = 0
    
    # Create a Hamiltonian object
    hamiltonian = AntimatterHamiltonian(
        nuclei, 
        n_electrons, 
        n_positrons,
        include_annihilation=False,
        include_relativistic=False
    )
    
    # Create a basis set for H2
    mixed_basis = MixedMatterBasis()
    h2_molecule = [
        ('H', np.array([0.0, 0.0, -bond_length/2])),
        ('H', np.array([0.0, 0.0, bond_length/2]))
    ]
    mixed_basis.create_for_molecule(h2_molecule, 'standard', 'minimal')
    
    print(f"\nBasis set information:")
    print(f"  Electron basis functions: {mixed_basis.n_electron_basis}")
    print(f"  Positron basis functions: {mixed_basis.n_positron_basis}")
    
    # Compute integrals
    print("\nComputing integrals...")
    hamiltonian.compute_integrals(mixed_basis)
    
    # Build the Hamiltonian
    print("\nBuilding Hamiltonian...")
    hamiltonian_dict = hamiltonian.build_hamiltonian()
    
    # Create SCF object
    scf = AntimatterSCF(
        hamiltonian_dict,
        mixed_basis,
        n_electrons,
        n_positrons
    )
    
    # Run SCF calculation
    print("\nRunning SCF calculation...")
    scf_result = scf.solve_scf(max_iterations=50, convergence_threshold=1e-6)
    
    # Calculate nuclear repulsion energy
    nuclear_repulsion = 0.0
    for i, (elem1, charge1, pos1) in enumerate(nuclei):
        for j, (elem2, charge2, pos2) in enumerate(nuclei):
            if i < j:
                distance = np.linalg.norm(pos1 - pos2)
                nuclear_repulsion += (charge1 * charge2) / distance
    
    # Print SCF results
    print("\nSCF Results:")
    print(f"  Converged: {scf_result['converged']}")
    print(f"  Iterations: {scf_result['iterations']}")
    print(f"  Electronic energy: {scf_result['energy']:.8f} Hartree")
    print(f"  Nuclear repulsion: {nuclear_repulsion:.8f} Hartree")
    print(f"  Total energy: {scf_result['energy'] + nuclear_repulsion:.8f} Hartree")
    
    # Expected energy for H2 with minimal basis is around -1.13 Hartree
    # This is just an approximate reference
    reference_energy = -1.13
    total_energy = scf_result['energy'] + nuclear_repulsion
    error = (total_energy - reference_energy)
    print(f"  Reference energy: {reference_energy:.8f} Hartree")
    print(f"  Error: {error:.8f} Hartree ({error/reference_energy*100:.2f}%)")
    
    # Analyze molecular orbitals
    if 'C_electron' in scf_result and scf_result['C_electron'] is not None:
        C = scf_result['C_electron']
        print("\nMolecular Orbital Analysis:")
        print(f"  Number of MOs: {C.shape[1]}")
        
        # Calculate the population on each atom (Mulliken population analysis)
        # This is a simplified version
        S = hamiltonian_dict['overlap'][:mixed_basis.n_electron_basis, :mixed_basis.n_electron_basis]
        P = scf_result['P_electron']
        
        # For a minimal basis of H2, we have 2 basis functions (1 per H atom)
        if mixed_basis.n_electron_basis == 2:
            PS = np.dot(P, S)
            atomic_charges = np.zeros(2)
            
            # Assuming the first basis function is on the first atom, 
            # and the second is on the second atom
            atomic_charges[0] = PS[0, 0]
            atomic_charges[1] = PS[1, 1]
            
            print(f"  Atomic populations: {atomic_charges}")
            print(f"  Sum of populations: {np.sum(atomic_charges)} (should be close to {n_electrons})")
            
            # Calculate electron density distribution along bond axis
            z = np.linspace(-3, 3, 100)
            density = np.zeros_like(z)
            
            # Create basis function evaluators
            basis_funcs = mixed_basis.electron_basis.basis_functions
            
            for i, zi in enumerate(z):
                point = np.array([0.0, 0.0, zi])
                values = np.array([func.evaluate(point) for func in basis_funcs])
                
                # Density is ψ†Pψ
                density[i] = np.dot(values, np.dot(P, values))
            
            # Plot electron density
            plt.figure(figsize=(10, 6))
            plt.plot(z, density)
            plt.axvline(-bond_length/2, color='r', linestyle='--', label='H atom 1')
            plt.axvline(bond_length/2, color='r', linestyle='--', label='H atom 2')
            plt.xlabel('z (Bohr)')
            plt.ylabel('Electron Density')
            plt.title('H2 Electron Density Along Bond Axis')
            plt.legend()
            plt.grid(True)
            plt.savefig('h2_electron_density.png')
            print("Plot saved as 'h2_electron_density.png'")
    
    return scf_result

def test_scf_positronium():
    """Test SCF procedure for positronium (e+ + e-)."""
    print("\nTesting SCF procedure for positronium...")
    
    # Create a system with one electron and one positron (no nuclei)
    nuclei = []
    n_electrons = 1
    n_positrons = 1
    
    # Create a Hamiltonian object
    hamiltonian = AntimatterHamiltonian(
        nuclei, 
        n_electrons, 
        n_positrons,
        include_annihilation=True,  # Include annihilation
        include_relativistic=False  # Exclude relativistic effects for simplicity
    )
    
    # Create a basis set for positronium
    # Since there's no nucleus, we'll create basis functions manually
    mixed_basis = MixedMatterBasis()
    
    # Electron basis
    e_basis = mixed_basis.electron_basis
    center = np.array([0.0, 0.0, 0.0])
    for exp in [5.0, 1.2, 0.3]:
        e_basis.add_function(GaussianBasisFunction(center, exp, (0, 0, 0)))
    
    # Positron basis (more diffuse)
    p_basis = mixed_basis.positron_basis
    for exp in [2.5, 0.6, 0.15]:
        p_basis.add_function(GaussianBasisFunction(center, exp, (0, 0, 0)))
    
    # Update basis counts
    mixed_basis.n_electron_basis = e_basis.n_basis
    mixed_basis.n_positron_basis = p_basis.n_basis
    mixed_basis.n_total_basis = mixed_basis.n_electron_basis + mixed_basis.n_positron_basis
    
    print(f"\nBasis set information:")
    print(f"  Electron basis functions: {mixed_basis.n_electron_basis}")
    print(f"  Positron basis functions: {mixed_basis.n_positron_basis}")
    
    # Compute integrals
    print("\nComputing integrals...")
    hamiltonian.compute_integrals(mixed_basis)
    
    # Build the Hamiltonian
    print("\nBuilding Hamiltonian...")
    hamiltonian_dict = hamiltonian.build_hamiltonian()
    
    # Create SCF object
    scf = AntimatterSCF(
        hamiltonian_dict,
        mixed_basis,
        n_electrons,
        n_positrons,
        include_annihilation=True
    )
    
    # Run SCF calculation
    print("\nRunning SCF calculation...")
    scf_result = scf.solve_scf(max_iterations=50, convergence_threshold=1e-6)
    
    # Print SCF results
    print("\nSCF Results:")
    print(f"  Converged: {scf_result['converged']}")
    print(f"  Iterations: {scf_result['iterations']}")
    print(f"  Total energy: {scf_result['energy']:.8f} Hartree")
    
    # Expected energy for positronium is -0.25 Hartree
    # (half of hydrogen because reduced mass is half)
    reference_energy = -0.25
    error = (scf_result['energy'] - reference_energy)
    print(f"  Reference energy: {reference_energy:.8f} Hartree")
    print(f"  Error: {error:.8f} Hartree ({error/reference_energy*100:.2f}%)")
    
    # Analyze density matrices
    if ('P_electron' in scf_result and 'P_positron' in scf_result and
        scf_result['P_electron'] is not None and scf_result['P_positron'] is not None):
        
        P_e = scf_result['P_electron']
        P_p = scf_result['P_positron']
        
        print("\nDensity Matrix Analysis:")
        print(f"  Electron trace: {np.trace(P_e):.6f} (should be close to {n_electrons})")
        print(f"  Positron trace: {np.trace(P_p):.6f} (should be close to {n_positrons})")
        
        # Calculate annihilation rate if annihilation matrix is available
        if 'annihilation' in hamiltonian_dict and hamiltonian_dict['annihilation'] is not None:
            ann = hamiltonian_dict['annihilation']
            
            # Calculate expectation value <Ψ|ann|Ψ>
            # This is a simplified form, not the full calculation
            rate = 0.0
            for i in range(mixed_basis.n_electron_basis):
                for j in range(mixed_basis.n_electron_basis, mixed_basis.n_total_basis):
                    j_rel = j - mixed_basis.n_electron_basis  # Index in positron basis
                    
                    for k in range(mixed_basis.n_electron_basis):
                        for l in range(mixed_basis.n_positron_basis):
                            l_abs = l + mixed_basis.n_electron_basis  # Absolute index
                            
                            if i < P_e.shape[0] and k < P_e.shape[1] and j_rel < P_p.shape[0] and l < P_p.shape[1]:
                                if i < ann.shape[0] and l_abs < ann.shape[1]:
                                    rate += P_e[i, k] * ann[i, l_abs] * P_p[j_rel, l]
            
            print(f"  Estimated annihilation rate: {rate:.6e}")
            
            # Expected lifetime (in atomic units, then converted to ns)
            if rate > 0:
                lifetime_au = 1.0 / rate
                # Convert to nanoseconds: 1 a.u. of time = 2.4189e-17 seconds
                lifetime_ns = lifetime_au * 2.4189e-17 * 1e9
                print(f"  Estimated lifetime: {lifetime_ns:.6f} ns")
                print(f"  Reference lifetime: ~125 ns (para-positronium)")
    
    return scf_result

if __name__ == "__main__":
    print("=== Self-Consistent Field Procedure Testing ===\n")
    
    # Import necessary additional modules
    from antimatter_qchem.core import GaussianBasisFunction
    
    # Test SCF for hydrogen atom
    h_atom_result = test_scf_hydrogen_atom()
    
    # Test SCF for H2 molecule
    h2_result = test_scf_h2_molecule()
    
    # Test SCF for positronium
    positronium_result = test_scf_positronium()
    
    print("\nAll SCF tests completed.")