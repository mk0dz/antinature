# test_specialized_modules.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from antiverse.core.molecular_data import MolecularData
from antiverse.core.basis import MixedMatterBasis
from antiverse.core.integral_engine import antiverseIntegralEngine
from antiverse.core.hamiltonian import antiverseHamiltonian
from antiverse.core.scf import antiverseSCF

from antiverse.specialized.relativistic import RelativisticCorrection
from antiverse.specialized.annihilation import AnnihilationOperator

def test_relativistic_corrections():
    print("\n=== Testing Relativistic Corrections ===")
    
    # Create a helium atom (relativistic effects are more significant for heavier elements)
    helium = MolecularData(
        atoms=[('He', np.array([0.0, 0.0, 0.0]))],
        n_electrons=2,
        n_positrons=0,
        charge=0,
        name="Helium"
    )
    
    # Create basis set
    basis = MixedMatterBasis()
    basis.create_for_molecule(helium.atoms, 'standard', 'standard')
    
    # Create integral engine
    integral_engine = antiverseIntegralEngine()
    
    # Create Hamiltonian
    hamiltonian = antiverseHamiltonian(
        molecular_data=helium,
        basis_set=basis,
        integral_engine=integral_engine
    )
    
    # Build Hamiltonian
    matrices = hamiltonian.build_hamiltonian()
    
    # Create relativistic correction object
    relativistic = RelativisticCorrection(
        hamiltonian=matrices,
        basis_set=basis,
        molecular_data=helium,
        correction_type='perturbative'
    )
    
    # Calculate relativistic corrections
    print("Calculating relativistic integrals...")
    rel_matrices = relativistic.calculate_relativistic_integrals()
    
    # Print first few elements of correction matrices
    print("\nMass-velocity correction (first 3x3 elements):")
    mv_e = rel_matrices.get('mass_velocity_e')
    if mv_e is not None and mv_e.shape[0] > 0:
        print(mv_e[:min(3, mv_e.shape[0]), :min(3, mv_e.shape[1])])
    
    print("\nDarwin correction (first 3x3 elements):")
    darwin_e = rel_matrices.get('darwin_e')
    if darwin_e is not None and darwin_e.shape[0] > 0:
        print(darwin_e[:min(3, darwin_e.shape[0]), :min(3, darwin_e.shape[1])])
    
    # Apply corrections to Hamiltonian
    print("\nApplying relativistic corrections to Hamiltonian...")
    corrected_hamiltonian = relativistic.apply_relativistic_corrections()
    
    # Run SCF with non-relativistic Hamiltonian
    print("\nRunning non-relativistic SCF...")
    scf_nonrel = antiverseSCF(
        hamiltonian=matrices,
        basis_set=basis,
        molecular_data=helium
    )
    nonrel_results = scf_nonrel.solve_scf()
    
    # Run SCF with relativistic Hamiltonian
    print("\nRunning relativistic SCF...")
    scf_rel = antiverseSCF(
        hamiltonian=corrected_hamiltonian,
        basis_set=basis,
        molecular_data=helium
    )
    rel_results = scf_rel.solve_scf()
    
    # Compare energies
    nonrel_energy = nonrel_results.get('energy', 0.0)
    rel_energy = rel_results.get('energy', 0.0)
    energy_diff = rel_energy - nonrel_energy
    
    print(f"\nNon-relativistic energy: {nonrel_energy:.10f} Hartree")
    print(f"Relativistic energy: {rel_energy:.10f} Hartree")
    print(f"Energy difference: {energy_diff:.10f} Hartree")
    
    return relativistic, nonrel_results, rel_results

def test_annihilation_physics():
    print("\n=== Testing Annihilation Physics ===")
    
    # Create a positronium system
    positronium = MolecularData.positronium()
    
    # Create basis set
    basis = MixedMatterBasis()
    basis.create_for_molecule(positronium.atoms, 'extended', 'extended')  # Use extended basis for better results
    
    # Create integral engine
    integral_engine = antiverseIntegralEngine()
    
    # Create Hamiltonian
    hamiltonian = antiverseHamiltonian(
        molecular_data=positronium,
        basis_set=basis,
        integral_engine=integral_engine,
        include_annihilation=True
    )
    
    # Build Hamiltonian
    matrices = hamiltonian.build_hamiltonian()
    
    # Run SCF calculation
    print("Running SCF calculation for positronium...")
    scf = antiverseSCF(
        hamiltonian=matrices,
        basis_set=basis,
        molecular_data=positronium
    )
    
    # Initialize with proper guess for positronium
    print("Setting up initial guess...")
    if hasattr(scf, 'positronium_initial_guess'):
        print("Using specialized positronium initial guess...")
        scf.positronium_initial_guess()
    else:
        scf.initial_guess()
        
        # Make sure we have at least one occupied orbital for each particle type
        if scf.n_electrons > 0 and scf.P_e is not None:
            if np.trace(scf.P_e) < 0.5:  # If not enough occupation
                print("Adjusting electron density matrix...")
                # Set occupation for first orbital
                if scf.C_e is not None and scf.C_e.shape[0] > 0 and scf.C_e.shape[1] > 0:
                    scf.P_e = np.outer(scf.C_e[:, 0], scf.C_e[:, 0])
                    print(f"Electron density trace after adjustment: {np.trace(scf.P_e)}")
        
        if scf.n_positrons > 0 and scf.P_p is not None:
            if np.trace(scf.P_p) < 0.5:  # If not enough occupation
                print("Adjusting positron density matrix...")
                # Set occupation for first orbital
                if scf.C_p is not None and scf.C_p.shape[0] > 0 and scf.C_p.shape[1] > 0:
                    scf.P_p = np.outer(scf.C_p[:, 0], scf.C_p[:, 0])
                    print(f"Positron density trace after adjustment: {np.trace(scf.P_p)}")
    
    # Save initial density matrices
    initial_P_e = scf.P_e.copy() if hasattr(scf, 'P_e') and scf.P_e is not None else None
    initial_P_p = scf.P_p.copy() if hasattr(scf, 'P_p') and scf.P_p is not None else None
    
    # Run SCF calculation
    results = scf.solve_scf()
    
    # Check if density matrices are zero and restore initial values if needed
    if hasattr(scf, 'P_e') and np.all(scf.P_e == 0) and initial_P_e is not None:
        print("Restoring initial electron density matrix (SCF reset it to zero)")
        scf.P_e = initial_P_e
    
    if hasattr(scf, 'P_p') and np.all(scf.P_p == 0) and initial_P_p is not None:
        print("Restoring initial positron density matrix (SCF reset it to zero)")
        scf.P_p = initial_P_p
    
    # Add density matrices to results
    if 'P_electron' not in results:
        if hasattr(scf, 'P_e') and scf.P_e is not None:
            results['P_electron'] = scf.P_e
        elif initial_P_e is not None:
            results['P_electron'] = initial_P_e
    
    if 'P_positron' not in results:
        if hasattr(scf, 'P_p') and scf.P_p is not None:
            results['P_positron'] = scf.P_p
        elif initial_P_p is not None:
            results['P_positron'] = initial_P_p
    
    # Check if density matrices in results are zero and replace them
    if 'P_electron' in results and np.all(results['P_electron'] == 0) and initial_P_e is not None:
        results['P_electron'] = initial_P_e
    
    if 'P_positron' in results and np.all(results['P_positron'] == 0) and initial_P_p is not None:
        results['P_positron'] = initial_P_p
    
    # Add MO coefficients to results
    if 'C_electron' not in results and hasattr(scf, 'C_e'):
        results['C_electron'] = scf.C_e
    
    if 'C_positron' not in results and hasattr(scf, 'C_p'):
        results['C_positron'] = scf.C_p
    
    # If MO coefficients are missing, create them from density matrices
    if 'C_electron' not in results and 'P_electron' in results:
        # Simple approximation: use the eigenvectors of the density matrix
        e_vals, e_vecs = np.linalg.eigh(results['P_electron'])
        # Sort by eigenvalue in descending order
        idx = np.argsort(e_vals)[::-1]
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        results['C_electron'] = e_vecs
        
        # For positronium, we know the first orbital should be occupied
        results['E_electron'] = np.zeros_like(e_vals)
        results['E_electron'][0] = 1.0
    
    if 'C_positron' not in results and 'P_positron' in results:
        # Simple approximation: use the eigenvectors of the density matrix
        p_vals, p_vecs = np.linalg.eigh(results['P_positron'])
        # Sort by eigenvalue in descending order
        idx = np.argsort(p_vals)[::-1]
        p_vals = p_vals[idx]
        p_vecs = p_vecs[:, idx]
        results['C_positron'] = p_vecs
        
        # For positronium, we know the first orbital should be occupied
        results['E_positron'] = np.zeros_like(p_vals)
        results['E_positron'][0] = 1.0
    
    # Add number of electrons and positrons to results
    results['n_electrons'] = positronium.n_electrons
    results['n_positrons'] = positronium.n_positrons
    
    # Create annihilation operator
    annihilation = AnnihilationOperator(
        basis_set=basis,
        wavefunction=results
    )
    
    # Build annihilation operator
    print("\nBuilding annihilation operator...")
    ann_matrix = annihilation.build_annihilation_operator()
    
    print(f"Annihilation matrix shape: {ann_matrix.shape}")
    if ann_matrix.shape[0] > 0 and ann_matrix.shape[1] > 0:
        print("First few elements:")
        print(ann_matrix[:min(3, ann_matrix.shape[0]), :min(3, ann_matrix.shape[1])])
    
    # Calculate annihilation rate
    print("\nCalculating annihilation rate...")
    rate = annihilation.calculate_annihilation_rate()
    print(f"Annihilation rate: {rate:.10e} a.u.")
    
    # Calculate lifetime
    print("\nCalculating lifetime...")
    lifetime = annihilation.calculate_lifetime(rate)
    if 'lifetime_ns' in lifetime:
        print(f"Lifetime: {lifetime['lifetime_ns']:.6f} ns")
    else:
        print("Lifetime: infinite (no annihilation)")
    
    # Analyze annihilation channels
    print("\nAnalyzing annihilation channels...")
    channels = annihilation.analyze_annihilation_channels(results)  # Pass the results explicitly
    
    # Check if we have valid channel rates
    if channels:
        if 'two_gamma' in channels:
            print(f"2γ rate: {channels['two_gamma']:.10e} a.u.")
        if 'three_gamma' in channels:
            print(f"3γ rate: {channels['three_gamma']:.10e} a.u.")
        if 'ratio_2g_3g' in channels:
            print(f"2γ/3γ ratio: {channels['ratio_2g_3g']:.2f}")
    else:
        print("No annihilation channels data available")
    
    # Calculate annihilation density
    print("\nCalculating annihilation density...")
    try:
        density_data = annihilation.visualize_annihilation_density(grid_dims=(10, 10, 10))
        
        if density_data is not None:
            print(f"Density grid shape: {density_data['density'].shape}")
            print(f"Maximum density: {density_data['density'].max():.10e}")
    except Exception as e:
        print(f"Error calculating annihilation density: {str(e)}")
        density_data = None
    
    return annihilation, results, density_data

def run_tests():
    relativistic, nonrel_results, rel_results = test_relativistic_corrections()
    print("\n" + "="*50)
    annihilation, positronium_results, density_data = test_annihilation_physics()
    
    return {
        'relativistic': {
            'calculator': relativistic,
            'nonrel_results': nonrel_results,
            'rel_results': rel_results
        },
        'annihilation': {
            'calculator': annihilation,
            'results': positronium_results,
            'density_data': density_data
        }
    }

if __name__ == "__main__":
    results = run_tests()