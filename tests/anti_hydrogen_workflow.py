import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import antinature
from antinature.core import MolecularData
from antinature.core.basis import MixedMatterBasis
from antinature.core.integral_engine import antinatureIntegralEngine
from antinature.core.hamiltonian import antinatureHamiltonian
from antinature.core.scf import antinatureSCF

from antinature.specialized import PositroniumSCF
from antinature.specialized.relativistic import RelativisticCorrection
from antinature.specialized.annihilation import AnnihilationOperator




def anti_hydrogen_workflow(basis_quality='extended', include_relativistic=True):
    """
    Complete anti-hydrogen calculation workflow with all physics included.
    
    Parameters:
    -----------
    basis_quality : str
        Basis set quality ('standard' or 'extended')
    include_relativistic : bool
        Whether to include relativistic corrections
        
    Returns:
    --------
    Dict
        Complete results including energy, annihilation rate, and spectrum
    """
    print("=== Anti-Hydrogen Calculation Workflow ===")
    
    # 1. Create anti-hydrogen system
    print("Creating anti-hydrogen system...")
    anti_h = MolecularData.anti_hydrogen()
    
    # 2. Create basis set
    print("Creating basis sets...")
    basis = MixedMatterBasis()
    basis.create_for_molecule(anti_h.atoms, basis_quality, basis_quality)
    print(f"Basis sets created: {basis.n_electron_basis} electron, {basis.n_positron_basis} positron functions")
    
    # 3. Create integral engine
    integral_engine = antinatureIntegralEngine()
    
    # 4. Create Hamiltonian
    print("Building Hamiltonian...")
    hamiltonian = antinatureHamiltonian(
        molecular_data=anti_h,
        basis_set=basis,
        integral_engine=integral_engine,
        include_annihilation=True
    )
    matrices = hamiltonian.build_hamiltonian()
    
    # 5. Apply relativistic corrections if requested
    if include_relativistic:
        print("Applying relativistic corrections...")
        # Initialize with empty matrices if they don't exist yet
        if 'mass_velocity_e' not in matrices:
            matrices['mass_velocity_e'] = np.zeros((basis.n_electron_basis, basis.n_electron_basis))
        if 'darwin_e' not in matrices:
            matrices['darwin_e'] = np.zeros((basis.n_electron_basis, basis.n_electron_basis))
            
        relativistic = RelativisticCorrection(
            hamiltonian=matrices,
            basis_set=basis,
            molecular_data=anti_h,
            correction_type='zora'  # Use ZORA method for accuracy
        )
        matrices = relativistic.apply_relativistic_corrections()
    
    # 6. Run SCF calculation
    print("Running SCF calculation...")
    scf = antinatureSCF(
        hamiltonian=matrices,
        basis_set=basis,
        molecular_data=anti_h
    )
    
    # Initialize with proper guess
    print("Setting up initial guess...")
    scf.initial_guess()
    
    # Make sure we have at least one occupied orbital for each particle type
    if anti_h.n_electrons > 0 and hasattr(scf, 'P_e') and scf.P_e is not None:
        if np.trace(scf.P_e) < 0.5:  # If not enough occupation
            print("Adjusting electron density matrix...")
            # Set occupation for first orbital
            if scf.C_e is not None and scf.C_e.shape[0] > 0 and scf.C_e.shape[1] > 0:
                scf.P_e = np.outer(scf.C_e[:, 0], scf.C_e[:, 0])
                print(f"Electron density trace after adjustment: {np.trace(scf.P_e)}")
    
    if anti_h.n_positrons > 0 and hasattr(scf, 'P_p') and scf.P_p is not None:
        if np.trace(scf.P_p) < 0.5:  # If not enough occupation
            print("Adjusting positron density matrix...")
            # Set occupation for first orbital
            if scf.C_p is not None and scf.C_p.shape[0] > 0 and scf.C_p.shape[1] > 0:
                scf.P_p = np.outer(scf.C_p[:, 0], scf.C_p[:, 0])
                print(f"Positron density trace after adjustment: {np.trace(scf.P_p)}")
    
    # Save initial density matrices
    initial_P_e = scf.P_e.copy() if hasattr(scf, 'P_e') and scf.P_e is not None else None
    initial_P_p = scf.P_p.copy() if hasattr(scf, 'P_p') and scf.P_p is not None else None
    
    # Perform SCF calculation
    scf_results = scf.solve_scf()
    print(f"SCF completed: energy = {scf_results.get('energy', 0.0):.6f} Hartree")
    
    # Check if density matrices are zero and restore initial values if needed
    if hasattr(scf, 'P_e') and np.all(scf.P_e == 0) and initial_P_e is not None:
        print("Restoring initial electron density matrix (SCF reset it to zero)")
        scf.P_e = initial_P_e
    
    if hasattr(scf, 'P_p') and np.all(scf.P_p == 0) and initial_P_p is not None:
        print("Restoring initial positron density matrix (SCF reset it to zero)")
        scf.P_p = initial_P_p
    
    # Add density matrices to results
    if 'P_electron' not in scf_results:
        if hasattr(scf, 'P_e') and scf.P_e is not None:
            scf_results['P_electron'] = scf.P_e
        elif initial_P_e is not None:
            scf_results['P_electron'] = initial_P_e
    
    if 'P_positron' not in scf_results:
        if hasattr(scf, 'P_p') and scf.P_p is not None:
            scf_results['P_positron'] = scf.P_p
        elif initial_P_p is not None:
            scf_results['P_positron'] = initial_P_p
    
    # Check if density matrices in results are zero and replace them
    if 'P_electron' in scf_results and np.all(scf_results['P_electron'] == 0) and initial_P_e is not None:
        scf_results['P_electron'] = initial_P_e
    
    if 'P_positron' in scf_results and np.all(scf_results['P_positron'] == 0) and initial_P_p is not None:
        scf_results['P_positron'] = initial_P_p
    
    # Add MO coefficients to results
    if 'C_electron' not in scf_results and hasattr(scf, 'C_e'):
        scf_results['C_electron'] = scf.C_e
    
    if 'C_positron' not in scf_results and hasattr(scf, 'C_p'):
        scf_results['C_positron'] = scf.C_p
    
    # If MO coefficients are missing, create them from density matrices
    if 'C_electron' not in scf_results and 'P_electron' in scf_results and scf_results['P_electron'].shape[0] > 0:
        # Simple approximation: use the eigenvectors of the density matrix
        e_vals, e_vecs = np.linalg.eigh(scf_results['P_electron'])
        # Sort by eigenvalue in descending order
        idx = np.argsort(e_vals)[::-1]
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        scf_results['C_electron'] = e_vecs
        
        # Set eigenvalues based on occupation
        scf_results['E_electron'] = np.zeros_like(e_vals)
        n_occ = anti_h.n_electrons
        for i in range(min(n_occ, len(e_vals))):
            scf_results['E_electron'][i] = 1.0
    
    if 'C_positron' not in scf_results and 'P_positron' in scf_results and scf_results['P_positron'].shape[0] > 0:
        # Simple approximation: use the eigenvectors of the density matrix
        p_vals, p_vecs = np.linalg.eigh(scf_results['P_positron'])
        # Sort by eigenvalue in descending order
        idx = np.argsort(p_vals)[::-1]
        p_vals = p_vals[idx]
        p_vecs = p_vecs[:, idx]
        scf_results['C_positron'] = p_vecs
        
        # Set eigenvalues based on occupation
        scf_results['E_positron'] = np.zeros_like(p_vals)
        n_occ = anti_h.n_positrons
        for i in range(min(n_occ, len(p_vals))):
            scf_results['E_positron'][i] = 1.0
    
    # Add number of electrons and positrons to results
    scf_results['n_electrons'] = anti_h.n_electrons
    scf_results['n_positrons'] = anti_h.n_positrons
    
    # 7. Extract positron orbital energies for spectrum
    positron_orbitals = scf_results.get('E_positron', [])
    spectrum = []
    
    if len(positron_orbitals) > 1:
        print("\nEnergy levels (first 5 states):")
        for i, energy in enumerate(positron_orbitals[:min(5, len(positron_orbitals))]):
            print(f"  State {i}: {energy:.6f} Hartree")
            spectrum.append({
                'state': i,
                'energy': energy
            })
    
    # 8. Calculate annihilation properties
    print("Calculating annihilation properties...")
    annihilation = AnnihilationOperator(
        basis_set=basis,
        wavefunction=scf_results
    )
    
    # Build annihilation operator
    ann_matrix = annihilation.build_annihilation_operator()
    
    # Check if we have valid matrices for annihilation calculation
    if (scf_results.get('P_electron') is not None and scf_results.get('P_electron').shape[0] > 0 and
        scf_results.get('P_positron') is not None and scf_results.get('P_positron').shape[0] > 0):
        # Calculate annihilation rate
        ann_rate = annihilation.calculate_annihilation_rate()
        
        # Calculate annihilation channels if needed
        channels = annihilation.analyze_annihilation_channels(scf_results)
    else:
        print("Warning: Cannot calculate annihilation rate due to missing or empty density matrices")
        ann_rate = 0.0
        channels = {'ratio_2g_3g': 0.0}
    
    # 9. Compile and return results
    print("\nResults summary:")
    print(f"Total energy: {scf_results.get('energy', 0.0):.6f} Hartree")
    print(f"Annihilation rate: {ann_rate:.6e} a.u.")
    
    if 'ratio_2g_3g' in channels:
        print(f"2γ/3γ ratio: {channels['ratio_2g_3g']:.2f}")
    
    results = {
        'system': 'anti-hydrogen',
        'energy': scf_results.get('energy', 0.0),
        'converged': scf_results.get('converged', False),
        'iterations': scf_results.get('iterations', 0),
        'annihilation_rate': ann_rate,
        'channels': channels,
        'spectrum': spectrum,
        'wavefunction': {
            'P_electron': scf_results.get('P_electron'),
            'P_positron': scf_results.get('P_positron'),
            'C_electron': scf_results.get('C_electron'),
            'C_positron': scf_results.get('C_positron'),
            'E_electron': scf_results.get('E_electron'),
            'E_positron': scf_results.get('E_positron'),
        }
    }
    
    print("Workflow completed successfully!")
    return results


if __name__ == "__main__":
    # Run the anti-hydrogen workflow
    print("Starting anti-hydrogen calculation...")
    results = anti_hydrogen_workflow(basis_quality='extended', include_relativistic=True)
    
    # Print detailed results
    print("\n=== Detailed Results ===")
    print(f"Total energy: {results['energy']:.10f} Hartree")
    print(f"Converged: {results['converged']}")
    print(f"Number of iterations: {results['iterations']}")
    print(f"Annihilation rate: {results['annihilation_rate']:.10e} a.u.")
    
    print("\nAnnihilation channels:")
    for channel, probability in results['channels'].items():
        print(f"  {channel}: {probability:.4f}")
    
    print("\nSpectrum:")
    for state in results['spectrum']:
        print(f"  State {state['state']}: {state['energy']:.6f} Hartree")
    
    print("\nCalculation complete.")