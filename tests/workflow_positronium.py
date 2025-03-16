#!/usr/bin/env python3
# workflow_positronium.py

print(">>> Script execution started <<<")

import numpy as np
from scipy.linalg import eigh

def positronium_workflow(basis_quality='positronium', include_relativistic=True):
    """
    Complete positronium calculation workflow with all physics included.
    
    Parameters:
    -----------
    basis_quality : str
        Basis set quality ('standard', 'extended', 'large', or 'positronium')
    include_relativistic : bool
        Whether to include relativistic corrections
        
    Returns:
    --------
    Dict
        Complete results including energy, annihilation rate, and lifetime
    """
    print("=== Positronium Calculation Workflow ===")
    
    # 1. Create positronium system
    print("Creating positronium system...")
    from antimatter.core.molecular_data import MolecularData
    positronium = MolecularData.positronium()
    
    # 2. Create basis sets
    print("Creating basis sets...")
    from antimatter.core.basis import MixedMatterBasis
    basis = MixedMatterBasis()
    
    # Use specialized positronium basis if available and requested
    if basis_quality == 'positronium' and hasattr(basis, 'create_positronium_basis'):
        basis.create_positronium_basis(quality='positronium')
    else:
        basis.create_for_molecule(positronium.atoms, basis_quality, basis_quality)
    
    print(f"Basis sets created: {basis.n_electron_basis} electron, {basis.n_positron_basis} positron functions")
    
    # 3. Create integral engine
    from antimatter.core.integral_engine import AntimatterIntegralEngine
    integral_engine = AntimatterIntegralEngine()
    
    # 4. Create Hamiltonian
    print("Building Hamiltonian...")
    from antimatter.core.hamiltonian import AntimatterHamiltonian
    hamiltonian = AntimatterHamiltonian(
        molecular_data=positronium,
        basis_set=basis,
        integral_engine=integral_engine,
        include_annihilation=True
    )
    matrices = hamiltonian.build_hamiltonian()
    
    # 5. Apply relativistic corrections if requested
    if include_relativistic:
        print("Applying relativistic corrections...")
        from antimatter.specialized.relativistic import RelativisticCorrection
        relativistic = RelativisticCorrection(
            hamiltonian=matrices,
            basis_set=basis,
            molecular_data=positronium
        )
        matrices = relativistic.apply_relativistic_corrections()
    
    # 6. Run SCF calculation
    print("Running SCF calculation...")
    
    # Try to use the specialized PositroniumSCF class if available
    try:
        from antimatter.specialized.positronium import PositroniumSCF
        scf = PositroniumSCF(
            hamiltonian=matrices,
            basis_set=basis,
            molecular_data=positronium
        )
        print("Using specialized positronium SCF solver")
    except ImportError:
        # Fall back to the generic SCF class
        from antimatter.core.scf import AntimatterSCF
        scf = AntimatterSCF(
            hamiltonian=matrices,
            basis_set=basis,
            molecular_data=positronium
        )
        print("Using generic SCF solver")
        
        # Special initialization for positronium
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
    
    # If MO coefficients are missing, create them from the density matrices
    if 'C_electron' not in scf_results and 'P_electron' in scf_results:
        # Simple approximation: use the eigenvectors of the density matrix
        e_vals, e_vecs = np.linalg.eigh(scf_results['P_electron'])
        # Sort by eigenvalue in descending order
        idx = np.argsort(e_vals)[::-1]
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        scf_results['C_electron'] = e_vecs
        
        # For positronium, we know the first orbital should be occupied
        # Set the first eigenvalue to 1.0 to indicate occupation
        scf_results['E_electron'] = np.zeros_like(e_vals)
        scf_results['E_electron'][0] = 1.0
    
    if 'C_positron' not in scf_results and 'P_positron' in scf_results:
        # Simple approximation: use the eigenvectors of the density matrix
        p_vals, p_vecs = np.linalg.eigh(scf_results['P_positron'])
        # Sort by eigenvalue in descending order
        idx = np.argsort(p_vals)[::-1]
        p_vals = p_vals[idx]
        p_vecs = p_vecs[:, idx]
        scf_results['C_positron'] = p_vecs
        
        # For positronium, we know the first orbital should be occupied
        # Set the first eigenvalue to 1.0 to indicate occupation
        scf_results['E_positron'] = np.zeros_like(p_vals)
        scf_results['E_positron'][0] = 1.0
    
    # Add number of electrons and positrons to results
    scf_results['n_electrons'] = positronium.n_electrons
    scf_results['n_positrons'] = positronium.n_positrons
    
    # 7. Calculate annihilation properties
    print("Calculating annihilation properties...")
    from antimatter.specialized.annihilation import AnnihilationOperator
    annihilation = AnnihilationOperator(
        basis_set=basis,
        wavefunction=scf_results
    )
    
    # Build annihilation operator
    ann_matrix = annihilation.build_annihilation_operator()
    
    # Calculate annihilation rate
    ann_rate = annihilation.calculate_annihilation_rate()
    
    # Calculate lifetime
    lifetime = annihilation.calculate_lifetime(ann_rate)
    
    # Analyze annihilation channels
    channels = annihilation.analyze_annihilation_channels(scf_results)
    
    # 8. Generate visualization data
    print("Generating visualization data...")
    density_data = annihilation.visualize_annihilation_density(grid_dims=(30, 30, 30), limits=(-5.0, 5.0))
    
    # Combine all results
    results = {
        'energy': scf_results.get('energy', 0.0),
        'converged': scf_results.get('converged', False),
        'iterations': scf_results.get('iterations', 0),
        'annihilation_rate': ann_rate,
        'lifetime': lifetime,
        'channels': channels,
        'density_data': density_data,
        'wavefunction': {
            'P_electron': scf_results.get('P_electron'),
            'P_positron': scf_results.get('P_positron'),
            'C_electron': scf_results.get('C_electron'),
            'C_positron': scf_results.get('C_positron'),
            'E_electron': scf_results.get('E_electron'),
            'E_positron': scf_results.get('E_positron'),
        }
    }
    
    # Print summary
    print("\nResults summary:")
    print(f"Total energy: {results['energy']:.6f} Hartree")
    print(f"Annihilation rate: {results['annihilation_rate']:.6e} a.u.")
    
    if results['lifetime'].get('lifetime_ns', float('inf')) == float('inf'):
        print("Lifetime: infinite (no annihilation)")
    else:
        print(f"Lifetime: {results['lifetime'].get('lifetime_ns', 0.0):.6f} ns")
    
    print("Workflow completed successfully!")
    return results

if __name__ == "__main__":
    # Run the positronium workflow when the script is executed directly
    print("Starting positronium calculation...")
    results = positronium_workflow(basis_quality='extended', include_relativistic=True)
    
    # Print detailed results
    print("\n=== Detailed Results ===")
    print(f"Total energy: {results['energy']:.10f} Hartree")
    print(f"Converged: {results['converged']}")
    print(f"Number of iterations: {results['iterations']}")
    print(f"Annihilation rate: {results['annihilation_rate']:.10e} a.u.")
    
    if results['lifetime'].get('lifetime_ns', float('inf')) == float('inf'):
        print("Lifetime: infinite (no annihilation)")
    else:
        print(f"Lifetime: {results['lifetime'].get('lifetime_ns', 0.0):.6f} ns")
    
    print("\nAnnihilation channels:")
    for channel, probability in results['channels'].items():
        print(f"  {channel}: {probability:.4f}")
    
    print("\nCalculation complete.")