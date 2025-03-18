#!/usr/bin/env python
# positronium_example.py - Stable example of positronium calculations
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from scipy import linalg

# Add the parent directory to the path to find the antinature module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from antinature.core.molecular_data import MolecularData
from antinature.core.basis import MixedMatterBasis
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.scf import AntinatureSCF
from antinature.core.correlation import AntinatureCorrelation
from antinature.specialized import PositroniumSCF

def run_positronium_standard():
    """
    Run a stable positronium calculation with standard basis set.
    
    This version uses the standard quality basis set which has fewer
    functions and is less likely to have linear dependency issues.
    """
    print("\n=== Positronium with Standard Basis ===\n")
    
    # Create positronium system
    positronium = MolecularData.positronium()
    
    # Print system information
    print(f"System: Positronium")
    print(f"Number of electrons: {positronium.n_electrons}")
    print(f"Number of positrons: {positronium.n_positrons}")
    
    # Create specialized basis set for positronium with standard quality
    basis = MixedMatterBasis()
    basis.create_positronium_basis(quality='standard')  # Use standard instead of extended
    
    # Print basis information
    e_basis_info = basis.electron_basis.get_function_types() if basis.electron_basis else {}
    p_basis_info = basis.positron_basis.get_function_types() if basis.positron_basis else {}
    
    print("\nBasis set information:")
    print(f"Electron basis functions: {len(basis.electron_basis) if basis.electron_basis else 0}")
    print(f"Electron function types: {e_basis_info}")
    print(f"Positron basis functions: {len(basis.positron_basis) if basis.positron_basis else 0}")
    print(f"Positron function types: {p_basis_info}")
    
    # Set up integral engine
    integral_engine = AntinatureIntegralEngine(
        use_analytical=True
    )
    
    # Build Hamiltonian
    print("\nBuilding Hamiltonian...")
    t_start = time.time()
    hamiltonian = AntinatureHamiltonian(
        molecular_data=positronium,
        basis_set=basis,
        integral_engine=integral_engine,
        include_annihilation=True
    )
    hamiltonian_matrices = hamiltonian.build_hamiltonian()
    t_hamiltonian = time.time() - t_start
    print(f"Hamiltonian built in {t_hamiltonian:.3f} seconds")
    
    # Run specialized positronium SCF calculation
    print("\nStarting SCF calculation...")
    t_start = time.time()
    try:
        scf_solver = PositroniumSCF(
            hamiltonian=hamiltonian_matrices,
            basis_set=basis,
            molecular_data=positronium,
            max_iterations=100,
            convergence_threshold=1e-6
        )
        results = scf_solver.solve_scf()
        t_scf = time.time() - t_start
        print(f"SCF completed in {t_scf:.3f} seconds")
        
        # Print results
        print(f"\nPositronium ground state energy: {results['energy']:.10f} Hartree")
        print(f"Iterations: {results.get('iterations', 'N/A')}")
        print(f"Converged: {results.get('converged', 'N/A')}")
        
        # Try correlation calculations
        print("\nStarting correlation calculations...")
        t_start = time.time()
        
        # Convert SCF results arrays from lists to NumPy arrays if necessary
        for key in ['C_electron', 'C_positron', 'E_electron', 'E_positron', 'P_electron', 'P_positron']:
            if key in results and isinstance(results[key], list):
                results[key] = np.array(results[key])
        
        # Create correlation object
        corr = AntinatureCorrelation(
            scf_result=results,
            hamiltonian=hamiltonian_matrices,
            basis_set=basis,
            print_level=2
        )
        
        # Calculate MP2 energy
        print("\nCalculating MP2 energy...")
        mp2_energy = corr.mp2_energy(include_electron_positron=True)
        
        # Print MP2 results
        print("\nMP2 Results:")
        print(f"  Correlation energy: {mp2_energy:.10f} Hartree")
        print(f"  Total energy: {results['energy'] + mp2_energy:.10f} Hartree")
        
        # Try to calculate annihilation rate
        try:
            print("\nCalculating positron annihilation rate...")
            annihilation_rate = corr.calculate_annihilation_rate()
            print(f"Positron annihilation rate: {annihilation_rate:.6e} s^-1")
        except Exception as e:
            print(f"Annihilation rate calculation error: {e}")
        
        return {
            "energy": results['energy'],
            "mp2_energy": mp2_energy,
            "total_energy": results['energy'] + mp2_energy,
            "annihilation_rate": annihilation_rate if 'annihilation_rate' in locals() else None
        }
    
    except Exception as e:
        print(f"SCF calculation error: {e}")
        return None

def run_positronium_with_conditioning():
    """
    Run a positronium calculation with extended basis but apply numerical conditioning.
    
    This version adds a custom implementation of overlap matrix conditioning to 
    improve numerical stability with extended basis sets.
    """
    print("\n=== Positronium with Extended Basis (Conditioned) ===\n")
    
    # Create positronium system
    positronium = MolecularData.positronium()
    
    # Create specialized basis set for positronium
    basis = MixedMatterBasis()
    basis.create_positronium_basis(quality='extended')
    
    # Print basis information
    e_basis_info = basis.electron_basis.get_function_types() if basis.electron_basis else {}
    p_basis_info = basis.positron_basis.get_function_types() if basis.positron_basis else {}
    
    print("\nBasis set information:")
    print(f"Electron basis functions: {len(basis.electron_basis) if basis.electron_basis else 0}")
    print(f"Electron function types: {e_basis_info}")
    print(f"Positron basis functions: {len(basis.positron_basis) if basis.positron_basis else 0}")
    print(f"Positron function types: {p_basis_info}")
    
    # Set up integral engine
    integral_engine = AntinatureIntegralEngine(
        use_analytical=True
    )
    
    # Build Hamiltonian
    print("\nBuilding Hamiltonian...")
    t_start = time.time()
    hamiltonian = AntinatureHamiltonian(
        molecular_data=positronium,
        basis_set=basis,
        integral_engine=integral_engine,
        include_annihilation=True
    )
    hamiltonian_matrices = hamiltonian.build_hamiltonian()
    t_hamiltonian = time.time() - t_start
    print(f"Hamiltonian built in {t_hamiltonian:.3f} seconds")
    
    # Get overlap matrices
    S_e = hamiltonian_matrices.get('S_e')
    S_p = hamiltonian_matrices.get('S_p')
    
    # Apply conditioning to overlap matrices if they exist
    if S_e is not None and len(S_e) > 0:
        print("\nConditioning electron overlap matrix...")
        # Find eigenvalues and eigenvectors
        eigval_e, eigvec_e = linalg.eigh(S_e)
        # Set threshold for eigenvalues (remove small eigenvalues)
        threshold = 1e-10
        # Keep only eigenvalues above threshold
        mask = eigval_e > threshold
        # Print conditioning information
        print(f"Original eigenvalue range: {np.min(eigval_e):.6e} to {np.max(eigval_e):.6e}")
        print(f"Keeping {np.sum(mask)}/{len(eigval_e)} eigenvalues")
        # Create conditioned matrix
        eigval_e_cond = eigval_e[mask]
        eigvec_e_cond = eigvec_e[:, mask]
        # Rebuild overlap matrix
        S_e_cond = eigvec_e_cond @ np.diag(eigval_e_cond) @ eigvec_e_cond.T
        # Replace original matrix
        hamiltonian_matrices['S_e'] = S_e_cond
    
    if S_p is not None and len(S_p) > 0:
        print("\nConditioning positron overlap matrix...")
        # Find eigenvalues and eigenvectors
        eigval_p, eigvec_p = linalg.eigh(S_p)
        # Set threshold for eigenvalues (remove small eigenvalues)
        threshold = 1e-10
        # Keep only eigenvalues above threshold
        mask = eigval_p > threshold
        # Print conditioning information
        print(f"Original eigenvalue range: {np.min(eigval_p):.6e} to {np.max(eigval_p):.6e}")
        print(f"Keeping {np.sum(mask)}/{len(eigval_p)} eigenvalues")
        # Create conditioned matrix
        eigval_p_cond = eigval_p[mask]
        eigvec_p_cond = eigvec_p[:, mask]
        # Rebuild overlap matrix
        S_p_cond = eigvec_p_cond @ np.diag(eigval_p_cond) @ eigvec_p_cond.T
        # Replace original matrix
        hamiltonian_matrices['S_p'] = S_p_cond
    
    # Run specialized positronium SCF calculation
    print("\nStarting SCF calculation with conditioned matrices...")
    t_start = time.time()
    try:
        scf_solver = PositroniumSCF(
            hamiltonian=hamiltonian_matrices,
            basis_set=basis,
            molecular_data=positronium,
            max_iterations=100,
            convergence_threshold=1e-6
        )
        # Add a custom solver with more robust numerical stability
        # This is a simple example; in practice, you might need more complex stabilization
        scf_solver.use_robust_solver = True
        results = scf_solver.solve_scf()
        t_scf = time.time() - t_start
        print(f"SCF completed in {t_scf:.3f} seconds")
        
        # Print results
        print(f"\nPositronium ground state energy: {results['energy']:.10f} Hartree")
        print(f"Iterations: {results.get('iterations', 'N/A')}")
        print(f"Converged: {results.get('converged', 'N/A')}")
        
        return {
            "energy": results['energy']
        }
    
    except Exception as e:
        print(f"SCF calculation error (even with conditioning): {e}")
        return None

def run_positronium_analytical():
    """
    Use analytical solution for positronium ground state energy.
    
    For simple positronium systems, analytical solutions are available and
    are more stable than numerical approaches.
    """
    print("\n=== Positronium Analytical Solution ===\n")
    
    # The ground state energy of positronium is -0.25 Hartree
    # (half of the hydrogen atom ground state energy)
    # See: https://en.wikipedia.org/wiki/Positronium
    
    # Rydberg energy in Hartree
    rydberg = 0.5
    
    # Positronium ground state energy is -Ry/2
    positronium_energy = -rydberg / 2.0
    
    # Print result
    print(f"Positronium ground state energy (analytical): {positronium_energy:.10f} Hartree")
    print("Note: This is the exact solution for the ground state of positronium.")
    print("The energy is -Ry/2 = -0.25 Hartree, where Ry = 0.5 Hartree is the Rydberg energy.")
    
    return {
        "energy": positronium_energy
    }

def run_custom_positronium_system():
    """
    Create a custom positronium system with optimized parameters.
    
    This approach uses a carefully selected basis set and parameters designed
    to avoid numerical instabilities while still capturing the physics.
    """
    print("\n=== Custom Positronium System ===\n")
    
    # Create positronium system
    positronium = MolecularData.positronium()
    
    # Create a custom minimal basis designed for stability
    basis = MixedMatterBasis()
    
    # Start with a minimal basis and then customize
    basis.create_positronium_basis(quality='minimal')
    
    # Import the GaussianBasisFunction class
    from antinature.core.basis import GaussianBasisFunction
    
    # Create new electron basis functions with well-conditioned exponents
    electron_funcs = []
    # s-type functions
    for alpha in [0.1, 0.3, 0.9, 2.7]:
        func = GaussianBasisFunction(
            center=np.array([0.0, 0.0, 0.0]),
            angular_momentum=(0, 0, 0),  # s-type
            exponent=alpha
        )
        electron_funcs.append(func)
    
    # p-type functions
    for alpha in [0.2, 0.6, 1.8]:
        for lm in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:  # p-type (px, py, pz)
            func = GaussianBasisFunction(
                center=np.array([0.0, 0.0, 0.0]),
                angular_momentum=lm,  
                exponent=alpha
            )
            electron_funcs.append(func)
    
    # Replace the electron basis functions
    basis.electron_basis.functions = electron_funcs
    
    # Create positron basis functions with well-conditioned exponents
    positron_funcs = []
    # s-type functions
    for alpha in [0.1, 0.3, 0.9, 2.7]:
        func = GaussianBasisFunction(
            center=np.array([0.0, 0.0, 0.0]),
            angular_momentum=(0, 0, 0),  # s-type
            exponent=alpha
        )
        positron_funcs.append(func)
    
    # p-type functions
    for alpha in [0.2, 0.6, 1.8]:
        for lm in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:  # p-type (px, py, pz)
            func = GaussianBasisFunction(
                center=np.array([0.0, 0.0, 0.0]),
                angular_momentum=lm,  
                exponent=alpha
            )
            positron_funcs.append(func)
    
    # Replace the positron basis functions
    basis.positron_basis.functions = positron_funcs
    
    # Print basis information
    e_basis_info = basis.electron_basis.get_function_types() if basis.electron_basis else {}
    p_basis_info = basis.positron_basis.get_function_types() if basis.positron_basis else {}
    
    print("\nBasis set information:")
    print(f"Electron basis functions: {len(basis.electron_basis) if basis.electron_basis else 0}")
    print(f"Electron function types: {e_basis_info}")
    print(f"Positron basis functions: {len(basis.positron_basis) if basis.positron_basis else 0}")
    print(f"Positron function types: {p_basis_info}")
    
    # Set up integral engine with high precision
    integral_engine = AntinatureIntegralEngine(
        use_analytical=True
    )
    
    # Build Hamiltonian
    print("\nBuilding Hamiltonian...")
    t_start = time.time()
    hamiltonian = AntinatureHamiltonian(
        molecular_data=positronium,
        basis_set=basis,
        integral_engine=integral_engine,
        include_annihilation=True,
        include_relativistic=False  # Disable relativistic effects for stability
    )
    hamiltonian_matrices = hamiltonian.build_hamiltonian()
    t_hamiltonian = time.time() - t_start
    print(f"Hamiltonian built in {t_hamiltonian:.3f} seconds")
    
    # Try standard SCF first
    print("\nAttempting regular SCF calculation...")
    try:
        # This will likely fail, but we try it first
        scf = AntinatureSCF(
            hamiltonian=hamiltonian_matrices,
            basis_set=basis,
            molecular_data=positronium,
            max_iterations=100,
            convergence_threshold=1e-6,
            use_diis=False,  # Disable DIIS for stability
            damping_factor=0.7,  # Add heavy damping
            print_level=1
        )
        scf_results = scf.solve_scf()
        print(f"SCF Energy: {scf_results['energy']:.10f} Hartree")
        
    except Exception as e:
        print(f"Regular SCF failed: {e}")
        print("Falling back to specialized positronium solver...")
        
        try:
            # Fall back to positronium-specific solver
            ps_scf = PositroniumSCF(
                hamiltonian=hamiltonian_matrices,
                basis_set=basis,
                molecular_data=positronium,
                max_iterations=100,
                convergence_threshold=1e-6
            )
            ps_results = ps_scf.solve_scf()
            print(f"Specialized SCF Energy: {ps_results['energy']:.10f} Hartree")
            
            # If that succeeded, try correlation
            try:
                # Convert SCF results arrays from lists to NumPy arrays if necessary
                for key in ['C_electron', 'C_positron', 'E_electron', 'E_positron', 'P_electron', 'P_positron']:
                    if key in ps_results and isinstance(ps_results[key], list):
                        ps_results[key] = np.array(ps_results[key])
                
                # Create correlation object
                corr = AntinatureCorrelation(
                    scf_result=ps_results,
                    hamiltonian=hamiltonian_matrices,
                    basis_set=basis,
                    print_level=2
                )
                
                # Calculate MP2 energy
                print("\nCalculating MP2 energy...")
                mp2_energy = corr.mp2_energy(include_electron_positron=True)
                
                # Print MP2 results
                print("\nMP2 Results:")
                print(f"  Correlation energy: {mp2_energy:.10f} Hartree")
                print(f"  Total energy: {ps_results['energy'] + mp2_energy:.10f} Hartree")
                
                return {
                    "energy": ps_results['energy'],
                    "mp2_energy": mp2_energy,
                    "total_energy": ps_results['energy'] + mp2_energy
                }
                
            except Exception as e:
                print(f"Correlation calculation error: {e}")
                return {"energy": ps_results['energy']}
                
        except Exception as e:
            print(f"All SCF attempts failed: {e}")
            print("Falling back to analytical solution...")
            
            # Analytical solution as fallback
            analytical_energy = -0.25  # Exact ground state energy for positronium
            print(f"Analytical positronium energy: {analytical_energy:.10f} Hartree")
            
            return {"energy": analytical_energy, "method": "analytical"}

def run_simplified_positronium():
    """
    Run a simplified positronium calculation with a minimal basis set.
    
    This version uses the minimal quality basis set and a direct approach 
    to maximize the chance of convergence.
    """
    print("\n=== Simplified Positronium ===\n")
    
    # Create positronium system
    positronium = MolecularData.positronium()
    
    # Print system information
    print(f"System: Positronium")
    print(f"Number of electrons: {positronium.n_electrons}")
    print(f"Number of positrons: {positronium.n_positrons}")
    
    # Create a minimal basis set for positronium
    basis = MixedMatterBasis()
    basis.create_positronium_basis(quality='minimal')
    
    # Print basis information
    e_basis_info = basis.electron_basis.get_function_types() if basis.electron_basis else {}
    p_basis_info = basis.positron_basis.get_function_types() if basis.positron_basis else {}
    
    print("\nBasis set information:")
    print(f"Electron basis functions: {len(basis.electron_basis) if basis.electron_basis else 0}")
    print(f"Electron function types: {e_basis_info}")
    print(f"Positron basis functions: {len(basis.positron_basis) if basis.positron_basis else 0}")
    print(f"Positron function types: {p_basis_info}")
    
    # Set up integral engine
    integral_engine = AntinatureIntegralEngine(use_analytical=True)
    
    # Build Hamiltonian
    print("\nBuilding Hamiltonian...")
    t_start = time.time()
    hamiltonian = AntinatureHamiltonian(
        molecular_data=positronium,
        basis_set=basis,
        integral_engine=integral_engine,
        include_annihilation=True
    )
    hamiltonian_matrices = hamiltonian.build_hamiltonian()
    t_hamiltonian = time.time() - t_start
    print(f"Hamiltonian built in {t_hamiltonian:.3f} seconds")
    
    # Try standard SCF first
    print("\nAttempting standard SCF calculation...")
    try:
        # Try regular SCF first
        scf = AntinatureSCF(
            hamiltonian=hamiltonian_matrices,
            basis_set=basis,
            molecular_data=positronium,
            max_iterations=100,
            convergence_threshold=1e-6,
            use_diis=False,  # Turn off DIIS for stability
            damping_factor=0.7,  # Heavy damping for stability
        )
        scf_results = scf.solve_scf()
        
        print(f"SCF Energy: {scf_results['energy']:.10f} Hartree")
        print(f"Iterations: {scf_results.get('iterations', 'N/A')}")
        
        return {
            "energy": scf_results['energy'],
            "method": "standard_scf"
        }
        
    except Exception as e:
        print(f"Standard SCF failed: {e}")
        
        # Try positronium-specific SCF as fallback
        try:
            print("\nFalling back to specialized positronium solver...")
            ps_scf = PositroniumSCF(
                hamiltonian=hamiltonian_matrices,
                basis_set=basis,
                molecular_data=positronium,
                max_iterations=100,
                convergence_threshold=1e-6
            )
            ps_results = ps_scf.solve_scf()
            
            print(f"Specialized SCF Energy: {ps_results['energy']:.10f} Hartree")
            
            return {
                "energy": ps_results['energy'],
                "method": "specialized_scf"
            }
            
        except Exception as e:
            print(f"All SCF methods failed: {e}")
            return None

if __name__ == "__main__":
    print("===== ANTINATURE POSITRONIUM TESTS =====")
    print("This script tests various approaches to positronium calculations")
    print("to demonstrate how to handle numerical instabilities.")
    
    # Create examples directory if it doesn't exist
    os.makedirs("examples/results", exist_ok=True)
    
    # Run all examples and collect results
    results = {}
    
    # Try the analytical solution first (this should always work)
    try:
        results["analytical"] = run_positronium_analytical()
        print("\n" + "="*60 + "\n")
    except Exception as e:
        print(f"Error in analytical solution: {e}")
    
    # Try the standard basis approach
    try:
        results["standard"] = run_positronium_standard()
        print("\n" + "="*60 + "\n")
    except Exception as e:
        print(f"Error in standard basis calculation: {e}")
    
    # Try the conditioned approach
    try:
        results["conditioned"] = run_positronium_with_conditioning()
        print("\n" + "="*60 + "\n")
    except Exception as e:
        print(f"Error in conditioned calculation: {e}")
    
    # Try the custom system approach
    try:
        results["custom"] = run_custom_positronium_system()
        print("\n" + "="*60 + "\n")
    except Exception as e:
        print(f"Error in custom system calculation: {e}")
    
    # Try the simplified positronium approach
    try:
        results["simplified"] = run_simplified_positronium()
        print("\n" + "="*60 + "\n")
    except Exception as e:
        print(f"Error in simplified calculation: {e}")
    
    # Generate a summary report
    print("\n===== SUMMARY OF RESULTS =====\n")
    
    for system_name, system_results in results.items():
        if system_results is None:
            print(f"System: {system_name} - Failed to complete")
            continue
            
        print(f"System: {system_name}")
        if "energy" in system_results:
            print(f"  Ground State Energy: {system_results['energy']:.10f} Hartree")
        if "mp2_energy" in system_results and system_results["mp2_energy"] is not None:
            print(f"  MP2 Correlation Energy: {system_results['mp2_energy']:.10f} Hartree")
        if "total_energy" in system_results and system_results["total_energy"] is not None:
            print(f"  Total MP2 Energy: {system_results['total_energy']:.10f} Hartree")
        if "annihilation_rate" in system_results and system_results["annihilation_rate"] is not None:
            print(f"  Positron Annihilation Rate: {system_results['annihilation_rate']:.6e} s^-1")
        if "method" in system_results:
            print(f"  Calculation Method: {system_results['method']}")
        print()
    
    # Compare to the exact analytical result
    print("\n===== COMPARISON TO EXACT RESULT =====\n")
    exact_energy = -0.25  # Hartree
    
    for system_name, system_results in results.items():
        if system_results is None or "energy" not in system_results:
            continue
            
        energy = system_results["energy"]
        error = energy - exact_energy
        error_ppm = error / abs(exact_energy) * 1e6
        
        print(f"System: {system_name}")
        print(f"  Energy: {energy:.10f} Hartree")
        print(f"  Error: {error:.10f} Hartree ({error_ppm:.2f} ppm)")
        print()
    
    print("Tests completed!") 