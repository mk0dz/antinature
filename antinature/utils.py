"""
User-friendly utilities for antinature quantum chemistry calculations.

This module provides simple, high-level functions that automatically handle
optimization, error recovery, and provide reliable results for antimatter systems.
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .core.basis import MixedMatterBasis
from .core.hamiltonian import AntinatureHamiltonian
from .core.integral_engine import AntinatureIntegralEngine
from .core.molecular_data import MolecularData
from .core.scf import AntinatureSCF


class AntinatureCalculator:
    """
    High-level calculator for antinature systems with automatic optimization.
    
    This class provides a user-friendly interface that automatically:
    - Selects optimal basis sets
    - Handles numerical stability issues
    - Applies error recovery techniques
    - Provides comprehensive results analysis
    """
    
    def __init__(self, print_level: int = 1):
        """
        Initialize the calculator.
        
        Parameters:
        -----------
        print_level : int
            Verbosity level (0=minimal, 1=normal, 2=verbose)
        """
        self.print_level = print_level
        self.results_history = []
        
    def calculate_positronium(self, accuracy: str = 'medium') -> Dict:
        """
        Calculate ground state properties of positronium with automatic optimization.
        
        Parameters:
        -----------
        accuracy : str
            Desired accuracy level ('low', 'medium', 'high')
            
        Returns:
        --------
        Dict
            Comprehensive results including energy, orbitals, and diagnostics
        """
        if self.print_level > 0:
            print("=" * 60)
            print("ANTINATURE POSITRONIUM CALCULATION")
            print("=" * 60)
            print(f"Accuracy level: {accuracy}")
            
        start_time = time.time()
        
        # Create positronium system
        ps_data = MolecularData.positronium()
        
        # Get optimal basis set
        basis_set = self._get_optimal_basis(ps_data, accuracy)
        
        # Build Hamiltonian with error handling
        hamiltonian_matrices = self._build_hamiltonian_safely(ps_data, basis_set)
        
        # Run SCF calculation with automatic optimization
        scf_results = self._run_scf_with_optimization(
            hamiltonian_matrices, basis_set, ps_data, accuracy
        )
        
        # Analyze and format results
        results = self._analyze_results(scf_results, ps_data, 'positronium', start_time)
        
        # Store in history
        self.results_history.append(results)
        
        return results
    
    def calculate_custom_system(
        self, 
        atoms: List[Tuple[str, np.ndarray]], 
        n_electrons: int, 
        n_positrons: int,
        accuracy: str = 'medium'
    ) -> Dict:
        """
        Calculate properties of a custom antimatter system.
        
        Parameters:
        -----------
        atoms : List[Tuple[str, np.ndarray]]
            List of (element, position) tuples
        n_electrons : int
            Number of electrons
        n_positrons : int
            Number of positrons
        accuracy : str
            Desired accuracy level
            
        Returns:
        --------
        Dict
            Comprehensive results
        """
        if self.print_level > 0:
            print("=" * 60)
            print("ANTINATURE CUSTOM SYSTEM CALCULATION")
            print("=" * 60)
            
        start_time = time.time()
        
        # Create molecular data
        mol_data = MolecularData(
            atoms=atoms,
            n_electrons=n_electrons,
            n_positrons=n_positrons,
            charge=0
        )
        
        # Get optimal basis set
        basis_set = self._get_optimal_basis(mol_data, accuracy)
        
        # Build Hamiltonian with error handling
        hamiltonian_matrices = self._build_hamiltonian_safely(mol_data, basis_set)
        
        # Run SCF calculation
        scf_results = self._run_scf_with_optimization(
            hamiltonian_matrices, basis_set, mol_data, accuracy
        )
        
        # Analyze results
        results = self._analyze_results(scf_results, mol_data, 'custom', start_time)
        
        return results
    
    def _get_optimal_basis(self, molecular_data, accuracy: str) -> MixedMatterBasis:
        """
        Automatically select and optimize basis set for the system.
        """
        basis = MixedMatterBasis()
        
        if molecular_data.is_positronium:
            # Use optimized positronium basis
            basis.create_optimized_positronium_basis(target_accuracy=accuracy)
        else:
            # Create basis for general system
            basis.create_for_molecule(
                atoms=molecular_data.atoms,
                e_quality=accuracy,
                p_quality=accuracy
            )
        
        # Set up integral engine
        integral_engine = AntinatureIntegralEngine()
        basis.set_integral_engine(integral_engine)
        
        # Check for and remove linear dependencies
        removal_info = basis.remove_linear_dependencies()
        
        if removal_info.get('removed_functions', 0) > 0 and self.print_level > 0:
            print(f"Removed {removal_info['removed_functions']} linearly dependent functions")
            print(f"Final basis: {basis.n_electron_basis} electron + {basis.n_positron_basis} positron")
        
        return basis
    
    def _build_hamiltonian_safely(self, molecular_data, basis_set) -> Dict:
        """
        Build Hamiltonian with error handling and recovery.
        """
        try:
            integral_engine = AntinatureIntegralEngine()
            hamiltonian = AntinatureHamiltonian(
                molecular_data=molecular_data,
                basis_set=basis_set,
                integral_engine=integral_engine,
                include_annihilation=True,
                include_relativistic=False  # Keep simple for now
            )
            
            matrices = hamiltonian.build_hamiltonian()
            
            # Check overlap matrix condition
            S = matrices['overlap']
            cond_num = np.linalg.cond(S)
            
            if cond_num > 1e12:
                if self.print_level > 0:
                    print(f"WARNING: Overlap matrix poorly conditioned (κ = {cond_num:.2e})")
                    print("Applying regularization...")
                
                # Apply regularization
                eigenvals, eigenvecs = np.linalg.eigh(S)
                threshold = 1e-10
                good_vals = eigenvals > threshold
                
                if np.sum(~good_vals) > 0:
                    eigenvals_reg = eigenvals.copy()
                    eigenvals_reg[~good_vals] = threshold
                    S_reg = eigenvecs @ np.diag(eigenvals_reg) @ eigenvecs.T
                    matrices['overlap'] = S_reg
                    
                    if self.print_level > 0:
                        print(f"Regularized {np.sum(~good_vals)} small eigenvalues")
            
            return matrices
            
        except Exception as e:
            if self.print_level > 0:
                print(f"Error building Hamiltonian: {e}")
                print("Attempting recovery with simpler basis...")
            
            # Try with simpler basis
            basis_simple = MixedMatterBasis()
            if molecular_data.is_positronium:
                basis_simple.create_optimized_positronium_basis(target_accuracy='low')
            else:
                basis_simple.create_for_molecule(
                    atoms=molecular_data.atoms,
                    e_quality='minimal',
                    p_quality='minimal'
                )
            
            integral_engine = AntinatureIntegralEngine()
            hamiltonian = AntinatureHamiltonian(
                molecular_data=molecular_data,
                basis_set=basis_simple,
                integral_engine=integral_engine,
                include_annihilation=False,
                include_relativistic=False
            )
            
            return hamiltonian.build_hamiltonian()
    
    def _run_scf_with_optimization(self, hamiltonian_matrices, basis_set, molecular_data, accuracy) -> Dict:
        """
        Run SCF calculation with automatic optimization and error recovery.
        """
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                if self.print_level > 0 and attempt > 0:
                    print(f"\nSCF attempt {attempt + 1}/{max_attempts}")
                
                # Configure SCF parameters based on accuracy
                if accuracy == 'low':
                    max_iter, conv_thresh, damping = 50, 1e-4, 0.7
                elif accuracy == 'medium':
                    max_iter, conv_thresh, damping = 100, 1e-6, 0.5
                else:  # high
                    max_iter, conv_thresh, damping = 200, 1e-8, 0.3
                
                scf_solver = AntinatureSCF(
                    hamiltonian=hamiltonian_matrices,
                    basis_set=basis_set,
                    molecular_data=molecular_data,
                    max_iterations=max_iter,
                    convergence_threshold=conv_thresh,
                    use_diis=True,
                    damping_factor=damping,
                    level_shifting=0.1 if attempt > 0 else 0.0,  # Use level shifting on retry
                    print_level=1 if self.print_level > 1 else 0
                )
                
                results = scf_solver.solve_scf()
                
                if results['converged']:
                    return results
                elif attempt < max_attempts - 1:
                    if self.print_level > 0:
                        print("SCF did not converge, trying with different parameters...")
                
            except Exception as e:
                if self.print_level > 0:
                    print(f"SCF attempt {attempt + 1} failed: {e}")
                
                if attempt == max_attempts - 1:
                    # Final attempt with very conservative settings
                    try:
                        scf_solver = AntinatureSCF(
                            hamiltonian=hamiltonian_matrices,
                            basis_set=basis_set,
                            molecular_data=molecular_data,
                            max_iterations=20,
                            convergence_threshold=1e-3,
                            use_diis=False,
                            damping_factor=0.9,
                            level_shifting=0.5,
                            print_level=0
                        )
                        return scf_solver.solve_scf()
                    except:
                        pass
        
        # If all attempts failed, return error result
        return {
            'converged': False,
            'energy': 0.0,
            'iterations': 0,
            'error': 'SCF convergence failed after all attempts'
        }
    
    def _analyze_results(self, scf_results, molecular_data, system_type, start_time) -> Dict:
        """
        Analyze and format results with comprehensive diagnostics.
        """
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Basic results
        results = {
            'system_type': system_type,
            'converged': scf_results.get('converged', False),
            'energy': scf_results.get('energy', 0.0),
            'iterations': scf_results.get('iterations', 0),
            'computation_time': computation_time,
            'n_electrons': molecular_data.n_electrons,
            'n_positrons': molecular_data.n_positrons
        }
        
        # Add orbital information if available
        if 'E_electron' in scf_results and scf_results['E_electron'] is not None:
            results['electron_orbital_energies'] = scf_results['E_electron']
            results['homo_electron'] = scf_results['E_electron'][molecular_data.n_electrons//2 - 1] if molecular_data.n_electrons > 0 else None
            results['lumo_electron'] = scf_results['E_electron'][molecular_data.n_electrons//2] if molecular_data.n_electrons < len(scf_results['E_electron']) else None
        
        if 'E_positron' in scf_results and scf_results['E_positron'] is not None:
            results['positron_orbital_energies'] = scf_results['E_positron']
            results['homo_positron'] = scf_results['E_positron'][molecular_data.n_positrons//2 - 1] if molecular_data.n_positrons > 0 else None
            results['lumo_positron'] = scf_results['E_positron'][molecular_data.n_positrons//2] if molecular_data.n_positrons < len(scf_results['E_positron']) else None
        
        # Theoretical comparison for known systems
        theoretical_energy = None
        if system_type == 'positronium':
            theoretical_energy = -0.25  # Exact value in Hartree
        
        if theoretical_energy is not None:
            error = abs(results['energy'] - theoretical_energy)
            results['theoretical_energy'] = theoretical_energy
            results['absolute_error'] = error
            results['relative_error'] = error / abs(theoretical_energy) if theoretical_energy != 0 else float('inf')
            
            # Performance assessment
            if error < 1e-8:
                results['accuracy_assessment'] = 'EXCELLENT'
            elif error < 1e-6:
                results['accuracy_assessment'] = 'GOOD'
            elif error < 1e-4:
                results['accuracy_assessment'] = 'FAIR'
            else:
                results['accuracy_assessment'] = 'POOR'
        
        # Print summary
        if self.print_level > 0:
            self._print_results_summary(results)
        
        return results
    
    def _print_results_summary(self, results):
        """Print a formatted summary of results."""
        print("\n" + "=" * 60)
        print("CALCULATION RESULTS")
        print("=" * 60)
        
        print(f"System: {results['system_type'].title()}")
        print(f"Converged: {'✓' if results['converged'] else '✗'}")
        print(f"Total Energy: {results['energy']:.8f} Hartree")
        print(f"Iterations: {results['iterations']}")
        print(f"Computation Time: {results['computation_time']:.2f} seconds")
        
        if 'theoretical_energy' in results:
            print(f"\nAccuracy Analysis:")
            print(f"  Theoretical Energy: {results['theoretical_energy']:.8f} Hartree")
            print(f"  Absolute Error: {results['absolute_error']:.2e} Hartree")
            print(f"  Relative Error: {results['relative_error']:.2e}")
            print(f"  Assessment: {results['accuracy_assessment']}")
        
        if 'homo_electron' in results and results['homo_electron'] is not None:
            print(f"\nOrbital Energies:")
            print(f"  Electron HOMO: {results['homo_electron']:.6f} Hartree")
            if results['lumo_electron'] is not None:
                print(f"  Electron LUMO: {results['lumo_electron']:.6f} Hartree")
        
        if 'homo_positron' in results and results['homo_positron'] is not None:
            print(f"  Positron HOMO: {results['homo_positron']:.6f} Hartree")
            if results['lumo_positron'] is not None:
                print(f"  Positron LUMO: {results['lumo_positron']:.6f} Hartree")
        
        print("=" * 60)


def calculate_positronium(accuracy: str = 'medium') -> Dict:
    """
    Convenient function to calculate positronium properties.
    
    Parameters:
    -----------
    accuracy : str
        Desired accuracy level ('low', 'medium', 'high')
        
    Returns:
    --------
    Dict
        Comprehensive results
    """
    calculator = AntinatureCalculator()
    return calculator.calculate_positronium(accuracy)


def calculate_antihydrogen(accuracy: str = 'medium') -> Dict:
    """
    Calculate anti-hydrogen ground state properties.
    
    Parameters:
    -----------
    accuracy : str
        Desired accuracy level
        
    Returns:
    --------
    Dict
        Comprehensive results
    """
    calculator = AntinatureCalculator()
    
    # Anti-hydrogen: antiproton + positron
    atoms = [('H', np.array([0.0, 0.0, 0.0]))]  # Antiproton at origin
    
    return calculator.calculate_custom_system(
        atoms=atoms,
        n_electrons=0,
        n_positrons=1,
        accuracy=accuracy
    )


def quick_test() -> bool:
    """
    Quick test to verify the antinature module is working correctly.
    
    Returns:
    --------
    bool
        True if test passes, False otherwise
    """
    try:
        print("Running quick test of antinature module...")
        results = calculate_positronium(accuracy='low')
        
        success = (
            results['converged'] and 
            abs(results['energy'] - (-0.25)) < 0.1  # Reasonable tolerance for quick test
        )
        
        if success:
            print("✓ Quick test PASSED")
            print(f"  Energy: {results['energy']:.6f} Hartree")
            print(f"  Error: {abs(results['energy'] - (-0.25)):.6f} Hartree")
        else:
            print("✗ Quick test FAILED")
            
        return success
        
    except Exception as e:
        print(f"✗ Quick test FAILED with error: {e}")
        return False


# Legacy compatibility functions
def calculate_molecule_properties(atoms, n_electrons, n_positrons, charge=0):
    """Legacy function for backward compatibility."""
    calculator = AntinatureCalculator()
    return calculator.calculate_custom_system(atoms, n_electrons, n_positrons)


def run_scf_calculation(molecular_data, basis_set=None, **kwargs):
    """Legacy function for backward compatibility."""
    calculator = AntinatureCalculator()
    
    if molecular_data.is_positronium:
        return calculator.calculate_positronium(kwargs.get('accuracy', 'medium'))
    else:
        return calculator.calculate_custom_system(
            molecular_data.atoms,
            molecular_data.n_electrons,
            molecular_data.n_positrons,
            kwargs.get('accuracy', 'medium')
        )


def create_antinature_calculation(
    system_type: str = 'positronium',
    accuracy: str = 'medium',
    **kwargs
) -> AntinatureCalculator:
    """
    Create and configure an antinature calculator for various systems.
    
    Parameters:
    -----------
    system_type : str
        Type of system to calculate ('positronium', 'antihydrogen', 'custom')
    accuracy : str
        Accuracy level ('low', 'medium', 'high')
    **kwargs : dict
        Additional keyword arguments for custom systems
        
    Returns:
    --------
    AntinatureCalculator
        Configured calculator instance ready for calculations
        
    Examples:
    ---------
    >>> calc = create_antinature_calculation('positronium')
    >>> results = calc.calculate_positronium()
    
    >>> calc = create_antinature_calculation('custom', atoms=[('H', [0,0,0])], n_positrons=1)
    >>> results = calc.calculate_custom_system(...)
    """
    calculator = AntinatureCalculator(print_level=kwargs.get('print_level', 1))
    
    # Pre-configure for specific systems if needed
    if system_type == 'positronium':
        # Positronium-specific configuration
        calculator._default_accuracy = accuracy
    elif system_type == 'antihydrogen':
        # Antihydrogen-specific configuration
        calculator._default_accuracy = accuracy
    elif system_type == 'custom':
        # Custom system configuration
        calculator._custom_params = kwargs
    
    return calculator
