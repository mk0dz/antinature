"""
Utility functions for quantimatter quantum chemistry calculations.
"""

import numpy as np
import importlib
from typing import Dict, Optional, Union, List, Tuple
import warnings

# Import project modules
from .core.molecular_data import MolecularData
from .core.basis import MixedMatterBasis
from .core.integral_engine import quantimatterIntegralEngine
from .core.hamiltonian import quantimatterHamiltonian
from .core.scf import quantimatterSCF
from .core.correlation import quantimatterCorrelation

def check_dependencies(dependencies: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    Check if all dependencies are installed with required versions.
    
    Parameters:
    -----------
    dependencies : Dict[str, str]
        Dictionary mapping package names to required version specs
        
    Returns:
    --------
    Tuple[bool, List[str]]
        (Success, List of missing/incompatible packages)
    
    Example:
    --------
    >>> check_dependencies({'numpy': '>=1.20.0', 'qiskit': '>=1.0.0'})
    """
    import pkg_resources
    
    missing = []
    
    for package, version_spec in dependencies.items():
        try:
            pkg_resources.require(f"{package}{version_spec}")
        except (pkg_resources.VersionConflict, pkg_resources.DistributionNotFound):
            missing.append(f"{package}{version_spec}")
    
    return len(missing) == 0, missing

def check_optional_dependencies() -> Dict[str, bool]:
    """
    Check which optional dependencies are available.
    
    Returns:
    --------
    Dict[str, bool]
        Dictionary indicating which optional features are available
    """
    dependencies = {
        'qiskit': False,       # For quantum simulation
        'pyscf': False,        # For advanced electronic structure
        'openfermion': False,  # For quantum chemistry mapping
    }
    
    # Check Qiskit
    try:
        import qiskit
        dependencies['qiskit'] = True
    except ImportError:
        pass
    
    # Check PySCF
    try:
        import pyscf
        dependencies['pyscf'] = True
    except ImportError:
        pass
    
    # Check OpenFermion
    try:
        import openfermion
        dependencies['openfermion'] = True
    except ImportError:
        pass
    
    return dependencies

def create_quantimatter_calculation(
    molecule_data: Union[Dict, MolecularData],
    basis_options: Optional[Dict] = None,
    calculation_options: Optional[Dict] = None
) -> Dict:
    """
    Create a complete quantimatter calculation workflow.
    
    Parameters:
    -----------
    molecule_data : Dict or MolecularData
        Molecular structure information
    basis_options : Dict, optional
        Options for basis set generation
    calculation_options : Dict, optional
        Options for calculation parameters
        
    Returns:
    --------
    Dict
        Configuration for the calculation
    """
    # Initialize molecular data if needed
    if not isinstance(molecule_data, MolecularData):
        molecule_data = MolecularData(**molecule_data)
    
    # Set default options
    if basis_options is None:
        basis_options = {'quality': 'standard'}
    
    if calculation_options is None:
        calculation_options = {
            'include_annihilation': True,
            'include_relativistic': False,
            'scf_options': {
                'max_iterations': 100,
                'convergence_threshold': 1e-6,
                'use_diis': True
            }
        }
    
    # Create basis
    basis = MixedMatterBasis()
    if hasattr(basis, f"create_{molecule_data.name.lower()}_basis") and basis_options.get('quality') == molecule_data.name.lower():
        # Use specialized basis if available
        getattr(basis, f"create_{molecule_data.name.lower()}_basis")()
    else:
        # Use general basis otherwise
        quality = basis_options.get('quality', 'standard')
        basis.create_for_molecule(
            molecule_data.atoms,
            electron_quality=quality,
            positron_quality=quality
        )
    
    # Create integral engine and compute integrals
    integral_engine = quantimatterIntegralEngine()
    integrals = integral_engine.compute_all_integrals(molecule_data, basis)
    
    # Create Hamiltonian
    hamiltonian = quantimatterHamiltonian()
    hamiltonian.build_hamiltonian(integrals, molecule_data, basis)
    
    # Apply relativistic corrections if requested
    if calculation_options.get('include_relativistic', False):
        from .specialized.relativistic import RelativisticCorrection
        rel_correction = RelativisticCorrection(hamiltonian, basis, molecule_data)
        rel_correction.calculate_relativistic_integrals()
        hamiltonian = rel_correction.apply_corrections()
    
    # Run SCF calculation
    scf_solver = quantimatterSCF(
        hamiltonian=hamiltonian,
        basis_set=basis,
        molecular_data=molecule_data,
        **calculation_options.get('scf_options', {})
    )
    
    scf_result = scf_solver.run()
    
    # Calculate annihilation rate if requested
    if calculation_options.get('include_annihilation', False) and molecule_data.n_positrons > 0:
        from .specialized.annihilation import AnnihilationOperator
        annihilation_op = AnnihilationOperator(basis, scf_result)
        annihilation_result = annihilation_op.calculate_annihilation_rate()
        scf_result.update(annihilation_result)
    
    return scf_result

def run_quantimatter_calculation(configuration: Dict) -> Dict:
    """
    Run a complete quantimatter calculation using the provided configuration.
    
    Parameters:
    -----------
    configuration : Dict
        Configuration from create_quantimatter_calculation
        
    Returns:
    --------
    Dict
        Results of the calculation
    """
    # Extract components
    scf_solver = configuration['scf_solver']
    
    # Run SCF calculation
    scf_results = scf_solver.solve_scf()
    
    # Optionally run post-SCF calculations
    post_scf_results = {}
    
    if configuration.get('run_mp2', False):
        correlation = quantimatterCorrelation(
            scf_result=scf_results,
            hamiltonian=configuration['hamiltonian_matrices'],
            basis=configuration['basis_set']
        )
        post_scf_results['mp2_energy'] = correlation.mp2_energy()
    
    if configuration.get('calculate_annihilation', False) and 'correlation' in locals():
        post_scf_results['annihilation_rate'] = correlation.calculate_annihilation_rate()
    
    # Combine results
    results = {
        'scf': scf_results,
        'post_scf': post_scf_results,
        'molecular_data': configuration['molecular_data'],
        'basis_info': {
            'n_electron_basis': configuration['basis_set'].n_electron_basis,
            'n_positron_basis': configuration['basis_set'].n_positron_basis,
            'n_total_basis': configuration['basis_set'].n_total_basis
        }
    }
    
    return results