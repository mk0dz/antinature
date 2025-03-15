"""
Utility functions for antimatter quantum chemistry calculations.
"""

import numpy as np
from typing import Dict, Optional, Union
from .core.molecular_data import MolecularData
from .core.basis import MixedMatterBasis
from .core.integral_engine import AntimatterIntegralEngine
from .core.hamiltonian import AntimatterHamiltonian
from .core.scf import AntimatterSCF
from .core.correlation import AntimatterCorrelation

def create_antimatter_calculation(
    molecule_data: Union[Dict, MolecularData],
    basis_options: Optional[Dict] = None,
    calculation_options: Optional[Dict] = None
) -> Dict:
    """
    Create a complete antimatter calculation workflow.
    
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
    
    # Create basis sets
    basis_set = MixedMatterBasis()
    basis_set.create_for_molecule(
        [(atom, position) for atom, position in molecule_data.atoms],
        e_quality=basis_options.get('e_quality', 'standard'),
        p_quality=basis_options.get('p_quality', 'standard')
    )
    
    # Create integral engine
    integral_engine = AntimatterIntegralEngine(
        use_analytical=calculation_options.get('use_analytical', True),
        cache_size=calculation_options.get('cache_size', 10000)
    )
    
    # Create Hamiltonian
    hamiltonian = AntimatterHamiltonian(
        molecular_data=molecule_data,
        basis_set=basis_set,
        integral_engine=integral_engine,
        include_annihilation=calculation_options.get('include_annihilation', True),
        include_relativistic=calculation_options.get('include_relativistic', False)
    )
    
    # Build Hamiltonian matrices
    hamiltonian_matrices = hamiltonian.build_hamiltonian()
    
    # Create SCF solver
    scf_solver = AntimatterSCF(
        hamiltonian=hamiltonian_matrices,
        basis_set=basis_set,
        molecular_data=molecule_data,
        **calculation_options.get('scf_options', {})
    )
    
    return {
        'molecular_data': molecule_data,
        'basis_set': basis_set,
        'integral_engine': integral_engine,
        'hamiltonian': hamiltonian,
        'hamiltonian_matrices': hamiltonian_matrices,
        'scf_solver': scf_solver
    }

def run_antimatter_calculation(configuration: Dict) -> Dict:
    """
    Run a complete antimatter calculation using the provided configuration.
    
    Parameters:
    -----------
    configuration : Dict
        Configuration from create_antimatter_calculation
        
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
        correlation = AntimatterCorrelation(
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