# example/positronium.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from antimatter_qchem.core import (
    MolecularData, create_antimatter_calculation, run_antimatter_calculation
)
from antimatter_qchem.specialized import AnnihilationOperator

def main():
    """Example of positronium calculation."""
    
    print("Performing positronium calculation...")
    
    # Create positronium molecular data
    positronium = MolecularData(
        atoms=[('H', np.array([0.0, 0.0, 0.0]))],  # Use H as placeholder
        n_electrons=1,
        n_positrons=1,
        charge=0
    )
    
    # Initialize calculation configuration
    calculation = create_antimatter_calculation(
        positronium,
        basis_options={
            'e_quality': 'extended',
            'p_quality': 'extended'
        },
        calculation_options={
            'include_annihilation': True,
            'include_relativistic': False,
            'scf_options': {
                'max_iterations': 50,
                'convergence_threshold': 1e-6,
                'use_diis': True
            }
        }
    )
    
    # Run the calculation
    results = run_antimatter_calculation(configuration=calculation)
    
    # Print SCF results
    print("\nSCF Results:")
    print(f"Energy: {results['scf']['energy']:.10f} Hartree")
    print(f"Converged: {results['scf']['converged']}")
    print(f"Iterations: {results['scf']['iterations']}")
    print(f"Computation time: {results['scf']['computation_time']:.3f} seconds")
    
    # Calculate annihilation rate and lifetime
    if results['scf']['converged']:
        # Create annihilation operator
        annihilation = AnnihilationOperator(calculation['basis_set'], results['scf'])
        
        # Calculate annihilation channels
        channels = annihilation.analyze_annihilation_channels()
        
        # Calculate lifetime
        lifetime = annihilation.calculate_lifetime(channels['total'])
        
        print("\nAnnihilation Results:")
        print(f"2γ rate: {channels['two_gamma']:.6e} a.u.")
        print(f"3γ rate: {channels['three_gamma']:.6e} a.u.")
        print(f"Total rate: {channels['total']:.6e} a.u.")
        print(f"2γ/3γ ratio: {channels['ratio_2g_3g']:.2f}")
        print(f"Lifetime: {lifetime['lifetime_ns']:.3f} ns")
    
if __name__ == "__main__":
    main()