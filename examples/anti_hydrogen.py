# example/anti_hydrogen.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from antimatter_qchem import (
    MolecularData, MixedMatterBasis, AntimatterIntegralEngine,
    AntimatterHamiltonian, AntimatterSCF, AntimatterCorrelation,
    create_antimatter_calculation, run_antimatter_calculation
)
from antimatter_qchem.specialized import RelativisticCorrection

def main():
    """Example of anti-hydrogen calculation with relativistic effects."""
    
    print("Performing anti-hydrogen calculation...")
    
    # Create anti-hydrogen molecular data
    # For anti-hydrogen, we use a positron orbiting an antiproton
    antihydrogen = MolecularData(
        atoms=[('H', np.array([0.0, 0.0, 0.0]))],  # Antiproton at origin
        n_electrons=0,
        n_positrons=1,
        charge=0  # Net charge is still 0
    )
    
    # Run without relativistic effects first
    print("\nCalculation without relativistic effects:")
    nonrel_calculation = create_antimatter_calculation(
        antihydrogen,
        basis_options={'p_quality': 'extended'},
        calculation_options={'include_relativistic': False}
    )
    
    nonrel_results = run_antimatter_calculation(nonrel_calculation)
    
    print(f"Non-relativistic energy: {nonrel_results['scf']['energy']:.10f} Hartree")
    
    # Run with relativistic effects
    print("\nCalculation with relativistic effects:")
    rel_calculation = create_antimatter_calculation(
        antihydrogen,
        basis_options={'p_quality': 'extended'},
        calculation_options={'include_relativistic': True}
    )
    
    rel_results = run_antimatter_calculation(rel_calculation)
    
    print(f"Relativistic energy: {rel_results['scf']['energy']:.10f} Hartree")
    
    # Calculate relativistic correction
    rel_correction = rel_results['scf']['energy'] - nonrel_results['scf']['energy']
    
    print(f"Relativistic correction: {rel_correction:.10f} Hartree")

if __name__ == "__main__":
    main()