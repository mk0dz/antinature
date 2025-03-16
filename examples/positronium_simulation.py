#!/usr/bin/env python3
"""
Positronium Simulation Example
==============================

This example demonstrates how to set up and run a basic positronium 
(electron-positron bound state) simulation using the antiverse package.
"""

import numpy as np
import matplotlib.pyplot as plt
from antiverse.core.molecular_data import MolecularData
from antiverse.core.basis import MixedMatterBasis
from antiverse.core.integral_engine import antiverseIntegralEngine
from antiverse.core.hamiltonian import antiverseHamiltonian
from antiverse.specialized.positronium import PositroniumSCF
from antiverse.specialized.annihilation import AnnihilationOperator
from antiverse.specialized.visualization import antiverseVisualizer

def main():
    print("Positronium Simulation Example")
    print("==============================")
    
    # Step 1: Create positronium system
    print("\nCreating positronium molecular system...")
    positronium = MolecularData.positronium()
    print(f"System created: {positronium}")
    
    # Step 2: Create mixed basis set
    print("\nSetting up specialized basis set...")
    basis = MixedMatterBasis()
    basis.create_positronium_basis(quality='standard')
    print(f"Basis created with {basis.n_electron_basis} electron and {basis.n_positron_basis} positron functions")
    
    # Step 3: Set up integral engine
    print("\nInitializing integral engine...")
    integral_engine = antiverseIntegralEngine()
    
    # Step 4: Create Hamiltonian
    print("\nConstructing Hamiltonian...")
    hamiltonian = antiverseHamiltonian(
        molecular_data=positronium,
        basis_set=basis,
        integral_engine=integral_engine,
        include_annihilation=True
    )
    
    # Build the Hamiltonian matrices
    matrices = hamiltonian.build_hamiltonian()
    print("Hamiltonian constructed")
    
    # Step 5: Run SCF calculation
    print("\nPerforming SCF calculation...")
    scf_solver = PositroniumSCF(
        hamiltonian=matrices,
        basis_set=basis,
        molecular_data=positronium,
        max_iterations=50,
        convergence_threshold=1e-6
    )
    
    scf_result = scf_solver.solve_scf()
    print(f"SCF calculation converged in {scf_result['iterations']} iterations")
    print(f"Final energy: {scf_result['energy']:.8f} Hartree")
    
    # Step 6: Calculate annihilation rate
    print("\nCalculating annihilation properties...")
    annihilation_op = AnnihilationOperator(basis, scf_result)
    annihilation_rate = annihilation_op.calculate_annihilation_rate()
    
    # Calculate lifetime
    lifetime_result = annihilation_op.calculate_lifetime(annihilation_rate)
    
    print(f"Annihilation rate: {annihilation_rate:.6e} s^-1")
    print(f"Positronium lifetime: {lifetime_result['lifetime_ns']:.4f} ns")
    
    # Step 7: Visualize results
    print("\nGenerating visualizations...")
    visualizer = antiverseVisualizer()
    fig = visualizer.plot_positronium_density(positronium, basis, scf_result)
    plt.savefig("positronium_density.png")
    print("Visualization saved to 'positronium_density.png'")
    
    print("\nSimulation completed successfully!")
    return {
        'energy': scf_result['energy'],
        'annihilation_rate': annihilation_rate,
        'lifetime_ns': lifetime_result['lifetime_ns']
    }

if __name__ == "__main__":
    results = main() 