#!/usr/bin/env python3
"""
Positronium Simulation Example
==============================

This example demonstrates how to set up and run a basic positronium 
(electron-positron bound state) simulation using the antimatter-beta package.
"""

import numpy as np
import matplotlib.pyplot as plt
from antimatter_qchem.core.molecular_data import MolecularData
from antimatter_qchem.core.basis import MixedMatterBasis
from antimatter_qchem.core.integral_engine import AntimatterIntegralEngine
from antimatter_qchem.core.hamiltonian import AntimatterHamiltonian
from antimatter_qchem.specialized.positronium import PositroniumSCF
from antimatter_qchem.specialized.annihilation import AnnihilationOperator
from antimatter_qchem.specialized.visualization import AntimatterVisualizer

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
    integral_engine = AntimatterIntegralEngine()
    integrals = integral_engine.compute_all_integrals(positronium, basis)
    print("Integrals computed successfully")
    
    # Step 4: Create Hamiltonian
    print("\nConstructing Hamiltonian...")
    hamiltonian = AntimatterHamiltonian()
    hamiltonian.build_hamiltonian(integrals, positronium, basis)
    print("Hamiltonian constructed")
    
    # Step 5: Run SCF calculation
    print("\nPerforming SCF calculation...")
    scf_solver = PositroniumSCF(
        hamiltonian=hamiltonian,
        basis_set=basis,
        molecular_data=positronium,
        max_iterations=50,
        convergence_threshold=1e-6
    )
    
    scf_result = scf_solver.run()
    print(f"SCF calculation converged in {scf_result['iterations']} iterations")
    print(f"Final energy: {scf_result['energy']:.8f} Hartree")
    
    # Step 6: Calculate annihilation rate
    print("\nCalculating annihilation properties...")
    annihilation_op = AnnihilationOperator(basis, scf_result)
    annihilation_result = annihilation_op.calculate_annihilation_rate()
    
    print(f"Annihilation rate: {annihilation_result['rate']:.6e} s^-1")
    print(f"Positronium lifetime: {annihilation_result['lifetime_ns']:.4f} ns")
    
    # Step 7: Visualize results
    print("\nGenerating visualizations...")
    visualizer = AntimatterVisualizer()
    fig = visualizer.plot_positronium_density(positronium, basis, scf_result)
    plt.savefig("positronium_density.png")
    print("Visualization saved to 'positronium_density.png'")
    
    print("\nSimulation completed successfully!")
    return {
        'energy': scf_result['energy'],
        'annihilation_rate': annihilation_result['rate'],
        'lifetime_ns': annihilation_result['lifetime_ns']
    }

if __name__ == "__main__":
    results = main() 