#!/usr/bin/env python3
"""
Anti-Hydrogen Simulation Example
================================

This example demonstrates how to set up and run a simulation of anti-hydrogen
(positron bound to an antiproton) using the antimatter package.
"""

import numpy as np
import matplotlib.pyplot as plt
from antimatter.core.molecular_data import MolecularData
from antimatter.core.basis import MixedMatterBasis
from antimatter.core.integral_engine import AntimatterIntegralEngine
from antimatter.core.hamiltonian import AntimatterHamiltonian
from antimatter.core.scf import AntimatterSCF
from antimatter.specialized.annihilation import AnnihilationOperator
from antimatter.specialized.relativistic import RelativisticCorrection
from antimatter.specialized.visualization import AntimatterVisualizer

def main():
    print("Anti-Hydrogen Simulation Example")
    print("================================")
    
    # Step 1: Create anti-hydrogen system
    print("\nCreating anti-hydrogen system...")
    
    # Anti-hydrogen: antiproton with positron (inverted charges)
    antihydrogen = MolecularData(
        atoms=[('H', np.array([0.0, 0.0, 0.0]))],
        n_electrons=0,         # No electrons
        n_positrons=1,         # One positron
        charge=-1,             # Overall negative charge (antiproton)
        name="Anti-Hydrogen",
        description="Antiproton with positron",
        units='bohr'
    )
    
    print(f"System created: {antihydrogen}")
    print(f"Particles: {antihydrogen.n_positrons} positron(s), {antihydrogen.n_electrons} electron(s)")
    
    # Step 2: Create basis sets
    print("\nSetting up basis set...")
    basis = MixedMatterBasis()
    
    # Anti-hydrogen needs positron basis functions
    basis.create_for_molecule(
        antihydrogen.atoms, 
        electron_quality='none',  # No electrons
        positron_quality='high'   # High quality for positron
    )
    
    print(f"Basis created with {basis.n_electron_basis} electron and {basis.n_positron_basis} positron functions")
    
    # Step 3: Compute integrals
    print("\nComputing integrals...")
    integral_engine = AntimatterIntegralEngine()
    integrals = integral_engine.compute_all_integrals(antihydrogen, basis)
    print("Integrals computed successfully")
    
    # Step 4: Build Hamiltonian
    print("\nBuilding Hamiltonian...")
    hamiltonian = AntimatterHamiltonian()
    hamiltonian.build_hamiltonian(integrals, antihydrogen, basis)
    print("Hamiltonian constructed")
    
    # Step 5: Include relativistic corrections
    print("\nAdding relativistic corrections...")
    rel_correction = RelativisticCorrection(hamiltonian, basis, antihydrogen)
    rel_correction.calculate_relativistic_integrals()
    hamiltonian_rel = rel_correction.apply_corrections()
    print("Relativistic corrections applied")
    
    # Step 6: Run SCF calculation
    print("\nPerforming SCF calculation...")
    scf_solver = AntimatterSCF(
        hamiltonian=hamiltonian_rel,
        basis_set=basis,
        molecular_data=antihydrogen,
        max_iterations=100,
        convergence_threshold=1e-6
    )
    
    scf_result = scf_solver.run()
    print(f"SCF calculation converged in {scf_result['iterations']} iterations")
    print(f"Anti-hydrogen energy: {scf_result['energy']:.8f} Hartree")
    
    # Step 7: Calculate properties
    print("\nCalculating properties...")
    
    # Calculate positron density
    positron_density = scf_solver.calculate_positron_density()
    
    # Calculate orbital energies
    orbital_energies = scf_result.get('orbital_energies', [])
    if orbital_energies:
        print("\nPositron orbital energies (Hartree):")
        for i, energy in enumerate(orbital_energies):
            print(f"  Orbital {i}: {energy:.6f}")
    
    # Step 8: Visualize results
    print("\nGenerating visualizations...")
    visualizer = AntimatterVisualizer()
    fig = visualizer.plot_positron_density(antihydrogen, basis, scf_result)
    plt.savefig("anti_hydrogen_density.png")
    print("Density plot saved to 'anti_hydrogen_density.png'")
    
    print("\nSimulation completed successfully!")
    return {
        'energy': scf_result['energy'],
        'iterations': scf_result['iterations'],
        'orbital_energies': orbital_energies
    }

if __name__ == "__main__":
    results = main() 