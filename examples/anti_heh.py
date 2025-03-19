#!/usr/bin/env python
# anti_heh.py - Anti-helium hydride ion (anti-HeH+) example
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

# Add the parent directory to the path to find the antinature module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from antinature.core.molecular_data import MolecularData
from antinature.core.basis import MixedMatterBasis
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.scf import AntinatureSCF
from antinature.core.correlation import AntinatureCorrelation

def run_anti_heh_ion():
    """
    Run calculations for anti-helium hydride ion (anti-HeH+).
    
    This represents an exotic antimatter system consisting of:
    - Anti-helium nucleus (anti-alpha particle with charge -2)
    - Anti-hydrogen nucleus (anti-proton with charge -1)
    - 2 positrons (e+) to balance the -3 charge of the nuclei
    
    It's the antimatter equivalent of the HeH+ ion (the first molecule
    formed in the early universe).
    """
    print("\n=== Anti-Helium Hydride Ion (anti-HeH+) Analysis ===\n")
    
    # Create anti-HeH+ system
    # The atoms list represents the nuclei positions, but they're anti-nuclei
    anti_heh_data = MolecularData(
        atoms=[
            ('He', np.array([0.0, 0.0, 0.0])),
            ('H', np.array([0.0, 0.0, 1.46]))  # ~1.46 Bohr ≈ 0.77 Å bond distance
        ],
        n_electrons=0,       # No electrons in antimatter system
        n_positrons=2,       # 2 positrons (equivalent to 2 electrons in normal HeH+)
        charge=0,            # Overall neutral (2 positrons balance -2 from anti-He, anti-H)
        name="Anti-HeH+",
        description="Anti-helium hydride ion (anti-HeH+) with exotic antimatter composition"
    )
    
    # Print molecular information
    print(f"Molecule: {anti_heh_data.name}")
    print(f"Description: {anti_heh_data.description}")
    print(f"Formula: {anti_heh_data.get_formula()}")
    print(f"Number of positrons: {anti_heh_data.n_positrons}")
    print(f"Nuclear repulsion energy: {anti_heh_data.get_nuclear_repulsion_energy():.8f} Hartree")
    
    # Create a basis set (use minimal for simplicity)
    basis = MixedMatterBasis()
    basis.create_for_molecule(
        atoms=anti_heh_data.atoms,
        e_quality='none',     # No electron basis needed
        p_quality='minimal'   # Minimal positron basis
    )
    
    # Print basis information
    p_basis_info = basis.positron_basis.get_function_types() if basis.positron_basis else {}
    print("\nBasis set information:")
    print(f"Positron basis functions: {len(basis.positron_basis) if basis.positron_basis else 0}")
    print(f"Positron function types: {p_basis_info}")
    
    # Create integral engine
    engine = AntinatureIntegralEngine()
    
    # Create Hamiltonian
    print("\nBuilding Hamiltonian...")
    t_start = time.time()
    hamiltonian = AntinatureHamiltonian(
        molecular_data=anti_heh_data,
        basis_set=basis,
        integral_engine=engine,
        include_relativistic=True  # Include relativistic effects for antimatter
    )
    
    # Build the Hamiltonian matrices
    h_matrices = hamiltonian.build_hamiltonian()
    t_hamiltonian = time.time() - t_start
    print(f"Hamiltonian built in {t_hamiltonian:.3f} seconds")
    
    # Create SCF solver
    print("\nStarting SCF calculation...")
    t_start = time.time()
    scf = AntinatureSCF(
        hamiltonian=h_matrices,
        basis_set=basis,
        molecular_data=anti_heh_data,
        max_iterations=50,
        convergence_threshold=1e-6,
        use_diis=True,
        damping_factor=0.5,
        print_level=1
    )
    
    # Run SCF calculation
    scf_results = scf.solve_scf()
    t_scf = time.time() - t_start
    print(f"SCF completed in {t_scf:.3f} seconds")
    
    # Print SCF results
    print(f"\nSCF energy: {scf_results['energy']:.10f} Hartree")
    print(f"Convergence: {scf_results['converged']}")
    print(f"Iterations: {scf_results['iterations']}")
    
    # Try correlation calculations
    print("\nStarting correlation calculations...")
    t_start = time.time()
    try:
        # Create correlation object
        corr = AntinatureCorrelation(
            scf_result=scf_results,
            hamiltonian=h_matrices,
            basis_set=basis,
            molecular_data=anti_heh_data,
            method="MP2"  # Use MP2 for correlation
        )
        
        # Run correlation calculation
        corr_results = corr.compute_correlation()
        t_corr = time.time() - t_start
        print(f"Correlation completed in {t_corr:.3f} seconds")
        
        # Print correlation results
        print(f"\nMP2 correlation energy: {corr_results['correlation_energy']:.10f} Hartree")
        print(f"Total MP2 energy: {corr_results['total_energy']:.10f} Hartree")
        
    except Exception as e:
        print(f"\nError in correlation calculation: {e}")
    
    # Visualize the system
    print("\nVisualizing anti-HeH+ system...")
    anti_heh_data.visualize(show_bonds=True)
    
    return {
        'molecule': anti_heh_data,
        'scf_results': scf_results,
        'correlation_results': corr_results if 'corr_results' in locals() else None
    }

def main():
    """Run all anti-HeH+ examples."""
    results = run_anti_heh_ion()
    
    # Save results plot
    plt.figure(figsize=(10, 6))
    plt.title("Anti-HeH+ Energy Levels")
    plt.xlabel("Orbital")
    plt.ylabel("Energy (Hartree)")
    
    # Plot positron energies if available
    if 'E_positron' in results['scf_results']:
        energies = results['scf_results']['E_positron']
        plt.plot(range(len(energies)), energies, 'ro-', label='Positron Orbitals')
        plt.legend()
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig(os.path.join('results', 'anti_heh_energies.png'))
    print(f"Results saved to results/anti_heh_energies.png")

if __name__ == "__main__":
    main() 