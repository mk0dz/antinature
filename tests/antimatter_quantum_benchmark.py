# examples/antimatter_quantum_benchmark.py

import matplotlib.pyplot as plt
import numpy as np

# Use the package-level import instead of reaching into submodules
try:
    # Try importing from the package level first (recommended)
    from antimatter_qchem.qiskit_integration import AntimatterQuantumSolver
except ImportError:
    # Fallback to direct import
    from antimatter_qchem.qiskit_integration.antimatter_solver import AntimatterQuantumSolver

def main():
    """Run benchmark for all antimatter systems."""
    print("=== Antimatter Quantum Systems Benchmark ===")
    
    # Systems to test
    systems = ['positronium', 'anti_hydrogen', 'positronium_molecule', 'anti_helium']
    
    # Create solver
    solver = AntimatterQuantumSolver(
        optimizer_name='COBYLA',
        shots=1024,
        mapper_type='jordan_wigner'
    )
    
    # Results containers
    theoretical = []
    classical = []
    vqe = []
    classical_errors = []
    vqe_errors = []
    
    # Run all systems
    for system in systems:
        print(f"\nSolving {system}...")
        results = solver.solve(system, use_classical=True, use_vqe=True)
        
        # Store results
        theoretical.append(results['theoretical_energy'])
        classical.append(results.get('classical_energy', 0.0))
        vqe.append(results.get('vqe_energy', 0.0))
        classical_errors.append(results.get('classical_error', 0.0))
        vqe_errors.append(results.get('vqe_error', 0.0))
        
        # Print results
        print(f"  Theoretical energy: {results['theoretical_energy']:.6f} Hartree")
        if 'classical_energy' in results:
            print(f"  Classical energy:    {results['classical_energy']:.6f} Hartree")
            print(f"  Classical error:     {results['classical_error']:.6f} Hartree")
        if 'vqe_energy' in results:
            print(f"  VQE energy:          {results['vqe_energy']:.6f} Hartree")
            print(f"  VQE error:           {results['vqe_error']:.6f} Hartree")
            print(f"  VQE iterations:      {results['iterations']}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy comparison
    x = np.arange(len(systems))
    width = 0.25
    
    ax1.bar(x - width, theoretical, width, label='Theoretical')
    ax1.bar(x, classical, width, label='Classical')
    ax1.bar(x + width, vqe, width, label='VQE')
    
    ax1.set_ylabel('Energy (Hartree)')
    ax1.set_title('Antimatter Systems Energy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems)
    ax1.legend()
    
    # Error comparison
    ax2.bar(x - width/2, classical_errors, width, label='Classical Error')
    ax2.bar(x + width/2, vqe_errors, width, label='VQE Error')
    
    ax2.set_ylabel('Error (Hartree)')
    ax2.set_title('Solver Error Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('antimatter_quantum_benchmark.png')
    plt.show()
    
    return {
        'systems': systems,
        'theoretical': theoretical,
        'classical': classical,
        'vqe': vqe,
        'classical_errors': classical_errors,
        'vqe_errors': vqe_errors
    }

if __name__ == "__main__":
    main()