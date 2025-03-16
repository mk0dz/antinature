# examples/antimatter_custom_ansatz.py

import numpy as np
import matplotlib.pyplot as plt
from anttimatter.qiskit_integration import AntimatterVQESolver
from anttimatter.qiskit_integration import AntimatterQuantumSystems

def main():
    """Test specialized antimatter ansätze."""
    print("=== Testing Specialized Antimatter Ansätze ===")
    
    # Create systems handler
    systems = AntimatterQuantumSystems()
    
    # Create solver
    solver = AntimatterVQESolver(
        optimizer_name='L_BFGS_B',  # Using L-BFGS-B for better convergence
        max_iterations=300,
        shots=2048
    )
    
    # Systems to test
    system_names = ['positronium', 'anti_hydrogen', 'positronium_molecule']
    ansatz_types = ['specialized', 'hardware_efficient']
    
    results = {}
    
    # Run calculations for each system with different ansatz types
    for system_name in system_names:
        results[system_name] = {}
        
        # Get Hamiltonian for this system
        if system_name == 'positronium':
            # Use positronium_molecule for positronium (simplified version)
            _, qubit_op, _ = systems.positronium_molecule()
        elif system_name == 'anti_hydrogen':
            _, qubit_op, _ = systems.anti_hydrogen(n_orbitals=3)
        elif system_name == 'positronium_molecule':
            _, qubit_op, _ = systems.positronium_molecule()
        
        # Test different ansatz types
        for ansatz_type in ansatz_types:
            print(f"\nSolving {system_name} with {ansatz_type} ansatz...")
            
            # Run VQE with this ansatz
            result = solver.solve_system(
                system_name=system_name,
                qubit_operator=qubit_op,
                ansatz_type=ansatz_type,
                reps=3
            )
            
            # Store result
            results[system_name][ansatz_type] = result
            
            # Print summary
            print(f"  Energy: {result['energy']:.6f} Hartree")
            print(f"  Optimizer iterations: {result['iterations']}")
            print(f"  Optimizer time: {result['optimizer_time']:.2f} seconds")
    
    # Create visualization comparing the different ansätze
    plt.figure(figsize=(10, 6))
    
    # Set up bar positions
    x = np.arange(len(system_names))
    width = 0.35
    
    # Plot energies for each ansatz type
    specialized_energies = [results[system][ansatz_types[0]]['energy'] for system in system_names]
    hw_efficient_energies = [results[system][ansatz_types[1]]['energy'] for system in system_names]
    
    plt.bar(x - width/2, specialized_energies, width, label='Specialized Ansatz')
    plt.bar(x + width/2, hw_efficient_energies, width, label='Hardware-Efficient Ansatz')
    
    # Add theoretical values
    theoretical = {
        'positronium': -0.25,
        'anti_hydrogen': -0.5,
        'positronium_molecule': -0.52
    }
    
    theoretical_values = [theoretical[system] for system in system_names]
    plt.plot(x, theoretical_values, 'ro-', label='Theoretical')
    
    # Add labels and legend
    plt.xlabel('Antimatter System')
    plt.ylabel('Energy (Hartree)')
    plt.title('Comparison of Antimatter Ansätze')
    plt.xticks(x, system_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('antimatter_ansatz_comparison.png')
    
    print("\nResults saved and visualization created.")
    return results

if __name__ == "__main__":
    main()