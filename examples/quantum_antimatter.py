#!/usr/bin/env python3
"""
Quantum Computing antinature Simulation Example
==============================================

This example demonstrates how to set up and run a quantum computing simulation
of positronium using the antinature package with Qiskit integration.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from qiskit import QuantumCircuit
    import qiskit.visualization as qiskit_vis
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Warning: This example requires Qiskit. Install with 'pip install antinature[qiskit]'")
    import sys
    sys.exit(1)

from antinature.core.molecular_data import MolecularData
from antinature.core.basis import MixedMatterBasis
from antinature.core.integral_engine import antinatureIntegralEngine
from antinature.core.hamiltonian import antinatureHamiltonian
from antinature.qiskit_integration import antinatureQuantumSolver, antinatureQuantumSystems
from antinature.qiskit_integration.ansatze import antinatureAnsatz

def main():
    print("Quantum Computing antinature Simulation Example")
    print("==============================================")
    
    # Step 1: Create positronium system
    print("\nCreating positronium molecular system...")
    positronium = MolecularData.positronium()
    print(f"System created: {positronium}")
    
    # Step 2: Create basis set and integrals
    print("\nSetting up basis and computing integrals...")
    basis = MixedMatterBasis()
    basis.create_positronium_basis(quality='minimal')  # Small basis for quantum simulation
    
    integral_engine = antinatureIntegralEngine()
    integrals = integral_engine.compute_all_integrals(positronium, basis)
    
    hamiltonian = antinatureHamiltonian()
    hamiltonian.build_hamiltonian(integrals, positronium, basis)
    print("Hamiltonian constructed")
    
    # Step 3: Setup quantum systems
    print("\nSetting up quantum system...")
    quantum_systems = antinatureQuantumSystems(mapper_type='jordan_wigner')
    operator = quantum_systems.get_positronium_operator(hamiltonian)
    
    print(f"Quantum operator created with {operator.num_qubits} qubits")
    
    # Step 4: Create and display quantum ansatz
    print("\nCreating specialized positronium ansatz...")
    ansatz = antinatureAnsatz.positronium_ansatz(reps=2)
    print(f"Ansatz created with {ansatz.num_qubits} qubits and {ansatz.num_parameters} parameters")
    
    # Display circuit
    print("\nQuantum circuit:")
    print(ansatz)
    
    # Step A: Optional - Draw the circuit
    circuit_fig = ansatz.draw(output='mpl')
    plt.savefig("positronium_circuit.png")
    print("Circuit visualization saved to 'positronium_circuit.png'")
    
    # Step 5: Set up quantum solver
    print("\nInitializing quantum solver...")
    quantum_solver = antinatureQuantumSolver(
        use_exact_solver=False,  # Use VQE
        optimizer_name='COBYLA',
        max_iterations=100,
        shots=1024
    )
    
    # Step 6: Run quantum simulation
    print("\nRunning quantum simulation (this may take a while)...")
    result = quantum_solver.solve_positronium(
        hamiltonian=hamiltonian,
        ansatz=ansatz,
        initial_point=None  # Use random initial parameters
    )
    
    # Step 7: Process and display results
    print("\nQuantum computation completed!")
    print(f"Ground state energy: {result['energy']:.8f} Hartree")
    print(f"VQE optimizer iterations: {result['optimizer_iterations']}")
    
    # Plot energy convergence
    if 'optimization_history' in result:
        plt.figure()
        plt.plot(result['optimization_history'])
        plt.xlabel('Iteration')
        plt.ylabel('Energy (Hartree)')
        plt.title('VQE Convergence')
        plt.savefig("vqe_convergence.png")
        print("Convergence plot saved to 'vqe_convergence.png'")
    
    print("\nSimulation completed successfully!")
    return result

if __name__ == "__main__":
    if HAS_QISKIT:
        results = main() 