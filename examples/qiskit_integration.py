# example/qiskit_integration.py

import numpy as np
from antimatter_qchem import (
    MolecularData, create_antimatter_calculation, run_antimatter_calculation
)
from antimatter_qchem.qiskit_integration import (
    QiskitNatureAdapter, AntimatterCircuits
)

def main():
    """Example of Qiskit integration for antimatter calculations."""
    
    print("Performing antimatter calculation with Qiskit...")
    
    # Create positronium molecular data
    positronium = MolecularData(
        atoms=[('H', np.array([0.0, 0.0, 0.0]))],
        n_electrons=1,
        n_positrons=1,
        charge=0
    )
    
    # Create and run classical calculation first
    print("\nClassical calculation:")
    classical_calc = create_antimatter_calculation(positronium)
    classical_results = run_antimatter_calculation(classical_calc)
    
    print(f"Classical energy: {classical_results['scf']['energy']:.10f} Hartree")
    
    # Create Qiskit adapter
    adapter = QiskitNatureAdapter(mapper_type='jordan_wigner')
    
    # Convert Hamiltonian to Qiskit format
    qiskit_ops = adapter.convert_hamiltonian(
        classical_calc['hamiltonian_matrices'],
        classical_calc['molecular_data']
    )
    
    # Set up VQE
    vqe_setup = adapter.setup_vqe(
        qiskit_ops,
        optimizer_name='COBYLA',
        ansatz_type='efficient_su2',
        ansatz_depth=2
    )
    
    # Run quantum calculation
    print("\nQuantum calculation:")
    quantum_results = adapter.run_ground_state_calculation(
        qiskit_ops,
        vqe_setup,
        use_classical_solver=True  # For testing, use classical solver
    )
    
    print(f"Quantum energy: {quantum_results['total_energy']:.10f} Hartree")
    
    # Create specialized circuit for positronium
    print("\nCreating specialized positronium circuit:")
    circuits = AntimatterCircuits(n_electron_qubits=2, n_positron_qubits=2)
    
    # Create particle-preserving ansatz
    ansatz = circuits.particle_preserving_ansatz(1, 1, reps=2)
    
    print(f"Circuit depth: {ansatz.depth()}")
    print(f"Number of parameters: {ansatz.num_parameters}")
    
    # Print the circuit
    print(ansatz)

if __name__ == "__main__":
    main()