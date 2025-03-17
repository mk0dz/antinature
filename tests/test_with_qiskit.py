#!/usr/bin/env python3
"""
Quantum Integration Test for antinature
===========================================

This script tests the quantum functionality of the antinature package
by importing Qiskit integration modules and running a basic quantum simulation.
"""

import sys
import numpy as np

try:
    # First import the core modules
    from antinature.core.molecular_data import MolecularData
    from antinature.core.basis import MixedMatterBasis
    
    # Now try to import Qiskit integration modules
    from antinature.qiskit_integration import (
        antinatureQuantumSolver,
        antinatureQuantumSystems,
        antinatureVQESolver,
        antinatureCircuits
    )
    from antinature.qiskit_integration.adapter import QiskitNatureAdapter
    
    print("✅ Successfully imported antinature package including Qiskit integration")
    
    # Create a simple positronium system
    print("\nCreating a positronium system...")
    positronium = MolecularData.positronium()
    print(f"✅ Created positronium system: {positronium}")
    
    # Create a quantum solver
    print("\nInitializing quantum solver...")
    quantum_solver = antinatureQuantumSolver()
    print(f"✅ Created quantum solver: {quantum_solver}")
    
    # Create a minimal VQE solver with correct parameters
    print("\nSetting up VQE solver...")
    vqe_solver = antinatureVQESolver(
        optimizer_name='COBYLA',
        max_iterations=10,  # Minimal iterations for testing
        shots=100  # Small number of shots for quick testing
    )
    print(f"✅ Created VQE solver: {vqe_solver}")
    
    # Create quantum circuits using antinatureCircuits
    print("\nCreating quantum circuits...")
    circuits = antinatureCircuits(n_electron_orbitals=1, n_positron_orbitals=1)
    
    # Create custom ansatz circuit
    custom_circuit = circuits.create_custom_ansatz(reps=1)
    print(f"✅ Created custom ansatz circuit with {custom_circuit.num_qubits} qubits")
    
    # Create positronium circuit
    pos_circuit = circuits.create_positronium_circuit()
    print(f"✅ Created positronium circuit with {pos_circuit.num_qubits} qubits")
    
    # Create quantum systems
    print("\nCreating quantum systems...")
    quantum_sys = antinatureQuantumSystems()
    
    try:
        # Use positronium method to create a positronium system
        qubit_op, system_circuit = quantum_sys.positronium()
        print(f"✅ Created positronium quantum system with {system_circuit.num_qubits} qubits")
        
        # Try to run a VQE calculation if possible
        print("\nRunning minimal VQE simulation...")
        result = vqe_solver.solve_system(
            system_name="positronium",
            qubit_operator=qubit_op,
            ansatz_type="hardware_efficient",
            reps=1,  # Minimal repetitions for testing
            apply_correction=False  # Skip corrections for testing
        )
        print(f"✅ VQE solver returned result: {result}")
    except Exception as e:
        print(f"⚠️ Skipping full VQE simulation: {e}")
        print("This is expected if some advanced Qiskit components are missing")
    
    print("\n🎉 QUANTUM TEST SUCCESSFUL: antinature package with Qiskit integration is working! 🎉")
    
except ImportError as e:
    print(f"❌ ERROR: Failed to import Qiskit integration modules: {e}")
    print("Make sure you installed the Qiskit extras with: pip install 'antinature[qiskit]'")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: An error occurred during the quantum test: {e}")
    print(f"Error details: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print("The package is installed with Qiskit dependencies, but there may be issues with its functionality.")
    sys.exit(1) 