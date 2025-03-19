# test_circuits.py

import os
import sys

import numpy as np
import pytest

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# First, check if Qiskit is available
try:
    import qiskit

    print(f"Qiskit is available: {qiskit.__version__}")
    QISKIT_AVAILABLE = True
except ImportError as e:
    print(f"Qiskit not available: {str(e)}")
    QISKIT_AVAILABLE = False

# Skip the whole module if Qiskit isn't available
pytestmark = pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not available")

# Only import these if qiskit is available
if QISKIT_AVAILABLE:
    from antinature.qiskit_integration.circuits import (
        AntinatureCircuits,
        PositroniumCircuit,
    )


def test_circuits():
    """Test the enhanced AntinatureCircuits class."""
    print("\nTesting AntinatureCircuits...")

    # Create circuits generator
    circuits = AntinatureCircuits(n_electron_orbitals=2, n_positron_orbitals=2)

    # Test a basic ansatz
    print("\nCreating a basic ansatz:")
    try:
        circuit = circuits.create_antinature_ansatz(
            reps=1, entanglement='linear', rotation_blocks='x', initial_state='zero'
        )
        print(f"  Circuit depth: {circuit.depth()}")
        print(f"  Number of parameters: {circuit.num_parameters}")
        print(f"  Gate counts: {circuit.count_ops()}")

        # Add assertions
        assert circuit.num_qubits > 0, "Circuit should have at least 1 qubit"
        assert circuit.depth() > 0, "Circuit should have non-zero depth"
    except Exception as e:
        print(f"  Error creating ansatz: {str(e)}")
        assert False, f"Test failed with error: {e}"

    # Test positronium circuits
    print("\nTesting positronium-specific circuits:")
    try:
        circuit = circuits.create_positronium_circuit()
        print(f"  Circuit depth: {circuit.depth()}")
        print(f"  Gate counts: {circuit.count_ops()}")

        # Add assertions
        assert circuit.num_qubits > 0, "Circuit should have at least 1 qubit"
        assert circuit.depth() > 0, "Circuit should have non-zero depth"
    except Exception as e:
        print(f"  Error creating positronium circuit: {str(e)}")
        assert False, f"Test failed with error: {e}"


def test_positronium_circuit():
    """Test the PositroniumCircuit class."""
    print("\nTesting PositroniumCircuit...")

    # Create positronium circuit
    ps_circuit = PositroniumCircuit(n_electron_orbitals=1, n_positron_orbitals=1)

    # Test ground state
    try:
        circuit = ps_circuit.create_positronium_ground_state()
        print(f"  Ground state circuit depth: {circuit.depth()}")
        print(f"  Gate counts: {circuit.count_ops()}")

        # Add assertions
        assert circuit.num_qubits > 0, "Circuit should have at least 1 qubit"
        assert circuit.depth() > 0, "Circuit should have non-zero depth"
    except Exception as e:
        print(f"  Error creating ground state circuit: {str(e)}")
        assert False, f"Test failed with error: {e}"


# Run the tests if this script is called directly and Qiskit is available
if __name__ == "__main__" and QISKIT_AVAILABLE:
    test_circuits()
    test_positronium_circuit()
