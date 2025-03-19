# test_adapter.py

import os
import sys

import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qiskit.quantum_info import SparsePauliOp

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Qiskit not installed. Skipping tests.")
    sys.exit(0)

from antinature.qiskit_integration.adapter import (
    PositroniumAdapter,
    QiskitNatureAdapter,
)


def test_qiskit_nature_adapter():
    """Test the QiskitNatureAdapter class."""
    print("Testing QiskitNatureAdapter...")

    # Create adapter
    try:
        adapter = QiskitNatureAdapter()
        print("  Successfully created QiskitNatureAdapter instance")
    except Exception as e:
        print(f"  Error creating QiskitNatureAdapter: {str(e)}")

    # Test using PositroniumAdapter instead for different mappers
    print("\nTesting PositroniumAdapter with different mappers:")

    # Test different mapper types
    for mapper_type in ['jordan_wigner', 'parity']:
        print(f"\n  Testing with {mapper_type} mapper:")

        # Create adapter
        try:
            adapter = PositroniumAdapter(mapper_type=mapper_type)
            print(
                f"    Successfully created PositroniumAdapter with {mapper_type} mapper"
            )

            # Test creating positronium Hamiltonian
            try:
                problem, qubit_op = adapter.create_positronium_hamiltonian()
                print(
                    f"    Created positronium operator with {qubit_op.num_qubits} qubits"
                )
                print(f"    Operator has {len(qubit_op)} terms")
            except Exception as e:
                print(f"    Failed to create positronium Hamiltonian: {str(e)}")
        except Exception as e:
            print(
                f"    Error creating PositroniumAdapter with {mapper_type} mapper: {str(e)}"
            )


def test_positronium_adapter():
    """Test the PositroniumAdapter class."""
    print("\nTesting PositroniumAdapter...")

    # Create adapter
    adapter = PositroniumAdapter(mapper_type='jordan_wigner')

    # Test positronium Hamiltonian creation
    print("\n  Testing positronium Hamiltonian creation:")
    try:
        problem, qubit_op = adapter.create_positronium_hamiltonian()
        print(f"    Created positronium operator with {qubit_op.num_qubits} qubits")
        print(f"    Operator has {len(qubit_op)} terms")

        # Print the operator
        print("\n    Positronium operator terms:")
        for i, (pauli, coeff) in enumerate(zip(qubit_op.paulis, qubit_op.coeffs)):
            print(f"      {i+1}: {pauli.to_label()} * {coeff.real:.4f}")
    except Exception as e:
        print(f"    Failed to create positronium Hamiltonian: {str(e)}")

    # Test para vs ortho positronium
    print("\n  Testing para vs ortho positronium:")
    try:
        # Check if the method exists
        if hasattr(adapter, 'create_ortho_para_hamiltonian'):
            # Para-positronium (singlet)
            _, para_op = adapter.create_ortho_para_hamiltonian(is_ortho=False)

            # Ortho-positronium (triplet)
            _, ortho_op = adapter.create_ortho_para_hamiltonian(is_ortho=True)

            print("    Para-positronium operator terms:")
            for i, (pauli, coeff) in enumerate(zip(para_op.paulis, para_op.coeffs)):
                print(f"      {i+1}: {pauli.to_label()} * {coeff.real:.4f}")

            print("\n    Ortho-positronium operator terms:")
            for i, (pauli, coeff) in enumerate(zip(ortho_op.paulis, ortho_op.coeffs)):
                print(f"      {i+1}: {pauli.to_label()} * {coeff.real:.4f}")
        else:
            print(
                "    Method 'create_ortho_para_hamiltonian' not available in this version"
            )
    except Exception as e:
        print(f"    Failed to create ortho/para Hamiltonians: {str(e)}")

    # Test varying attraction strengths
    print("\n  Testing varying attraction strengths:")
    try:
        if hasattr(adapter, 'create_varying_attraction_hamiltonians'):
            attractions = [-0.5, -1.0, -1.5]
            hamiltonians = adapter.create_varying_attraction_hamiltonians(attractions)

            print(
                f"    Created {len(hamiltonians)} Hamiltonians with different attractions"
            )

            for attraction, op in hamiltonians.items():
                print(
                    f"      Attraction = {attraction}: {len(op)} terms, {op.num_qubits} qubits"
                )
        else:
            print(
                "    Method 'create_varying_attraction_hamiltonians' not available in this version"
            )

            # Alternative: Create multiple Hamiltonians directly
            attractions = [-0.5, -1.0, -1.5]
            print(
                f"    Creating {len(attractions)} Hamiltonians with different attractions directly:"
            )

            for attraction in attractions:
                _, op = adapter.create_positronium_hamiltonian(ep_attraction=attraction)
                print(
                    f"      Attraction = {attraction}: {len(op)} terms, {op.num_qubits} qubits"
                )
    except Exception as e:
        print(f"    Failed to create varying attraction Hamiltonians: {str(e)}")


if __name__ == "__main__":
    if HAS_QISKIT:
        test_qiskit_nature_adapter()
        test_positronium_adapter()
    else:
        print("Qiskit not available. Cannot run tests.")
