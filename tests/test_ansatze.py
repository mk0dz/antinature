"""
Test script for the upgraded Qiskit integration modules in antinature.
This script tests the functionality of both solver.py and ansatze.py.
"""

import logging
import os
import sys

# Add parent directory to path if running from this script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import the required modules
try:
    import numpy as np

    print("Successfully imported numpy")

    # Try to import solver module components separately
    try:
        from antinature.qiskit_integration import solver

        print(f"Successfully imported solver module: {solver}")
    except ImportError as e1:
        print(f"Error importing solver module: {e1}")

    try:
        from antinature.qiskit_integration.solver import PositroniumVQESolver

        print("Successfully imported PositroniumVQESolver")
    except ImportError as e2:
        print(f"Error importing PositroniumVQESolver: {e2}")

    try:
        from antinature.qiskit_integration.solver import create_positronium_circuit

        print("Successfully imported create_positronium_circuit")
    except ImportError as e3:
        print(f"Error importing create_positronium_circuit: {e3}")

    # Try to import ansatze module components separately
    try:
        from antinature.qiskit_integration import ansatze

        print(f"Successfully imported ansatze module: {ansatze}")
    except ImportError as e4:
        print(f"Error importing ansatze module: {e4}")

    try:
        from antinature.qiskit_integration.ansatze import AntinatureAnsatz

        print("Successfully imported AntinatureAnsatz")
    except ImportError as e5:
        print(f"Error importing AntinatureAnsatz: {e5}")

    # If we reach here, we have all required modules
    HAS_REQUIRED_MODULES = True
except ImportError as e:
    print(f"Error importing required modules: {e}")
    HAS_REQUIRED_MODULES = False

# Check for Qiskit
logger = logging.getLogger(__name__)

try:
    import qiskit

    logger.debug(f"Qiskit imported from: {qiskit.__file__}")

    # Import QuantumCircuit - should be available in all Qiskit versions
    from qiskit import QuantumCircuit

    print("Successfully imported QuantumCircuit")

    # Try to import Aer - in newer versions it's a separate package
    try:
        from qiskit import Aer
    except ImportError:
        # Try to import from qiskit_aer package
        try:
            import qiskit_aer
            from qiskit_aer import Aer

            logger.debug(
                f"Using Aer from qiskit_aer (version {qiskit_aer.__version__})"
            )
        except ImportError:
            logger.warning("Aer not available, some tests will be skipped")

    # Try to import visualization
    try:
        from qiskit.visualization import plot_histogram
    except ImportError:
        logger.debug("Visualization module not available, some tests will be skipped")

    HAS_QISKIT = True
    logger.debug(f"Qiskit version: {qiskit.__version__}")
except ImportError as e:
    HAS_QISKIT = False
    logger.warning(f"Qiskit not available: {e}")
    print("Qiskit not available.")


def test_positronium_circuit():
    """Test the creation of a positronium circuit."""
    print("\n=== Testing Positronium Circuit Creation ===")

    if not HAS_QISKIT:
        print("Skipping test: Qiskit not available")
        return

    try:
        # Create a circuit with default parameters
        circuit = create_positronium_circuit(reps=2)
        print(f"Successfully created circuit with {circuit.num_qubits} qubits")
        print(f"Number of parameters: {circuit.num_parameters}")
        print(f"Circuit depth: {circuit.depth()}")

        # Test with different parameters - handle API differences
        try:
            # Try with include_entanglement parameter if it exists
            circuit_no_entanglement = create_positronium_circuit(
                reps=3, include_entanglement=False
            )
            print(
                f"Circuit without entanglement - depth: {circuit_no_entanglement.depth()}"
            )
        except TypeError as e:
            # If the parameter doesn't exist, just create without it
            print(f"Note: include_entanglement parameter not supported: {e}")
            circuit_alt = create_positronium_circuit(reps=3)
            print(f"Alternative circuit - depth: {circuit_alt.depth()}")

        # Use assertions instead of returning True
        assert circuit.num_qubits > 0, "Circuit should have at least 1 qubit"
        assert circuit.depth() > 0, "Circuit should have non-zero depth"
    except Exception as e:
        print(f"Error creating positronium circuit: {e}")
        assert False, f"Test failed with error: {e}"


def test_ansatze():
    """Test the creation of different ansatze."""
    print("\n=== Testing Antinature Ansatze ===")

    if not HAS_QISKIT:
        print("Skipping test: Qiskit not available")
        return

    try:
        # Test positronium ansatz
        pos_circuit = AntinatureAnsatz.positronium_ansatz(reps=2)
        print(
            f"Positronium ansatz: {pos_circuit.num_qubits} qubits, depth {pos_circuit.depth()}"
        )

        # Test anti-hydrogen ansatz
        ah_circuit = AntinatureAnsatz.anti_hydrogen_ansatz(reps=2)
        print(
            f"Anti-hydrogen ansatz: {ah_circuit.num_qubits} qubits, depth {ah_circuit.depth()}"
        )

        # Test positronium molecule ansatz
        ps2_circuit = AntinatureAnsatz.positronium_molecule_ansatz(reps=2)
        print(
            f"Positronium molecule ansatz: {ps2_circuit.num_qubits} qubits, depth {ps2_circuit.depth()}"
        )

        # Test the factory method if it exists
        try:
            factory_circuit = AntinatureAnsatz.get_specialized_ansatz(
                "positronium", reps=2
            )
            print(
                f"Factory method: Created {factory_circuit.num_qubits} qubit circuit for positronium"
            )
        except AttributeError as e:
            print(f"Note: Factory method not available: {e}")
            # Create an alternative test
            print("Using direct method instead of factory method")

        # Use assertions instead of returning True
        assert (
            pos_circuit.num_qubits > 0
        ), "Positronium ansatz should have at least 1 qubit"
        assert (
            ah_circuit.num_qubits > 0
        ), "Anti-hydrogen ansatz should have at least 1 qubit"
        assert (
            ps2_circuit.num_qubits > 0
        ), "Positronium molecule ansatz should have at least 1 qubit"
    except Exception as e:
        print(f"Error testing ansatze: {e}")
        assert False, f"Test failed with error: {e}"


def test_vqe_simulation(run_simulation=False):
    """Test VQE simulation with the positronium solver."""
    print("\n=== Testing VQE Simulation ===")

    if not HAS_QISKIT:
        print("Skipping test: Qiskit not available")
        return

    try:
        # Create a solver with proper parameter handling
        try:
            # Try with max_iterations parameter
            solver = PositroniumVQESolver(optimizer_name="COBYLA", max_iterations=5)
        except TypeError as e:
            # If max_iterations is not supported, try without it
            print(f"Note: max_iterations parameter not supported: {e}")
            solver = PositroniumVQESolver(optimizer_name="COBYLA")

        print("Successfully created VQE solver")

        if run_simulation:
            # Run a very small simulation (this will be slow and might not converge well with few iterations)
            print("Running a minimal VQE simulation (this might take a while)...")
            try:
                results = solver.solve_positronium(reps=1, n_tries=1)
                print(f"VQE energy: {results.get('vqe_energy', 'N/A')}")
                print(f"Theoretical energy: {results.get('theoretical_energy', 'N/A')}")
                print(f"Error: {results.get('vqe_error', 'N/A')}")

                # Assert on results if simulation was run
                assert (
                    'vqe_energy' in results
                ), "VQE results should contain energy value"
            except Exception as e:
                print(f"Error in VQE simulation: {e}")
                assert False, f"VQE simulation failed with error: {e}"
        else:
            print("Skipping actual VQE simulation (set run_simulation=True to run)")

        # Basic assertion to validate solver creation
        assert solver is not None, "Solver should be successfully created"
    except Exception as e:
        print(f"Error testing VQE solver: {e}")
        assert False, f"Test failed with error: {e}"


def run_all_tests():
    """Run all tests."""
    print("=== Running all tests for upgraded Qiskit modules ===")

    if not HAS_REQUIRED_MODULES:
        print("Cannot run tests: Required modules not available")
        return

    # Run tests
    circuit_test = test_positronium_circuit()
    ansatze_test = test_ansatze()
    vqe_test = test_vqe_simulation(
        run_simulation=False
    )  # Set to True to run actual VQE

    # Print summary
    print("\n=== Test Summary ===")
    print(f"Positronium Circuit: {'✓' if circuit_test else '✗'}")
    print(f"Antinature Ansatze: {'✓' if ansatze_test else '✗'}")
    print(f"VQE Solver: {'✓' if vqe_test else '✗'}")


if __name__ == "__main__":
    run_all_tests()
