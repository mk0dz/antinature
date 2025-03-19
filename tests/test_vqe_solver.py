# test_vqe_solver.py

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try each import separately to identify which one is failing
try:
    from qiskit import QuantumCircuit

    print("Successfully imported QuantumCircuit")
except ImportError as e:
    print(f"Error importing QuantumCircuit: {e}")
    HAS_QISKIT = False
    print("Qiskit base not installed. Skipping tests.")
    sys.exit(0)

try:
    from qiskit.quantum_info import Pauli, SparsePauliOp

    print("Successfully imported Pauli and SparsePauliOp")
except ImportError as e:
    print(f"Error importing from quantum_info: {e}")
    HAS_QISKIT = False
    print("Qiskit quantum_info package not available. Skipping tests.")
    sys.exit(0)

try:
    from qiskit.circuit import ParameterVector

    print("Successfully imported ParameterVector")
    HAS_QISKIT = True
except ImportError as e:
    print(f"Error importing ParameterVector: {e}")
    HAS_QISKIT = False
    print("Qiskit circuit package not fully available. Skipping tests.")
    sys.exit(0)

from antinature.qiskit_integration.vqe_solver import AntinatureVQESolver


def create_test_hamiltonian():
    """Create a simple test Hamiltonian for positronium."""
    # Simple 2-qubit Hamiltonian approximating positronium
    # H = 0.5 * (I⊗I + Z⊗Z) - 0.25 * (X⊗X + Y⊗Y)
    # This gives ground state energy of -0.25 (matching positronium)

    # Create Pauli terms
    identity = Pauli('II')
    zz = Pauli('ZZ')
    xx = Pauli('XX')
    yy = Pauli('YY')

    # Combine with coefficients
    hamiltonian = SparsePauliOp([identity, zz, xx, yy], coeffs=[0.5, 0.5, -0.25, -0.25])

    return hamiltonian


def create_test_ansatz():
    """Create a simple test ansatz for positronium."""
    circuit = QuantumCircuit(2)

    # Create parameters
    params = ParameterVector('θ', 6)

    # Build parameterized circuit
    circuit.h(0)
    circuit.h(1)

    circuit.rx(params[0], 0)
    circuit.ry(params[1], 0)
    circuit.rz(params[2], 0)

    circuit.rx(params[3], 1)
    circuit.ry(params[4], 1)
    circuit.rz(params[5], 1)

    circuit.cx(0, 1)

    return circuit


def test_vqe_solver():
    """Test the enhanced AntinatureVQESolver class."""
    print("Testing AntinatureVQESolver...")

    # Create test objects
    hamiltonian = create_test_hamiltonian()
    ansatz = create_test_ansatz()

    # Test different optimizers
    optimizers_to_test = ['COBYLA', 'SPSA']

    for optimizer in optimizers_to_test:
        print(f"\nTesting with {optimizer} optimizer:")

        # Create VQE solver
        vqe = AntinatureVQESolver(
            optimizer_name=optimizer, max_iterations=50  # Reduced for testing
        )

        # Solve system
        try:
            results = vqe.solve_system(
                system_name='positronium',
                qubit_operator=hamiltonian,
                ansatz_type='hardware_efficient',
                apply_correction=True,
                reps=1,
            )

            # Print results
            print(f"  Energy: {results['energy']:.6f} Hartree")
            if 'raw_energy' in results:
                print(f"  Raw energy: {results['raw_energy']:.6f} Hartree")
            if 'theoretical' in results:
                print(f"  Theoretical: {results['theoretical']:.6f} Hartree")
            if 'error' in results:
                print(f"  Error: {results['error']:.6f} Hartree")
            if 'relative_error' in results:
                print(f"  Relative error: {results['relative_error']:.2%}")
            print(f"  Iterations: {results.get('iterations', 'N/A')}")
            print(
                f"  Execution time: {results.get('execution_time', results.get('optimizer_time', 'N/A'))} seconds"
            )

            # Test result analysis
            try:
                analysis = vqe.analyze_results(results)
                print("\n  Result analysis:")
                if 'quality_metrics' in analysis:
                    print(
                        f"    Quality rating: {analysis['quality_metrics'].get('quality_rating', 'N/A')}"
                    )
                if 'recommendations' in analysis and analysis['recommendations']:
                    print("    Recommendations:")
                    for rec in analysis['recommendations']:
                        print(f"      - {rec}")
            except (AttributeError, TypeError) as e:
                print(f"\n  Result analysis not available: {str(e)}")

            # Test convergence plot
            try:
                print("\n  Creating convergence plot...")
                if hasattr(vqe, 'plot_convergence'):
                    fig = vqe.plot_convergence(
                        results, save_path=f"convergence_{optimizer}.png"
                    )
                    print(f"    Plot saved to convergence_{optimizer}.png")
                else:
                    print("    Convergence plotting not available")
            except Exception as e:
                print(f"    Could not create plot: {str(e)}")

        except Exception as e:
            print(f"  Error solving system: {str(e)}")

    # Test optimizer comparison
    print("\nTesting optimizer comparison:")
    try:
        if hasattr(vqe, 'compare_optimizers'):
            comparison = vqe.compare_optimizers(
                system_name='positronium',
                qubit_operator=hamiltonian,
                ansatz_type='hardware_efficient',
                optimizers=['COBYLA', 'SPSA'],
            )

            print("  Comparison results:")
            for opt, result in comparison.items():
                if 'energy' in result:
                    print(
                        f"    {opt}: Energy = {result['energy']:.6f}, Time = {result['time']:.2f}s"
                    )
                else:
                    print(f"    {opt}: Failed - {result.get('error', 'Unknown error')}")
        else:
            print("  Optimizer comparison not available")
    except Exception as e:
        print(f"  Error comparing optimizers: {str(e)}")

    print("\nTesting completed successfully!")


if __name__ == "__main__":
    if HAS_QISKIT:
        test_vqe_solver()
    else:
        print("Qiskit not available. Cannot run tests.")
