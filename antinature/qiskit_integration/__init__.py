"""
Qiskit integration module for antinature quantum chemistry.

This module provides integration with Qiskit and Qiskit-Nature
for simulating antinature systems on quantum computers.
"""

# Initialize variables
HAS_QISKIT = False
HAS_SPARSE_PAULIOP = False
HAS_QISKIT_NATURE = False

# Check if Qiskit basic package is available
try:
    import qiskit

    print("Successfully imported Qiskit base package")
    HAS_QISKIT = True

    # Import Pauli operators
    try:
        from qiskit.quantum_info import Operator, Pauli, SparsePauliOp

        print("Successfully imported Pauli and SparsePauliOp")
        HAS_SPARSE_PAULIOP = True
    except ImportError:
        print("Warning: Pauli operators not available")

        # Create dummy classes for compatibility
        class SparsePauliOp:
            pass

        class Pauli:
            pass

        class Operator:
            pass

    # Import parameter vector
    try:
        from qiskit.circuit import ParameterVector

        print("Successfully imported ParameterVector")
    except ImportError:
        print("Warning: ParameterVector not available")

        # Create dummy class
        class ParameterVector:
            pass

    # Import Qiskit Nature
    try:
        import qiskit_nature
        from qiskit_algorithms import VQE, NumPyMinimumEigensolver
        from qiskit_algorithms.optimizers import COBYLA, SPSA

        print("Qiskit successfully imported.")
        HAS_QISKIT_NATURE = True

        # Check for estimator
        try:
            from qiskit.primitives import Estimator

            print("Primitives (Estimator) available.")
        except ImportError:
            print("Warning: Qiskit Estimator not available.")
    except ImportError:
        print("Warning: Qiskit Nature or algorithms not available.")

except ImportError:
    print(
        "Warning: Qiskit or dependent packages not available. Qiskit integration will be disabled."
    )

# Define placeholder classes in case imports fail
if not HAS_SPARSE_PAULIOP:

    class SparsePauliOp:
        pass


# Import our modules, but wrap in try-except to handle missing dependencies
if HAS_QISKIT:
    try:
        # Import the adapter directly - this is most important for tests
        # that check if the main package can be imported
        try:
            from .adapter import QiskitNatureAdapter
        except ImportError as e:
            print(f"Error importing QiskitNatureAdapter: {e}")

            # Define a dummy adapter for compatibility
            class QiskitNatureAdapter:
                def __init__(self, *args, **kwargs):
                    raise ImportError(
                        "QiskitNatureAdapter not available. Install required dependencies."
                    )

        # Attempt to import other components with graceful fallback
        try:
            from .ansatze import AntinatureAnsatz
            from .antimatter_solver import AntinatureQuantumSolver
            from .circuits import AntinatureCircuits, PositroniumCircuit
            from .solver import PositroniumVQESolver
            from .systems import AntinatureQuantumSystems
            from .vqe_solver import AntinatureVQESolver
        except ImportError as e:
            print(
                f"Warning: Some Qiskit integration modules could not be imported: {e}"
            )

        # Define what should be exposed at package level
        __all__ = [
            'QiskitNatureAdapter',
            'HAS_QISKIT',
            'HAS_SPARSE_PAULIOP',
            'HAS_QISKIT_NATURE',
        ]

        # Add optional components to __all__ if they're available
        try:
            AntinatureCircuits
            __all__.extend(['AntinatureCircuits', 'PositroniumCircuit'])
        except NameError:
            pass

        try:
            PositroniumVQESolver
            __all__.extend(['PositroniumVQESolver'])
        except NameError:
            pass

        try:
            AntinatureQuantumSystems
            __all__.extend(['AntinatureQuantumSystems'])
        except NameError:
            pass

        try:
            AntinatureQuantumSolver
            __all__.extend(['AntinatureQuantumSolver'])
        except NameError:
            pass

        try:
            AntinatureVQESolver
            __all__.extend(['AntinatureVQESolver'])
        except NameError:
            pass

        try:
            AntinatureAnsatz
            __all__.extend(['AntinatureAnsatz'])
        except NameError:
            pass

    except ImportError as e:
        print(f"Warning: Not all Qiskit integration modules could be imported: {e}")
        __all__ = ['HAS_QISKIT', 'HAS_SPARSE_PAULIOP', 'HAS_QISKIT_NATURE']
else:
    # Define minimal exports when Qiskit is not available
    __all__ = ['HAS_QISKIT', 'HAS_SPARSE_PAULIOP', 'HAS_QISKIT_NATURE']
