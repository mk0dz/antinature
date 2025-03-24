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

    HAS_QISKIT = True

    # Import Pauli operators
    try:
        from qiskit.quantum_info import Operator, Pauli, SparsePauliOp

        HAS_SPARSE_PAULIOP = True
    except ImportError:
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
    except ImportError:
        # Create dummy class
        class ParameterVector:
            pass

    # Import Qiskit Nature
    try:
        import qiskit_nature
        from qiskit_algorithms import VQE, NumPyMinimumEigensolver
        from qiskit_algorithms.optimizers import COBYLA, SPSA

        HAS_QISKIT_NATURE = True

        # Check for estimator
        try:
            from qiskit.primitives import Estimator
        except ImportError:
            pass
    except ImportError:
        pass

except ImportError:
    pass

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
        except ImportError:
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
        except ImportError:
            pass

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

    except ImportError:
        __all__ = ['HAS_QISKIT', 'HAS_SPARSE_PAULIOP', 'HAS_QISKIT_NATURE']
else:
    # Define minimal exports when Qiskit is not available
    __all__ = ['HAS_QISKIT', 'HAS_SPARSE_PAULIOP', 'HAS_QISKIT_NATURE']
