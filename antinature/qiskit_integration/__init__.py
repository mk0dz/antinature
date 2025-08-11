"""
Qiskit integration for quantum algorithms in antimatter systems.

This module provides quantum computing implementations for antinature
calculations using IBM's Qiskit framework.
"""
import warnings

# Check for Qiskit availability first
HAS_QISKIT = False
HAS_QISKIT_ALGORITHMS = False
HAS_QISKIT_NATURE = False

try:
    import qiskit
    from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
    HAS_QISKIT = True
    print(f"Qiskit available (version {qiskit.__version__})")
except ImportError:
    print("Warning: Qiskit not available. Limited functionality.")

try:
    import qiskit_algorithms
    HAS_QISKIT_ALGORITHMS = True
    print(f"Qiskit Algorithms available (version {qiskit_algorithms.__version__})")
except ImportError:
    print("Warning: Qiskit Algorithms not available. Limited functionality.")

try:
    import qiskit_nature
    HAS_QISKIT_NATURE = True
    print(f"Qiskit Nature available (version {qiskit_nature.__version__})")
except ImportError:
    print("Warning: Qiskit Nature not available. Limited functionality.")

# Import quantum circuit related classes
if HAS_QISKIT:
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        __all__ = ['QuantumCircuit', 'ParameterVector']
    except ImportError:
        __all__ = []
else:
    __all__ = []

# Import algorithms if available
if HAS_QISKIT_ALGORITHMS:
    try:
        from qiskit_algorithms import VQE, NumPyMinimumEigensolver
        from qiskit_algorithms.optimizers import COBYLA, SPSA
        __all__.extend(['VQE', 'NumPyMinimumEigensolver', 'COBYLA', 'SPSA'])
    except ImportError:
        pass

# Import primitives if available
if HAS_QISKIT:
    try:
        from qiskit.primitives import Estimator
        __all__.append('Estimator')
    except ImportError:
        pass

# Conditionally import module components only if dependencies are available
if HAS_QISKIT and HAS_QISKIT_ALGORITHMS:
    try:
        from .adapter import AntinatureQuantumAdapter
        __all__.append('AntinatureQuantumAdapter')
    except ImportError as e:
        warnings.warn(f"Could not import AntinatureQuantumAdapter: {e}")

    try:
        from .ansatze import AntinatureAnsatz
        __all__.append('AntinatureAnsatz')
    except ImportError as e:
        warnings.warn(f"Could not import AntinatureAnsatz: {e}")

    try:
        from .circuits import QuantumCircuitBuilder
        __all__.append('QuantumCircuitBuilder')
    except ImportError as e:
        warnings.warn(f"Could not import QuantumCircuitBuilder: {e}")

    # solver.py removed - functionality merged into antimatter_solver.py

    try:
        # Only import these if all dependencies are available
        from .vqe_solver import AntinatureVQESolver
        __all__.append('AntinatureVQESolver')
    except ImportError as e:
        warnings.warn(f"Could not import AntinatureVQESolver: {e}")

    try:
        from .antimatter_solver import AntinatureQuantumSolver
        __all__.append('AntinatureQuantumSolver')
    except ImportError as e:
        warnings.warn(f"Could not import AntinatureQuantumSolver: {e}")

    try:
        from .systems import AntinatureQuantumSystems
        __all__.append('AntinatureQuantumSystems')
    except ImportError as e:
        warnings.warn(f"Could not import AntinatureQuantumSystems: {e}")

else:
    warnings.warn("Qiskit and/or Qiskit Algorithms not available. Quantum functionality disabled.")
