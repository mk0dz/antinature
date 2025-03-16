"""
Qiskit integration module for quantimatter quantum chemistry.

This module provides integration with Qiskit and Qiskit-Nature
for simulating quantimatter systems on quantum computers.
"""

# Check if Qiskit is available
try:
    import qiskit
    import qiskit_nature
    from qiskit_algorithms import VQE, NumPyMinimumEigensolver
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit.primitives import Estimator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Warning: Qiskit or dependent packages not available. Qiskit integration will be disabled.")

# Always import the modules, but the classes will raise an error if Qiskit is not available
from .adapter import QiskitNatureAdapter
from .circuits import quantimatterCircuits, PositroniumCircuit
from .solver import PositroniumVQESolver
from .systems import quantimatterQuantumSystems
from .qquantimatter_solver import quantimatterQuantumSolver
from .vqe_solver import quantimatterVQESolver
from .ansatze import quantimatterAnsatz

# Define what should be exposed at package level
__all__ = [
    'QiskitNatureAdapter',
    'quantimatterCircuits',
    'PositroniumCircuit',
    'PositroniumVQESolver',
    'quantimatterQuantumSystems',
    'quantimatterQuantumSolver',
    'quantimatterVQESolver',
    'quantimatterAnsatz',
    'HAS_QISKIT'
]