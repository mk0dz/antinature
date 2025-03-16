"""
Qiskit integration module for qantimatter quantum chemistry.

This module provides integration with Qiskit and Qiskit-Nature
for simulating qantimatter systems on quantum computers.
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
from .circuits import AntimatterCircuits, PositroniumCircuit
from .solver import PositroniumVQESolver
from .systems import AntimatterQuantumSystems
from .qqantimatter_solver import AntimatterQuantumSolver
from .vqe_solver import AntimatterVQESolver
from .ansatze import AntimatterAnsatz

# Define what should be exposed at package level
__all__ = [
    'QiskitNatureAdapter',
    'AntimatterCircuits',
    'PositroniumCircuit',
    'PositroniumVQESolver',
    'AntimatterQuantumSystems',
    'AntimatterQuantumSolver',
    'AntimatterVQESolver',
    'AntimatterAnsatz',
    'HAS_QISKIT'
]