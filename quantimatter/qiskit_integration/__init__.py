"""
Qiskit integration module for antiverse quantum chemistry.

This module provides integration with Qiskit and Qiskit-Nature
for simulating antiverse systems on quantum computers.
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
from .circuits import antiverseCircuits, PositroniumCircuit
from .solver import PositroniumVQESolver
from .systems import antiverseQuantumSystems
from .qantiverse_solver import antiverseQuantumSolver
from .vqe_solver import antiverseVQESolver
from .ansatze import antiverseAnsatz

# Define what should be exposed at package level
__all__ = [
    'QiskitNatureAdapter',
    'antiverseCircuits',
    'PositroniumCircuit',
    'PositroniumVQESolver',
    'antiverseQuantumSystems',
    'antiverseQuantumSolver',
    'antiverseVQESolver',
    'antiverseAnsatz',
    'HAS_QISKIT'
]