"""
Qiskit integration module for antimatter quantum chemistry.

This module provides integration with Qiskit and Qiskit-Nature
for simulating antimatter systems on quantum computers.
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
from .circuits import AntimatterCircuits
from .solver import PositroniumVQESolver
from .systems import AntimatterQuantumSystems
from .antimatter_solver import AntimatterQuantumSolver

# Make these classes available at the package level
__all__ = [
    'QiskitNatureAdapter',
    'AntimatterCircuits',
    'PositroniumVQESolver',
    'AntimatterQuantumSystems',
    'AntimatterQuantumSolver',
    'HAS_QISKIT'
]