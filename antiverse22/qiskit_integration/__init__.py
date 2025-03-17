"""
Qiskit integration module for antiverse2 quantum chemistry.

This module provides integration with Qiskit and Qiskit-Nature
for simulating antiverse2 systems on quantum computers.
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
from .circuits import antiverse2Circuits, PositroniumCircuit
from .solver import PositroniumVQESolver
from .systems import antiverse2QuantumSystems
from .antiverse2_solver import antiverse2QuantumSolver
from .vqe_solver import antiverse2VQESolver
from .ansatze import antiverse2Ansatz

# Define what should be exposed at package level
__all__ = [
    'QiskitNatureAdapter',
    'antiverse2Circuits',
    'PositroniumCircuit',
    'PositroniumVQESolver',
    'antiverse2QuantumSystems',
    'antiverse2QuantumSolver',
    'antiverse2VQESolver',
    'antiverse2Ansatz',
    'HAS_QISKIT'
]