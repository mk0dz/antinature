"""
Qiskit integration module for antinature quantum chemistry.

This module provides integration with Qiskit and Qiskit-Nature
for simulating antinature systems on quantum computers.
"""

# Check if Qiskit is available
try:
    import qiskit
    import qiskit_nature
    from qiskit.primitives import Estimator
    from qiskit_algorithms import VQE, NumPyMinimumEigensolver
    from qiskit_algorithms.optimizers import COBYLA, SPSA

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print(
        "Warning: Qiskit or dependent packages not available. Qiskit integration will be disabled."
    )

# Always import the modules, but the classes will raise an error if Qiskit is not available
from .adapter import QiskitNatureAdapter
from .ansatze import AntinatureAnsatz
from .antimatter_solver import AntinatureQuantumSolver
from .circuits import AntinatureCircuits, PositroniumCircuit
from .solver import PositroniumVQESolver
from .systems import AntinatureQuantumSystems
from .vqe_solver import AntinatureVQESolver

# Define what should be exposed at package level
__all__ = [
    'QiskitNatureAdapter',
    'AntinatureCircuits',
    'PositroniumCircuit',
    'PositroniumVQESolver',
    'AntinatureQuantumSystems',
    'AntinatureQuantumSolver',
    'AntinatureVQESolver',
    'AntinatureAnsatz',
    'HAS_QISKIT',
]
