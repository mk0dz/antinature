"""
Qiskit integration module for antimatter quantum chemistry.

This module provides integration with Qiskit and Qiskit-Nature
for simulating antimatter systems on quantum computers.
"""

# Check if Qiskit is available
try:
    import qiskit
    import qiskit_nature
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

if HAS_QISKIT:
    from .adapter import QiskitNatureAdapter
    from .circuits import AntimatterCircuits
