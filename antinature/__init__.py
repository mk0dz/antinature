"""
antinature Quantum Chemistry Framework
======================================

A high-performance framework for simulating antinature systems, including
positronium, anti-hydrogen, and other exotic matter-antinature configurations.

The package includes specialized algorithms for positrons and positron-electron
interactions, relativistic corrections, and electron-positron annihilation processes.
"""

__version__ = "0.1.1"


# Define dummy classes to prevent NameError in importing code
class QuantumCircuit:
    """Dummy QuantumCircuit class to prevent NameError when importing with missing Qiskit."""

    def __init__(self, *args, **kwargs):
        pass


# Core components
from .core.basis import BasisSet, GaussianBasisFunction, MixedMatterBasis, PositronBasis
from .core.correlation import AntinatureCorrelation
from .core.hamiltonian import AntinatureHamiltonian
from .core.integral_engine import AntinatureIntegralEngine
from .core.molecular_data import MolecularData
from .core.scf import AntinatureSCF
from .specialized.annihilation import AnnihilationOperator
from .specialized.positronium import PositroniumSCF

# Specialized components
from .specialized.relativistic import RelativisticCorrection
from .specialized.visualization import AntinatureVisualizer

# Utilities
from .utils import create_antinature_calculation

# Set default flags
HAS_QISKIT = False

# Optional quantum components - with careful import handling
try:
    # First just check if qiskit is available
    import qiskit

    # Import necessary basic classes to avoid NameError
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        # We already defined a placeholder above
        pass

    # Import the integration module which will handle its own imports gracefully
    from .qiskit_integration import HAS_QISKIT

    # Only import specific components if Qiskit is actually available
    if HAS_QISKIT:
        try:
            from .qiskit_integration import (
                AntinatureQuantumSolver,
                AntinatureQuantumSystems,
                AntinatureVQESolver,
            )
        except ImportError:
            pass
except ImportError:
    HAS_QISKIT = False
