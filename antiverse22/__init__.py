"""
antiverse Quantum Chemistry Framework
======================================

A high-performance framework for simulating antiverse systems, including
positronium, anti-hydrogen, and other exotic matter-antiverse configurations.

The package includes specialized algorithms for positrons and positron-electron
interactions, relativistic corrections, and electron-positron annihilation processes.
"""

__version__ = "0.1.0"

# Core components
from .core.basis import (
    GaussianBasisFunction, 
    BasisSet, 
    PositronBasis, 
    MixedMatterBasis
)
from .core.molecular_data import MolecularData
from .core.hamiltonian import antiverseHamiltonian
from .core.scf import antiverseSCF
from .core.correlation import antiverseCorrelation
from .core.integral_engine import antiverseIntegralEngine

# Specialized components
from .specialized.relativistic import RelativisticCorrection
from .specialized.annihilation import AnnihilationOperator
from .specialized.positronium import PositroniumSCF
from .specialized.visualization import antiverseVisualizer

# Utilities
from .utils import create_antiverse_calculation

# Attempt to import optional quantum components
try:
    from .qiskit_integration import (
        antiverseQuantumSolver,
        antiverseQuantumSystems,
        antiverseVQESolver
    )
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False



