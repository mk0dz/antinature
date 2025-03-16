"""
quantimatter Quantum Chemistry Framework
======================================

A high-performance framework for simulating quantimatter systems, including
positronium, anti-hydrogen, and other exotic matter-quantimatter configurations.

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
from .core.hamiltonian import quantimatterHamiltonian
from .core.scf import quantimatterSCF
from .core.correlation import quantimatterCorrelation
from .core.integral_engine import quantimatterIntegralEngine

# Specialized components
from .specialized.relativistic import RelativisticCorrection
from .specialized.annihilation import AnnihilationOperator
from .specialized.positronium import PositroniumSCF
from .specialized.visualization import quantimatterVisualizer

# Utilities
from .utils import create_quantimatter_calculation

# Attempt to import optional quantum components
try:
    from .qiskit_integration import (
        quantimatterQuantumSolver,
        quantimatterQuantumSystems,
        quantimatterVQESolver
    )
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False



