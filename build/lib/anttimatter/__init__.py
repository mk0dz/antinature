"""
Antimatter Quantum Chemistry Framework
======================================

A high-performance framework for simulating antimatter systems, including
positronium, anti-hydrogen, and other exotic matter-antimatter configurations.

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
from .core.hamiltonian import AntimatterHamiltonian
from .core.scf import AntimatterSCF
from .core.correlation import AntimatterCorrelation
from .core.integral_engine import AntimatterIntegralEngine

# Specialized components
from .specialized.relativistic import RelativisticCorrection
from .specialized.annihilation import AnnihilationOperator
from .specialized.positronium import PositroniumSCF
from .specialized.visualization import AntimatterVisualizer

# Utilities
from .utils import create_antimatter_calculation

# Attempt to import optional quantum components
try:
    from .qiskit_integration import (
        AntimatterQuantumSolver,
        AntimatterQuantumSystems,
        AntimatterVQESolver
    )
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False



