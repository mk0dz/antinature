"""
antinature Quantum Chemistry Framework
======================================

A high-performance framework for simulating antinature systems, including
positronium, anti-hydrogen, and other exotic matter-antinature configurations.

The package includes specialized algorithms for positrons and positron-electron
interactions, relativistic corrections, and electron-positron annihilation processes.
"""

__version__ = "0.1.1"

# Core components
from .core.basis import (
    GaussianBasisFunction, 
    BasisSet, 
    PositronBasis, 
    MixedMatterBasis
)
from .core.molecular_data import MolecularData
from .core.hamiltonian import AntinatureHamiltonian
from .core.scf import AntinatureSCF
from .core.correlation import AntinatureCorrelation
from .core.integral_engine import AntinatureIntegralEngine

# Specialized components
from .specialized.relativistic import RelativisticCorrection
from .specialized.annihilation import AnnihilationOperator
from .specialized.positronium import PositroniumSCF
from .specialized.visualization import AntinatureVisualizer

# Utilities
from .utils import create_antinature_calculation

# # Attempt to import optional quantum components
# try:
#     from .qiskit_integration import (
#         AntinatureQuantumSolver,
#         AntinatureQuantumSystems,
#         AntinatureVQESolver
#     )
#     HAS_QISKIT = True
# except ImportError:
#     HAS_QISKIT = False



