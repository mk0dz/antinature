"""
Antimatter Quantum Chemistry Package
===================================

A framework for quantum chemistry calculations involving antimatter systems,
with a focus on electron-positron interactions.
"""

# Make core classes available at package level
from .antimatter_core import AntimatterQuantumChemistry, Molecule

# Make all modules available
from . import positron_basis
from . import antimatter_operators
from . import antimatter_integrals
from . import antimatter_scf
from . import antimatter_visualization

__version__ = '0.1.0'