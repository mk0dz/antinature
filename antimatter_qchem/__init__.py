"""
Antimatter Quantum Chemistry Framework
======================================

A high-performance framework for simulating antimatter systems.
"""

__version__ = "0.1.0"

# Import core components after they're created
from .core.basis import (
    GaussianBasisFunction, 
    BasisSet, 
    PositronBasis, 
    MixedMatterBasis
)
from .core.molecular_data import MolecularData
# Add other imports as modules are created