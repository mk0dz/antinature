"""
Core components for antiverse quantum chemistry.

This module includes the fundamental building blocks for antiverse chemistry simulations:
- Basis sets for electrons and positrons
- Molecular data structures
- Hamiltonian construction
- SCF (Self-Consistent Field) solvers
- Correlation methods
- Integral calculation engines
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import basis components
from .basis import (
    GaussianBasisFunction, 
    BasisSet, 
    PositronBasis, 
    MixedMatterBasis
)

# Import core computational components
from .molecular_data import MolecularData
from .hamiltonian import antiverseHamiltonian
from .scf import antiverseSCF
from .correlation import antiverseCorrelation
from .integral_engine import antiverseIntegralEngine
# Add imports for other modules when they're created