"""
Core components for antinature quantum chemistry.

This module includes the fundamental building blocks for antinature chemistry simulations:
- Basis sets for electrons and positrons
- Molecular data structures
- Hamiltonian construction
- SCF (Self-Consistent Field) solvers
- Correlation methods
- Integral calculation engines
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import basis components
from .basis import BasisSet, GaussianBasisFunction, MixedMatterBasis, PositronBasis
from .correlation import AntinatureCorrelation
from .hamiltonian import AntinatureHamiltonian
from .integral_engine import AntinatureIntegralEngine

# Import core computational components
from .molecular_data import MolecularData
from .scf import AntinatureSCF

# Add imports for other modules when they're created
