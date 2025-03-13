"""
Core components for antimatter quantum chemistry.

This module contains the fundamental building blocks for 
antimatter quantum chemistry calculations:
- Hamiltonian constructors
- Integral calculation engines
- Basis set handling
- SCF procedures
- Post-SCF correlation methods
"""

from .hamiltonian import AntimatterHamiltonian
from .integral_engine import AntimatterIntegralEngine
from .basis import (
    GaussianBasisFunction, 
    BasisSet, 
    PositronBasis, 
    MixedMatterBasis, 
    BasisTransformation 
    )
from .scf import AntimatterSCF
from .correlation import AntimatterCorrelation
