"""
Antimatter Quantum Chemistry Framework
=====================================

A specialized extension of Qiskit-Nature for simulating antimatter systems on quantum computers.
"""
# Core imports
from .core.hamiltonian import AntimatterHamiltonian
from .core.integral_engine import AntimatterIntegralEngine
from .core.basis import GaussianBasisFunction, BasisSet, PositronBasis, MixedMatterBasis, BasisTransformation
from .core.scf import AntimatterSCF
from .core.correlation import AntimatterCorrelation





# Specialized imports
from .specialized.annihilation import AnnihilationOperator
from .specialized.relativistic import RelativisticCorrection

# Qiskit & Quantum computing integration

from .qiskit_integration.adapter import QiskitNatureAdapter
from .qiskit_integration.circuits import AntimatterCircuits

# Validation imports
from .validation.validator import AntimatterValidator
from .validation.test_suite import TestSuite

# framework

from .qchem_test import * 

__version__ = "0.1.0"






