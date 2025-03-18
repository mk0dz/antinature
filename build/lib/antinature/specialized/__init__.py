"""
Specialized physics module for antinature quantum chemistry.

This module provides specialized physics components for antinature systems,
including annihilation operators and relativistic corrections.
"""

from .annihilation import AnnihilationOperator
from .relativistic import RelativisticCorrection
from .visualization import visualize_annihilation_density, plot_wavefunction
from .positronium import PositroniumSCF
