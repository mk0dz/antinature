"""
Specialized physics module for antinature quantum chemistry.

This module provides specialized physics components for antinature systems,
including annihilation operators and relativistic corrections.
"""

from .annihilation import AnnihilationOperator
from .positronium import PositroniumSCF
from .relativistic import RelativisticCorrection
from .visualization import plot_wavefunction, visualize_annihilation_density
