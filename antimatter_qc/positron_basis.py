"""
Mock Positron Basis Module for Testing
=====================================

This is a simplified version used for testing the framework structure.
"""

import numpy as np

class GaussianPositronBasis:
    """Mock implementation of the GaussianPositronBasis class."""
    
    def __init__(self, molecule=None, basis_type='positron-minimal'):
        """Initialize the basis set."""
        self.molecule = molecule
        self.basis_type = basis_type
        self.basis_functions = []
        
        # Create some dummy basis functions
        if molecule is not None:
            self._create_dummy_basis()
    
    def _create_dummy_basis(self):
        """Create dummy basis functions for testing."""
        # Add a few basis functions
        for i, atom in enumerate(self.molecule.atoms):
            self.basis_functions.append({
                'type': 'gaussian',
                'center': np.array(atom['position']),
                'exponent': 0.5,
                'powers': (0, 0, 0)
            })
            self.basis_functions.append({
                'type': 'gaussian',
                'center': np.array(atom['position']),
                'exponent': 1.0,
                'powers': (0, 0, 0)
            })
    
    def get_num_basis_functions(self):
        """Get the number of basis functions."""
        return len(self.basis_functions)
    
    def evaluate(self, idx, point):
        """Evaluate a basis function at a given point."""
        if idx >= len(self.basis_functions):
            return 0.0
            
        func = self.basis_functions[idx]
        center = func['center']
        exponent = func['exponent']
        
        # Simple Gaussian function
        r_squared = np.sum((np.array(point) - center)**2)
        return np.exp(-exponent * r_squared)