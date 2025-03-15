import numpy as np
from typing import List, Tuple, Optional, Dict

class GaussianBasisFunction:
    """Optimized Gaussian basis function implementation."""
    
    def __init__(self, 
                 center: np.ndarray, 
                 exponent: float, 
                 angular_momentum: Tuple[int, int, int] = (0, 0, 0),
                 normalization: Optional[float] = None):
        self.center = center
        self.exponent = exponent
        self.angular_momentum = angular_momentum
        
        # Calculate normalization immediately if not provided
        self.normalization = normalization or self._calculate_normalization()
        
        # Cache common calculations
        self.alpha = exponent
        self.nx, self.ny, self.nz = angular_momentum
        
        # Pre-compute coefficient for polynomial part
        self.poly_coef = 1.0
    
    def _calculate_normalization(self) -> float:
        """Efficient normalization calculation with caching."""
        # Implementation with optimized calculation
        alpha = self.exponent
        nx, ny, nz = self.angular_momentum
        
        # Use vectorized operations where possible
        prefactor = (2 * alpha / np.pi) ** 0.75
        
        # Optimized normalization for angular momentum components
        def double_factorial(n):
            if n <= 0: return 1.0
            return np.prod(np.arange(n, 0, -2, dtype=float))
        
        x_norm = (4 * alpha) ** (nx / 2) / np.sqrt(double_factorial(2 * nx - 1)) if nx > 0 else 1.0
        y_norm = (4 * alpha) ** (ny / 2) / np.sqrt(double_factorial(2 * ny - 1)) if ny > 0 else 1.0
        z_norm = (4 * alpha) ** (nz / 2) / np.sqrt(double_factorial(2 * nz - 1)) if nz > 0 else 1.0
        
        return prefactor * x_norm * y_norm * z_norm
    
    def evaluate(self, r: np.ndarray) -> float:
        """
        Evaluate the basis function at position r with optimized calculation.
        """
        # Vectorized displacement calculation
        dr = r - self.center
        
        # Efficient polynomial calculation
        polynomial = dr[0]**self.nx * dr[1]**self.ny * dr[2]**self.nz if any(self.angular_momentum) else 1.0
        
        # Fast exponential calculation
        r_squared = np.sum(dr**2)
        exponential = np.exp(-self.alpha * r_squared)
        
        return self.normalization * polynomial * exponential

class MolecularData:
    """Container for molecular structure data."""
    
    def __init__(self, 
                 atoms: List[Tuple[str, np.ndarray]],
                 n_electrons: int, 
                 n_positrons: int = 0,
                 charge: int = 0):
        """
        Initialize molecular data.
        
        Parameters:
        -----------
        atoms : List[Tuple[str, np.ndarray]]
            List of (element, position) tuples
        n_electrons : int
            Number of electrons
        n_positrons : int
            Number of positrons
        charge : int
            Total charge of the system
        """
        self.atoms = atoms
        self.n_electrons = n_electrons
        self.n_positrons = n_positrons
        self.charge = charge
        
        # Create nuclei data with atomic charges
        self.nuclei = []
        for atom, position in atoms:
            # Get nuclear charge from element symbol
            charge = self._get_nuclear_charge(atom)
            self.nuclei.append((atom, charge, position))
    
    @staticmethod
    def _get_nuclear_charge(element: str) -> int:
        """Get nuclear charge from element symbol."""
        charges = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6,
            'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12,
            'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18
        }
        return charges.get(element, 0)
    
    def get_nuclear_repulsion_energy(self) -> float:
        """Calculate nuclear repulsion energy."""
        energy = 0.0
        for i, (_, charge_i, pos_i) in enumerate(self.nuclei):
            for j, (_, charge_j, pos_j) in enumerate(self.nuclei[i+1:], i+1):
                r_ij = np.linalg.norm(pos_i - pos_j)
                energy += charge_i * charge_j / r_ij
        return energy