# antimatter_qchem/core/basis.py

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


class BasisSet:
    """Efficient basis set implementation for electrons."""
    
    def __init__(self, basis_functions: List[GaussianBasisFunction] = None):
        """
        Initialize basis set with given basis functions.
        
        Parameters:
        -----------
        basis_functions : List[GaussianBasisFunction], optional
            List of basis functions
        """
        self.basis_functions = basis_functions or []
        self.n_basis = len(self.basis_functions)
        
        # Precompute some frequently used data
        self._cache = {}
    
    def add_function(self, basis_function: GaussianBasisFunction):
        """Add a basis function to the set."""
        self.basis_functions.append(basis_function)
        self.n_basis = len(self.basis_functions)
        
        # Clear cache when basis set changes
        self._cache = {}
    
    def add_functions(self, basis_functions: List[GaussianBasisFunction]):
        """Add multiple basis functions to the set."""
        self.basis_functions.extend(basis_functions)
        self.n_basis = len(self.basis_functions)
        
        # Clear cache when basis set changes
        self._cache = {}
    
    def create_for_atom(self, element: str, position: np.ndarray, quality: str = 'standard'):
        """
        Create basis functions for a given atom at specified position.
        
        Parameters:
        -----------
        element : str
            Element symbol
        position : np.ndarray
            Atomic position
        quality : str
            Basis set quality ('minimal', 'standard', 'extended')
        """
        # Get basis parameters based on quality
        basis_params = self._get_basis_params(element, quality)
        
        if not basis_params:
            raise ValueError(f"No basis parameters available for {element} with quality {quality}")
        
        # Create basis functions from parameters
        new_functions = []
        for shell_type, exponents, coefficients in basis_params:
            for exp, coef in zip(exponents, coefficients):
                if shell_type == 's':
                    new_functions.append(
                        GaussianBasisFunction(
                            center=position,
                            exponent=exp,
                            angular_momentum=(0, 0, 0),
                            normalization=coef
                        )
                    )
                elif shell_type == 'p':
                    # Add p-type functions (px, py, pz)
                    for am in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        new_functions.append(
                            GaussianBasisFunction(
                                center=position,
                                exponent=exp,
                                angular_momentum=am,
                                normalization=coef
                            )
                        )
        
        self.add_functions(new_functions)
    
    def _get_basis_params(self, element: str, quality: str) -> List[Tuple]:
        """
        Get basis set parameters for a given element and quality.
        """
        # Define basis parameters for common elements
        basis_params = {
            'minimal': {
                'H': [
                    ('s', [0.5], [1.0])  # Minimal H basis with single s function
                ],
                'He': [
                    ('s', [0.8], [1.0])  # Minimal He basis
                ],
                # Add more elements as needed
            },
            'standard': {
                'H': [
                    ('s', [13.0, 1.96, 0.4446], [0.0334, 0.2347, 0.8137])  # STO-3G
                ],
                'He': [
                    ('s', [38.4, 5.77, 1.24], [0.0236, 0.1557, 0.4685])
                ],
                # Add more elements with standard basis
            },
            'extended': {
                'H': [
                    ('s', [33.8650, 5.0947, 1.1587, 0.3258], [0.0254, 0.1907, 0.5523, 0.5672]),
                    ('p', [1.0], [1.0])  # Add p functions for polarization
                ],
                # Add more elements with extended basis
            }
        }
        
        return basis_params.get(quality, {}).get(element, [])


class PositronBasis(BasisSet):
    """Specialized basis set for positrons."""
    
    def __init__(self, basis_functions: List[GaussianBasisFunction] = None):
        """
        Initialize positron basis set.
        
        Parameters:
        -----------
        basis_functions : List[GaussianBasisFunction], optional
            List of basis functions
        """
        super().__init__(basis_functions)
        
        # Positron-specific parameters
        self.is_positron_basis = True
        self.scaling_factor = 0.5  # Typical scaling factor for positron basis relative to electron basis
    
    def create_for_atom(self, element: str, position: np.ndarray, quality: str = 'standard'):
        """
        Create basis functions for positrons interacting with a given atom.
        
        Parameters:
        -----------
        element : str
            Element symbol
        position : np.ndarray
            Atomic position
        quality : str
            Basis set quality ('minimal', 'standard', 'extended')
        """
        # For positrons, we typically need more diffuse functions
        # We'll scale the electron basis and add more diffuse functions
        
        # First get electron basis parameters
        electron_params = self._get_basis_params(element, quality)
        
        if not electron_params:
            raise ValueError(f"No basis parameters available for {element} with quality {quality}")
        
        # Create modified parameters for positrons
        positron_params = []
        for shell_type, exponents, coefficients in electron_params:
            # Scale exponents for positrons (make more diffuse)
            scaled_exponents = [exp * self.scaling_factor for exp in exponents]
            
            # Add to positron parameters
            positron_params.append((shell_type, scaled_exponents, coefficients))
        
        # Add extra diffuse functions
        smallest_exp = min([exp for _, exponents, _ in positron_params for exp in exponents])
        extra_diffuse = smallest_exp * 0.1  # Extra diffuse function
        
        positron_params.append(('s', [extra_diffuse], [1.0]))
        
        # Create basis functions from parameters
        new_functions = []
        for shell_type, exponents, coefficients in positron_params:
            for exp, coef in zip(exponents, coefficients):
                if shell_type == 's':
                    new_functions.append(
                        GaussianBasisFunction(
                            center=position,
                            exponent=exp,
                            angular_momentum=(0, 0, 0),
                            normalization=coef
                        )
                    )
                elif shell_type == 'p':
                    # Add p-type functions (px, py, pz)
                    for am in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        new_functions.append(
                            GaussianBasisFunction(
                                center=position,
                                exponent=exp,
                                angular_momentum=am,
                                normalization=coef
                            )
                        )
        
        self.add_functions(new_functions)
    
    def nuclear_attraction_integral(self, i: int, j: int, nuclear_pos: np.ndarray) -> float:
        """
        Calculate nuclear attraction integral for positrons (opposite sign from electrons).
        """
        # For positrons, the nuclear attraction is repulsive, so we change the sign
        # This will be implemented in the integral engine
        pass


class MixedMatterBasis:
    """Combined basis set for both electrons and positrons."""
    
    def __init__(self, 
                 electron_basis: BasisSet = None,
                 positron_basis: PositronBasis = None):
        """
        Initialize mixed matter basis set.
        
        Parameters:
        -----------
        electron_basis : BasisSet, optional
            Basis set for electrons
        positron_basis : PositronBasis, optional
            Basis set for positrons
        """
        self.electron_basis = electron_basis or BasisSet()
        self.positron_basis = positron_basis or PositronBasis()
        
        # Total number of basis functions
        self.n_electron_basis = self.electron_basis.n_basis
        self.n_positron_basis = self.positron_basis.n_basis
        self.n_total_basis = self.n_electron_basis + self.n_positron_basis
        
        # Integral cache
        self._integral_cache = {}
    
    def create_for_molecule(self, 
                          atoms: List[Tuple[str, np.ndarray]],
                          e_quality: str = 'standard',
                          p_quality: str = 'standard'):
        """
        Create basis sets for a complete molecule with both electrons and positrons.
        
        Parameters:
        -----------
        atoms : List[Tuple[str, np.ndarray]]
            List of (element, position) tuples
        e_quality : str
            Quality of electron basis set
        p_quality : str
            Quality of positron basis set
        """
        # Create electron basis
        for element, position in atoms:
            self.electron_basis.create_for_atom(element, position, e_quality)
        
        # Create positron basis
        for element, position in atoms:
            self.positron_basis.create_for_atom(element, position, p_quality)
        
        # Update counts
        self.n_electron_basis = self.electron_basis.n_basis
        self.n_positron_basis = self.positron_basis.n_basis
        self.n_total_basis = self.n_electron_basis + self.n_positron_basis
        
        # Clear integral cache
        self._integral_cache = {}
    
    def overlap_integral(self, i: int, j: int) -> float:
        """
        Calculate overlap integral between basis functions.
        
        Parameters:
        -----------
        i, j : int
            Indices of basis functions in the combined basis
            
        Returns:
        --------
        float
            Overlap integral <i|j>
        """
        # Check cache
        cache_key = ('overlap', i, j)
        if cache_key in self._integral_cache:
            return self._integral_cache[cache_key]
        
        # Simplified implementation - full version would calculate the overlap
        return 1.0 if i == j else 0.0
    
    def kinetic_integral(self, i: int, j: int) -> float:
        """
        Calculate kinetic energy integral.
        
        Parameters:
        -----------
        i, j : int
            Indices of basis functions in the combined basis
            
        Returns:
        --------
        float
            Kinetic energy integral
        """
        # Simplified implementation - full version would calculate kinetic integral
        return 0.5 if i == j else 0.0
    
    def nuclear_attraction_integral(self, i: int, j: int, nuclear_pos: np.ndarray) -> float:
        """
        Calculate nuclear attraction integral.
        
        Parameters:
        -----------
        i, j : int
            Indices of basis functions in the combined basis
        nuclear_pos : np.ndarray
            Position of the nucleus
            
        Returns:
        --------
        float
            Nuclear attraction integral
        """
        # Simplified implementation - full version would calculate nuclear attraction integral
        return -1.0 if i == j else 0.0