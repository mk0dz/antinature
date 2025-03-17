# antinature/core/basis.py

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
    
    def create_for_molecule(self, atoms, quality='standard'):
        """
        Create basis set for a complete molecule.
        
        Parameters:
        -----------
        atoms : List[Tuple[str, np.ndarray]]
            List of (element, position) tuples
        quality : str
            Quality of basis set ('minimal', 'standard', 'extended', 'large')
        """
        # Clear any existing basis functions
        self.basis_functions = []
        
        # Add basis functions for each atom
        for element, position in atoms:
            self.create_for_atom(element, position, quality)
        
        # Update count
        self.n_basis = len(self.basis_functions)
        return self
    
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
                'O': [
                    ('s', [7.6], [1.0]),  # Improved minimal O basis
                    ('p', [2.0], [1.0])   # Include p-orbitals for O
                ],
                'C': [
                    ('s', [0.9], [1.0]),  # Minimal C basis
                    ('p', [0.7], [1.0])   # Include p-orbitals for C 
                ],
                'N': [
                    ('s', [1.0], [1.0]),  # Minimal N basis
                    ('p', [0.7], [1.0])   # Include p-orbitals for N
                ],
                'Li': [
                    ('s', [0.6], [1.0])   # Minimal Li basis
                ],
                'Na': [
                    ('s', [0.4], [1.0])   # Minimal Na basis
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
                'O': [
                    ('s', [5909.0, 887.5, 204.7, 59.84, 19.14, 6.57], 
                        [0.0018, 0.0139, 0.0684, 0.2321, 0.4679, 0.3620]),
                    ('s', [2.93, 0.93, 0.29], [0.4888, 0.5818, 0.1446]),
                    ('p', [35.18, 7.904, 2.305, 0.717], [0.0597, 0.2392, 0.5082, 0.4754])
                ],
                'C': [
                    ('s', [3047.5, 457.4, 103.9, 29.21, 9.29, 3.16], 
                        [0.0018, 0.0138, 0.0680, 0.2306, 0.4670, 0.3623]),
                    ('s', [1.22, 0.37, 0.11], [0.5566, 0.5328, 0.0988]),
                    ('p', [13.50, 3.067, 0.905, 0.276], [0.0733, 0.2964, 0.5057, 0.3993])
                ],
                # Add more elements with standard basis
            },
            'extended': {
                'H': [
                    ('s', [33.8650, 5.0947, 1.1587, 0.3258], [0.0254, 0.1907, 0.5523, 0.5672]),
                    ('p', [1.0], [1.0])  # Add p functions for polarization
                ],
                'O': [
                    ('s', [9532.0, 1426.0, 326.0, 93.4, 30.4, 10.5, 3.72, 1.31], 
                        [0.0012, 0.0094, 0.0480, 0.1651, 0.3657, 0.4031, 0.1954, 0.0169]),
                    ('s', [0.54, 0.20], [0.8071, 0.3184]),
                    ('p', [49.98, 11.42, 3.35, 1.03, 0.31], [0.0339, 0.1868, 0.4640, 0.4112, 0.0621]),
                    ('d', [1.43, 0.36], [0.6667, 0.3333])
                ],
                # Add more elements with extended basis
            }
        }
        
        return basis_params.get(quality, {}).get(element, [])


class PositronBasis(BasisSet):
    """
    Specialized basis set for positrons.
    """
    
    def __init__(self):
        """Initialize an empty positron basis set."""
        super().__init__()
    
    def create_for_atom(self, element, position, quality='standard'):
        """
        Add positron-specific basis functions for an atom.
        
        Parameters:
        -----------
        element : str
            Element symbol
        position : array_like
            Position of the atom
        quality : str
            Quality of basis ('minimal', 'standard', 'extended', 'large')
        """
        # Get appropriate basis parameters for positrons
        # Positrons generally need more diffuse functions
        params = self._get_positron_basis_params(element, quality)
        
        # Create basis functions
        new_functions = []
        for exponent, angular_momentum in params:
            new_function = GaussianBasisFunction(
                center=position,
                exponent=exponent,
                angular_momentum=angular_momentum
            )
            new_functions.append(new_function)
        
        # Add functions to the basis set
        self.add_functions(new_functions)
    
    def create_for_molecule(self, atoms, quality='standard'):
        """
        Create positron basis set for a complete molecule.
        
        Parameters:
        -----------
        atoms : List[Tuple[str, np.ndarray]]
            List of (element, position) tuples
        quality : str
            Quality of basis set ('minimal', 'standard', 'extended', 'large')
        """
        # Clear any existing basis functions
        self.basis_functions = []
        
        # Add basis functions for each atom
        for element, position in atoms:
            self.create_for_atom(element, position, quality)
        
        # Update count
        self.n_basis = len(self.basis_functions)
        return self
    
    def _get_positron_basis_params(self, element, quality='standard'):
        """
        Get basis parameters for positrons for a specific element.
        
        Parameters:
        -----------
        element : str
            Element symbol
        quality : str
            Quality of basis ('minimal', 'standard', 'extended', 'large')
            
        Returns:
        --------
        List[Tuple[float, Tuple[int, int, int]]]
            List of (exponent, angular_momentum) tuples
        """
        # For positrons, we need more diffuse functions than for electrons
        params = []
        
        # Basic exponents for s-type functions
        if quality == 'minimal':
            exponents = [0.4]
        elif quality == 'standard':
            exponents = [0.2, 0.4, 0.8]
        elif quality == 'extended':
            exponents = [0.1, 0.2, 0.4, 0.8, 1.6]
        else:  # 'large'
            exponents = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        
        # Add s-type functions
        for alpha in exponents:
            params.append((alpha, (0, 0, 0)))
        
        # Add p-type functions for better description of positron distribution
        if quality != 'minimal':
            p_exponents = exponents[:-1] if len(exponents) > 1 else exponents
            for alpha in p_exponents:
                params.append((alpha, (1, 0, 0)))
                params.append((alpha, (0, 1, 0)))
                params.append((alpha, (0, 0, 1)))
        
        return params


class MixedMatterBasis:
    """
    Combined basis set for mixed matter/antinature calculations.
    """
    
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
    
    def create_for_molecule(self, atoms, e_quality='standard', p_quality='standard'):
        """
        Create basis sets for a molecular system.
        
        Parameters:
        -----------
        atoms : List[Tuple[str, np.ndarray]]
            List of (element, position) tuples
        e_quality : str
            Quality of electron basis ('standard', 'extended', or 'large')
        p_quality : str
            Quality of positron basis ('standard', 'extended', or 'large')
        """
        # Create electron basis
        self.electron_basis = BasisSet()
        self.electron_basis.create_for_molecule(atoms, quality=e_quality)
        
        # Create positron basis
        self.positron_basis = PositronBasis()
        self.positron_basis.create_for_molecule(atoms, quality=p_quality)
        
        # Combine the basis sets
        self.n_electron_basis = len(self.electron_basis.basis_functions)
        self.n_positron_basis = len(self.positron_basis.basis_functions)
        self.n_total_basis = self.n_electron_basis + self.n_positron_basis
    
    def create_positronium_basis(self, quality='extended'):
        """
        Create specialized basis sets optimized for positronium calculations.
        
        Parameters:
        -----------
        quality : str
            Quality of basis set: 'standard', 'extended', 'large', or 'positronium'
            
        Notes:
        ------
        'positronium' quality is specifically tuned for positronium systems
        and includes basis functions optimized for electron-positron correlation.
        """
        # Simple hydrogen atom at origin as a placeholder
        atoms = [('H', np.array([0.0, 0.0, 0.0]))]
        
        # Start with standard basis sets
        self.electron_basis = BasisSet()
        self.positron_basis = PositronBasis()
        
        if quality == 'positronium':
            # Specialized positronium basis
            # Add electron basis functions with exponents tuned for positronium
            exponents = [0.25, 0.5, 1.0, 2.0, 4.0]  # Optimized for positronium
            for alpha in exponents:
                # s-type functions
                center = np.array([0.0, 0.0, 0.0])
                angular_momentum = (0, 0, 0)
                self.electron_basis.add_function(
                    GaussianBasisFunction(center, alpha, angular_momentum)
                )
                
                # p-type functions for correlation
                if alpha <= 2.0:  # Only add p-functions for smaller exponents
                    self.electron_basis.add_function(
                        GaussianBasisFunction(center, alpha, (1, 0, 0))
                    )
                    self.electron_basis.add_function(
                        GaussianBasisFunction(center, alpha, (0, 1, 0))
                    )
                    self.electron_basis.add_function(
                        GaussianBasisFunction(center, alpha, (0, 0, 1))
                    )
            
            # Add positron basis functions with smaller exponents
            # Positrons tend to have more diffuse wavefunctions
            pos_exponents = [0.2, 0.4, 0.8, 1.6, 3.2]
            for alpha in pos_exponents:
                # s-type functions
                center = np.array([0.0, 0.0, 0.0])
                angular_momentum = (0, 0, 0)
                self.positron_basis.add_function(
                    GaussianBasisFunction(center, alpha, angular_momentum)
                )
                
                # p-type functions
                if alpha <= 1.6:  # Only add p-functions for smaller exponents
                    self.positron_basis.add_function(
                        GaussianBasisFunction(center, alpha, (1, 0, 0))
                    )
                    self.positron_basis.add_function(
                        GaussianBasisFunction(center, alpha, (0, 1, 0))
                    )
                    self.positron_basis.add_function(
                        GaussianBasisFunction(center, alpha, (0, 0, 1))
                    )
        else:
            # For other qualities, fall back to the standard method
            self.electron_basis.create_for_molecule(atoms, quality=quality)
            self.positron_basis.create_for_molecule(atoms, quality=quality)
            
            # If extended or large, add extra diffuse functions for positronium
            if quality in ['extended', 'large']:
                center = np.array([0.0, 0.0, 0.0])
                self.electron_basis.add_function(
                    GaussianBasisFunction(center, 0.25, (0, 0, 0))
                )
                self.positron_basis.add_function(
                    GaussianBasisFunction(center, 0.2, (0, 0, 0))
                )
        
        # Update counts
        self.n_electron_basis = len(self.electron_basis.basis_functions)
        self.n_positron_basis = len(self.positron_basis.basis_functions)
        self.n_total_basis = self.n_electron_basis + self.n_positron_basis
        
        print(f"Created positronium-optimized basis: {self.n_electron_basis} electron functions, {self.n_positron_basis} positron functions")
        return self
    
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