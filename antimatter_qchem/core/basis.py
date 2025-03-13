import numpy as np
from typing import List, Tuple, Optional, Dict

class GaussianBasisFunction:
    """
    Representation of a Gaussian basis function for quantum chemistry.
    """
    def __init__(self, 
                 center: np.ndarray, 
                 exponent: float, 
                 angular_momentum: Tuple[int, int, int] = (0, 0, 0),
                 normalization: Optional[float] = None):
        """
        Initialize a Gaussian basis function.
        
        Parameters:
        -----------
        center : np.ndarray
            Center of the Gaussian (3D coordinates)
        exponent : float
            Exponent of the Gaussian
        angular_momentum : Tuple[int, int, int]
            Angular momentum components (nx, ny, nz)
        normalization : float, optional
            Normalization constant. If None, will be calculated.
        """
        self.center = center
        self.exponent = exponent
        self.angular_momentum = angular_momentum
        
        # Calculate normalization if not provided
        if normalization is None:
            self.normalization = self._calculate_normalization()
        else:
            self.normalization = normalization
    
    def _calculate_normalization(self) -> float:
        """Calculate normalization constant for the Gaussian."""
        alpha = self.exponent
        nx, ny, nz = self.angular_momentum
        
        # Double factorial function
        def double_factorial(n):
            if n <= 0:
                return 1
            return n * double_factorial(n - 2)
        
        # Normalization formula for Cartesian Gaussian
        prefactor = (2 * alpha / np.pi) ** 0.75
        
        # Normalization for angular momentum components
        x_norm = (4 * alpha) ** (nx / 2) / np.sqrt(double_factorial(2 * nx - 1))
        y_norm = (4 * alpha) ** (ny / 2) / np.sqrt(double_factorial(2 * ny - 1))
        z_norm = (4 * alpha) ** (nz / 2) / np.sqrt(double_factorial(2 * nz - 1))
        
        return prefactor * x_norm * y_norm * z_norm
    
    def evaluate(self, r: np.ndarray) -> float:
        """
        Evaluate the basis function at position r.
        
        Parameters:
        -----------
        r : np.ndarray
            Position vector
            
        Returns:
        --------
        float
            Value of the basis function at r
        """
        # Calculate displacements
        dx = r[0] - self.center[0]
        dy = r[1] - self.center[1]
        dz = r[2] - self.center[2]
        
        # Distance squared
        r_squared = dx*dx + dy*dy + dz*dz
        
        # Angular momentum components
        nx, ny, nz = self.angular_momentum
        
        # Polynomial part
        polynomial = dx**nx * dy**ny * dz**nz
        
        # Exponential part
        exponential = np.exp(-self.exponent * r_squared)
        
        return self.normalization * polynomial * exponential
    
    def evaluate_gradient(self, r: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of the basis function at position r.
        
        Parameters:
        -----------
        r : np.ndarray
            Position vector
            
        Returns:
        --------
        np.ndarray
            Gradient of the basis function at r
        """
        # Calculate displacements
        dx = r[0] - self.center[0]
        dy = r[1] - self.center[1]
        dz = r[2] - self.center[2]
        
        # Distance squared
        r_squared = dx*dx + dy*dy + dz*dz
        
        # Angular momentum components
        nx, ny, nz = self.angular_momentum
        
        # Value of the basis function
        value = self.evaluate(r)
        
        # Gradient components
        gradient = np.zeros(3)
        
        # X component
        if nx > 0:
            # Term from derivative of polynomial part
            gradient[0] = nx * dx**(nx-1) * dy**ny * dz**nz
        else:
            gradient[0] = 0
        # Add term from derivative of exponential
        gradient[0] -= 2 * self.exponent * dx * dx**nx * dy**ny * dz**nz
        # Multiply by exponential and normalization
        gradient[0] *= self.normalization * np.exp(-self.exponent * r_squared)
        
        # Y component (similar pattern)
        if ny > 0:
            gradient[1] = ny * dx**nx * dy**(ny-1) * dz**nz
        else:
            gradient[1] = 0
        gradient[1] -= 2 * self.exponent * dy * dx**nx * dy**ny * dz**nz
        gradient[1] *= self.normalization * np.exp(-self.exponent * r_squared)
        
        # Z component
        if nz > 0:
            gradient[2] = nz * dx**nx * dy**ny * dz**(nz-1)
        else:
            gradient[2] = 0
        gradient[2] -= 2 * self.exponent * dz * dx**nx * dy**ny * dz**nz
        gradient[2] *= self.normalization * np.exp(-self.exponent * r_squared)
        
        return gradient
    
    def evaluate_laplacian(self, r: np.ndarray) -> float:
        """
        Evaluate the Laplacian of the basis function at position r.
        
        Parameters:
        -----------
        r : np.ndarray
            Position vector
            
        Returns:
        --------
        float
            Laplacian of the basis function at r
        """
        # Calculate displacements
        dx = r[0] - self.center[0]
        dy = r[1] - self.center[1]
        dz = r[2] - self.center[2]
        
        # Distance squared
        r_squared = dx*dx + dy*dy + dz*dz
        
        # Angular momentum components
        nx, ny, nz = self.angular_momentum
        
        # Base value
        value = self.evaluate(r)
        
        # Term 1: -2α(3 + 2(nx+ny+nz) - 4α*r²)
        term1 = -2 * self.exponent * (3 + 2*(nx+ny+nz) - 4*self.exponent*r_squared)
        
        # Term 2: Contribution from second derivatives of polynomial part
        term2 = 0
        if nx > 1:
            term2 += nx * (nx-1) * dx**(nx-2) / dx**nx
        if ny > 1:
            term2 += ny * (ny-1) * dy**(ny-2) / dy**ny
        if nz > 1:
            term2 += nz * (nz-1) * dz**(nz-2) / dz**nz
        
        return value * (term1 + term2)

class BasisSet:
    """
    Collection of basis functions for a molecular system.
    """
    def __init__(self, basis_functions: List[GaussianBasisFunction] = None):
        """
        Initialize a basis set.
        
        Parameters:
        -----------
        basis_functions : List[GaussianBasisFunction], optional
            Initial set of basis functions
        """
        self.basis_functions = basis_functions or []
        self.n_basis = len(self.basis_functions)
    
    def add_function(self, basis_function: GaussianBasisFunction):
        """Add a basis function to the set."""
        self.basis_functions.append(basis_function)
        self.n_basis = len(self.basis_functions)
    
    def extend(self, other_basis: 'BasisSet'):
        """Extend this basis set with another one."""
        self.basis_functions.extend(other_basis.basis_functions)
        self.n_basis = len(self.basis_functions)
    
    def get_function(self, index: int) -> GaussianBasisFunction:
        """Get a specific basis function by index."""
        return self.basis_functions[index]
    
    def evaluate_all_at(self, r: np.ndarray) -> np.ndarray:
        """
        Evaluate all basis functions at position r.
        
        Parameters:
        -----------
        r : np.ndarray
            Position vector
            
        Returns:
        --------
        np.ndarray
            Array of values of all basis functions at r
        """
        values = np.zeros(self.n_basis)
        for i, func in enumerate(self.basis_functions):
            values[i] = func.evaluate(r)
        return values

class PositronBasis(BasisSet):
    """
    Specialized basis functions optimized for positron systems.
    """
    def __init__(self):
        """Initialize an empty positron basis set."""
        super().__init__()
        self.annihilation_functions = []  # Functions optimized for annihilation
    
    def generate_positron_basis(self, atom_type: str, position: np.ndarray, 
                                quality: str = 'standard'):
        """
        Generate a positron-optimized basis set for an atom.
        
        Parameters:
        -----------
        atom_type : str
            Element symbol
        position : np.ndarray
            Position of the atom
        quality : str
            Basis set quality ('minimal', 'standard', 'extended')
        """
        # Define positron-optimized exponents for different elements
        # These would ideally be derived from positronic calculations
        # Here we use a simple scaling of standard electronic values
        exponent_scaling = 0.5  # Positron functions are more diffuse
        
        # Base exponents for common elements (H, C, N, O)
        base_exponents = {
            'H': [0.5, 0.1],
            'C': [0.8, 0.25, 0.08],
            'N': [0.9, 0.3, 0.09],
            'O': [1.0, 0.35, 0.1]
        }
        
        # Add more diffuse functions for positrons
        if atom_type in base_exponents:
            exponents = [exp * exponent_scaling for exp in base_exponents[atom_type]]
            
            # Add extra diffuse functions for extended basis
            if quality == 'extended':
                exponents.append(0.03 * exponent_scaling)  # Very diffuse
                exponents.append(0.01 * exponent_scaling)  # Extremely diffuse
            
            # Create basis functions
            for exponent in exponents:
                # s-type function (spherical)
                self.add_function(GaussianBasisFunction(
                    center=position,
                    exponent=exponent,
                    angular_momentum=(0, 0, 0)
                ))
                
                # For non-hydrogen atoms, add p-type functions
                if atom_type != 'H' and (quality != 'minimal'):
                    self.add_function(GaussianBasisFunction(
                        center=position,
                        exponent=exponent,
                        angular_momentum=(1, 0, 0)  # px
                    ))
                    self.add_function(GaussianBasisFunction(
                        center=position,
                        exponent=exponent,
                        angular_momentum=(0, 1, 0)  # py
                    ))
                    self.add_function(GaussianBasisFunction(
                        center=position,
                        exponent=exponent,
                        angular_momentum=(0, 0, 1)  # pz
                    ))
    
    def add_annihilation_functions(self, region_center: np.ndarray, 
                                  size: float = 1.0, 
                                  n_functions: int = 3):
        """
        Add specialized basis functions optimized for electron-positron annihilation.
        
        Parameters:
        -----------
        region_center : np.ndarray
            Center of the annihilation region
        size : float
            Approximate size of the annihilation region
        n_functions : int
            Number of functions to add
        """
        if size <= 0:
           size = 1.0 
        # Exponents for annihilation functions - intermediate values
        # Not too diffuse (needs good overlap) and not too tight
        base_exponent = 1.0 / (size * size)
        
        for i in range(n_functions):
            # Scale exponent for multiple functions
            exponent = base_exponent * (1.5 ** (i - n_functions//2))
            
            # Create s-type function centered in annihilation region
            function = GaussianBasisFunction(
                center=region_center,
                exponent=exponent,
                angular_momentum=(0, 0, 0)
            )
            
            # Add to basis and track in annihilation functions list
            self.add_function(function)
            self.annihilation_functions.append(function)
    
    def add_bond_functions(self, bond_centers: List[np.ndarray], 
                          bond_lengths: List[float]):
        """
        Add bond-centered functions for better positron representation.
        
        Parameters:
        -----------
        bond_centers : List[np.ndarray]
            Centers of chemical bonds
        bond_lengths : List[float]
            Lengths of the bonds
        """
        for center, length in zip(bond_centers, bond_lengths):
            # Exponent based on bond length
            exponent = 0.2 / (length * length)
            
            # Add s-type function at bond center
            self.add_function(GaussianBasisFunction(
                center=center,
                exponent=exponent,
                angular_momentum=(0, 0, 0)
            ))

class MixedMatterBasis:
    """
    Combined basis set for mixed electron-positron systems.
    """
    def __init__(self, 
                 electron_basis: BasisSet = None, 
                 positron_basis: PositronBasis = None):
        """
        Initialize a mixed matter basis set.
        
        Parameters:
        -----------
        electron_basis : BasisSet, optional
            Basis for electrons
        positron_basis : PositronBasis, optional
            Basis for positrons
        """
        self.electron_basis = electron_basis or BasisSet()
        self.positron_basis = positron_basis or PositronBasis()
        
        # For convenience, track the number of functions
        self.n_electron_basis = self.electron_basis.n_basis
        self.n_positron_basis = self.positron_basis.n_basis
        self.n_total_basis = self.n_electron_basis + self.n_positron_basis
        
        # Create integral engine for calculations
        from .integral_engine import AntimatterIntegralEngine
        self.integral_engine = AntimatterIntegralEngine()
    
    def overlap_integral(self, i, j):
        """
        Calculate overlap integral between two basis functions.
        
        Parameters:
        -----------
        i : int
            Index of first basis function
        j : int
            Index of second basis function
            
        Returns:
        --------
        float
            Overlap integral <i|j>
        """
        basis_i = self.get_basis_function(i)
        basis_j = self.get_basis_function(j)
        return self.integral_engine.overlap_integral(basis_i, basis_j)
    
    def kinetic_integral(self, i, j):
        """
        Calculate kinetic integral between two basis functions.
        
        Parameters:
        -----------
        i : int
            Index of first basis function
        j : int
            Index of second basis function
            
        Returns:
        --------
        float
            Kinetic integral <i|-∇²/2|j>
        """
        basis_i = self.get_basis_function(i)
        basis_j = self.get_basis_function(j)
        return self.integral_engine.kinetic_integral(basis_i, basis_j)
    
    def nuclear_attraction_integral(self, i, j, nuclear_pos):
        """
        Calculate nuclear attraction integral between two basis functions.
        
        Parameters:
        -----------
        i : int
            Index of first basis function
        j : int
            Index of second basis function
        nuclear_pos : np.ndarray
            Position of nucleus
            
        Returns:
        --------
        float
            Nuclear attraction integral <i|-Z/r|j>
        """
        basis_i = self.get_basis_function(i)
        basis_j = self.get_basis_function(j)
        return self.integral_engine.nuclear_attraction_integral(basis_i, basis_j, nuclear_pos)
    
    def electron_repulsion_integral(self, i, j, k, l):
        """
        Calculate electron repulsion integral between four basis functions.
        
        Parameters:
        -----------
        i, j, k, l : int
            Indices of the four basis functions
            
        Returns:
        --------
        float
            Electron repulsion integral <ij|1/r12|kl>
        """
        basis_i = self.get_basis_function(i)
        basis_j = self.get_basis_function(j)
        basis_k = self.get_basis_function(k)
        basis_l = self.get_basis_function(l)
        return self.integral_engine.electron_repulsion_integral(basis_i, basis_j, basis_k, basis_l)
    
    def annihilation_integral(self, i, j):
        """
        Calculate annihilation integral between electron and positron basis functions.
        
        Parameters:
        -----------
        i : int
            Index of electron basis function
        j : int
            Index of positron basis function
            
        Returns:
        --------
        float
            Annihilation integral
        """
        # Check if the indices correspond to electron and positron functions
        if i < self.n_electron_basis and j >= self.n_electron_basis:
            electron_basis = self.get_basis_function(i)
            positron_basis = self.get_basis_function(j)
            return self.integral_engine.annihilation_integral(electron_basis, positron_basis)
        elif j < self.n_electron_basis and i >= self.n_electron_basis:
            electron_basis = self.get_basis_function(j)
            positron_basis = self.get_basis_function(i)
            return self.integral_engine.annihilation_integral(electron_basis, positron_basis)
        else:
            # Annihilation only happens between electrons and positrons
            return 0.0
    
    def mass_velocity_integral(self, i, j):
        """
        Calculate mass-velocity relativistic correction integral.
        
        Parameters:
        -----------
        i : int
            Index of first basis function
        j : int
            Index of second basis function
            
        Returns:
        --------
        float
            Mass-velocity integral <i|-∇⁴/8|j>
        """
        basis_i = self.get_basis_function(i)
        basis_j = self.get_basis_function(j)
        return self.integral_engine.mass_velocity_integral(basis_i, basis_j)
    
    def darwin_integral(self, i, j, nuclear_pos):
        """
        Calculate Darwin term relativistic correction integral.
        
        Parameters:
        -----------
        i : int
            Index of first basis function
        j : int
            Index of second basis function
        nuclear_pos : np.ndarray
            Position of nucleus
            
        Returns:
        --------
        float
            Darwin integral <i|πδ(r)/2|j>
        """
        basis_i = self.get_basis_function(i)
        basis_j = self.get_basis_function(j)
        return self.integral_engine.darwin_integral(basis_i, basis_j, nuclear_pos)
    
    def get_basis_function(self, index):
        """
        Get basis function at given index, accounting for mixed basis.
        
        Parameters:
        -----------
        index : int
            Index of basis function
            
        Returns:
        --------
        GaussianBasisFunction
            The basis function
        """
        if index < self.n_electron_basis:
            return self.electron_basis.get_function(index)
        else:
            return self.positron_basis.get_function(index - self.n_electron_basis)
    
    def create_standard_electronic_basis(self, molecule: List[Tuple[str, np.ndarray]], 
                                        quality: str = 'standard'):
        """
        Create a standard electronic basis set for a molecule.
        
        Parameters:
        -----------
        molecule : List[Tuple[str, np.ndarray]]
            List of (element, position) tuples
        quality : str
            Basis set quality ('minimal', 'standard', 'extended')
        """
        # Define standard exponents for different elements
        # This is a simplified version of standard basis sets like STO-3G
        standard_exponents = {
            'H': {'minimal': [0.5], 'standard': [1.0, 0.3], 'extended': [1.0, 0.3, 0.1]},
            'C': {'minimal': [1.5], 'standard': [3.0, 0.8, 0.2], 'extended': [3.0, 0.8, 0.2, 0.07]},
            'N': {'minimal': [1.8], 'standard': [3.5, 0.9, 0.25], 'extended': [3.5, 0.9, 0.25, 0.08]},
            'O': {'minimal': [2.0], 'standard': [4.0, 1.0, 0.3], 'extended': [4.0, 1.0, 0.3, 0.09]},
            'Si': {'minimal': [2.2], 'standard': [4.5, 1.2, 0.35], 'extended': [4.5, 1.2, 0.35, 0.1]}
        }
        
        # Create basis functions for each atom
        for element, position in molecule:
            if element in standard_exponents:
                exponents = standard_exponents[element][quality]
                
                # Add basis functions for this atom
                for exponent in exponents:
                    # s-type function
                    self.electron_basis.add_function(GaussianBasisFunction(
                        center=position,
                        exponent=exponent,
                        angular_momentum=(0, 0, 0)
                    ))
                    
                    # For non-hydrogen atoms, add p-type functions
                    if element != 'H' and quality != 'minimal':
                        self.electron_basis.add_function(GaussianBasisFunction(
                            center=position,
                            exponent=exponent,
                            angular_momentum=(1, 0, 0)  # px
                        ))
                        self.electron_basis.add_function(GaussianBasisFunction(
                            center=position,
                            exponent=exponent,
                            angular_momentum=(0, 1, 0)  # py
                        ))
                        self.electron_basis.add_function(GaussianBasisFunction(
                            center=position,
                            exponent=exponent,
                            angular_momentum=(0, 0, 1)  # pz
                        ))
    
    def create_positronic_basis(self, molecule: List[Tuple[str, np.ndarray]], 
                               quality: str = 'standard',
                               add_bond_functions: bool = True,
                               add_annihilation_functions: bool = True):
        """
        Create a positron-optimized basis set for a molecule.
        
        Parameters:
        -----------
        molecule : List[Tuple[str, np.ndarray]]
            List of (element, position) tuples
        quality : str
            Basis set quality ('minimal', 'standard', 'extended')
        add_bond_functions : bool
            Whether to add bond-centered functions
        add_annihilation_functions : bool
            Whether to add specialized annihilation functions
        """
        # Generate atomic basis functions
        for element, position in molecule:
            self.positron_basis.generate_positron_basis(element, position, quality)
        
        # Add bond-centered functions if requested
        if add_bond_functions and len(molecule) > 1:
            # Find bonds (simplified by assuming bonds between all atoms within a cutoff)
            bond_centers = []
            bond_lengths = []
            
            # Reasonable cutoff for bonds
            cutoff = 3.0  # angstroms
            
            for i in range(len(molecule)):
                for j in range(i+1, len(molecule)):
                    pos_i = molecule[i][1]
                    pos_j = molecule[j][1]
                    
                    # Calculate bond length
                    bond_length = np.linalg.norm(pos_i - pos_j)
                    
                    if bond_length < cutoff:
                        # Calculate bond center
                        bond_center = 0.5 * (pos_i + pos_j)
                        
                        bond_centers.append(bond_center)
                        bond_lengths.append(bond_length)
            
            # Add bond functions
            self.positron_basis.add_bond_functions(bond_centers, bond_lengths)
        
        # Add annihilation functions if requested
        if add_annihilation_functions and len(molecule) > 0:
            # Simplified approach: place annihilation functions at molecule center
            center = np.zeros(3)
            for _, position in molecule:
                center += position
            center /= len(molecule)
            
            # Estimate molecular size
            size = 0.0
            for _, position in molecule:
                dist = np.linalg.norm(position - center)
                if dist > size:
                    size = dist
            
            # Add annihilation functions
            self.positron_basis.add_annihilation_functions(center, size)
    
    def create_for_molecule(self, molecule: List[Tuple[str, np.ndarray]], 
                           e_quality: str = 'standard',
                           p_quality: str = 'standard'):
        """
        Create a complete mixed basis set for a molecule.
        
        Parameters:
        -----------
        molecule : List[Tuple[str, np.ndarray]]
            List of (element, position) tuples
        e_quality : str
            Electronic basis quality
        p_quality : str
            Positronic basis quality
        """
        # Create electronic basis
        self.create_standard_electronic_basis(molecule, e_quality)
        
        # Create positronic basis
        self.create_positronic_basis(molecule, p_quality)
        
        # Update counts
        self.n_electron_basis = self.electron_basis.n_basis
        self.n_positron_basis = self.positron_basis.n_basis
        self.n_total_basis = self.n_electron_basis + self.n_positron_basis
    
    def transform_to_orthogonal(self):
        """
        Transform the combined basis set to an orthogonal one.
        
        This is a simplified implementation. A full version would:
        1. Calculate the overlap matrix S
        2. Find S^(-1/2) via eigendecomposition
        3. Apply S^(-1/2) to the basis functions
        """
        # Not implemented in this demonstration
        # Would require calculating overlap matrix and transformation
        pass

class BasisTransformation:
    """
    Utilities for basis set transformations and operations.
    """
    @staticmethod
    def ao_to_mo_transformation(coefficients: np.ndarray, ao_values: np.ndarray) -> np.ndarray:
        """
        Transform from atomic orbital to molecular orbital basis.
        
        Parameters:
        -----------
        coefficients : np.ndarray
            MO coefficients matrix (n_mo × n_ao)
        ao_values : np.ndarray
            Values in the AO basis
            
        Returns:
        --------
        np.ndarray
            Values in the MO basis
        """
        return np.dot(coefficients, ao_values)
    
    @staticmethod
    def localize_orbitals(mo_coefficients: np.ndarray, overlap_matrix: np.ndarray) -> np.ndarray:
        """
        Create localized orbitals using Boys localization.
        
        Parameters:
        -----------
        mo_coefficients : np.ndarray
            MO coefficients matrix (n_mo × n_ao)
        overlap_matrix : np.ndarray
            AO overlap matrix
            
        Returns:
        --------
        np.ndarray
            Coefficients of localized orbitals
        """
        # This is a simplified skeleton - full implementation would require:
        # 1. Computing orbital centers (dipole integrals)
        # 2. Maximizing sum of distances between orbital centers
        # 3. Iterative optimization
        
        # Placeholder for demonstration
        return mo_coefficients