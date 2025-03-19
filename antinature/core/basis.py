import math
import warnings
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np


class GaussianBasisFunction:
    """
    Gaussian basis function with comprehensive support for arbitrary angular momentum.

    Represents a primitive Gaussian function of the form:
    g(r) = N * (x-Rx)^a * (y-Ry)^b * (z-Rz)^c * exp(-alpha * |r-R|^2)

    Where N is the normalization constant, (a,b,c) are the angular momentum components,
    alpha is the exponent, and R is the center coordinates.
    """

    def __init__(
        self,
        center: np.ndarray,
        exponent: float,
        angular_momentum: Tuple[int, int, int] = (0, 0, 0),
        normalization: Optional[float] = None,
    ):
        """
        Initialize a Gaussian basis function.

        Parameters:
        -----------
        center : np.ndarray
            Coordinates of the center (3D vector)
        exponent : float
            Gaussian exponent (alpha)
        angular_momentum : Tuple[int, int, int]
            Angular momentum components (a,b,c) for x,y,z
        normalization : float, optional
            Normalization constant (computed if not provided)
        """
        self.center = np.asarray(center)
        self.exponent = float(exponent)
        self.angular_momentum = tuple(angular_momentum)

        # Calculate normalization if not provided
        self.normalization = (
            normalization
            if normalization is not None
            else self.calculate_normalization()
        )

        # Cache common calculations
        self.alpha = self.exponent
        self.nx, self.ny, self.nz = self.angular_momentum

    def calculate_normalization(self) -> float:
        """
        Calculate the normalization constant for the Gaussian basis function.

        Returns:
        --------
        float
            Normalization constant
        """
        alpha = self.exponent
        nx, ny, nz = self.angular_momentum

        # Prefactor from the Gaussian part
        prefactor = (2 * alpha / np.pi) ** 0.75

        # Normalization for each angular momentum component
        def norm_component(n):
            if n == 0:
                return 1.0
            else:
                # Use double factorial (1*3*5*...) for normalization
                double_fact = 1.0
                for i in range(1, 2 * n, 2):
                    double_fact *= i
                return math.sqrt((4 * alpha) ** n / double_fact)

        nx_norm = norm_component(nx)
        ny_norm = norm_component(ny)
        nz_norm = norm_component(nz)

        return prefactor * nx_norm * ny_norm * nz_norm

    def evaluate(self, r: np.ndarray) -> float:
        """
        Evaluate the basis function at position r.

        Parameters:
        -----------
        r : np.ndarray
            Position vector (3D)

        Returns:
        --------
        float
            Value of the basis function at position r
        """
        # Ensure r is a numpy array
        r = np.asarray(r)

        # Vector from center to position r
        dr = r - self.center

        # Compute polynomial part (x-Rx)^a * (y-Ry)^b * (z-Rz)^c
        polynomial = 1.0
        if self.nx > 0:
            polynomial *= dr[0] ** self.nx
        if self.ny > 0:
            polynomial *= dr[1] ** self.ny
        if self.nz > 0:
            polynomial *= dr[2] ** self.nz

        # Compute exponential part exp(-alpha * |r-R|^2)
        r_squared = np.sum(dr**2)
        exponential = np.exp(-self.alpha * r_squared)

        # Combine all parts
        return self.normalization * polynomial * exponential

    def get_angular_momentum_sum(self) -> int:
        """
        Get the total angular momentum (L = a + b + c).

        Returns:
        --------
        int
            Sum of angular momentum components
        """
        return self.nx + self.ny + self.nz

    def get_type(self) -> str:
        """
        Get the orbital type (s, p, d, etc.) based on angular momentum.

        Returns:
        --------
        str
            Orbital type label
        """
        L = self.get_angular_momentum_sum()
        types = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h'}
        return types.get(L, f"L={L}")

    def __str__(self) -> str:
        """String representation of the basis function."""
        return (
            f"{self.get_type()}-type Gaussian at {self.center}, "
            f"α={self.exponent:.4f}, L={self.angular_momentum}"
        )

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"GaussianBasisFunction(center={self.center}, "
            f"exponent={self.exponent}, "
            f"angular_momentum={self.angular_momentum}, "
            f"normalization={self.normalization:.8e})"
        )


class BasisSet:
    """
    Comprehensive basis set implementation for quantum chemistry calculations.

    This class represents a collection of Gaussian basis functions
    and provides methods for basis set creation, manipulation, and information access.
    """

    def __init__(self, basis_functions=None, name=""):
        """
        Initialize basis set with given basis functions.

        Parameters:
        -----------
        basis_functions : List[GaussianBasisFunction], optional
            List of basis functions
        name : str
            Name or identifier for the basis set
        """
        # Handle different call patterns to make initialization more robust
        if basis_functions is None:
            self.basis_functions = []
        elif isinstance(basis_functions, str) and name == "":
            # If first argument is a string and second is empty, assume it's the name
            self.basis_functions = []
            self.name = basis_functions
        else:
            self.basis_functions = basis_functions

        self.name = name
        self.n_basis = len(self.basis_functions)

        # Cache for performance optimization
        self._cache = {}

    def add_function(self, basis_function: GaussianBasisFunction):
        """
        Add a basis function to the set.

        Parameters:
        -----------
        basis_function : GaussianBasisFunction
            Basis function to add
        """
        self.basis_functions.append(basis_function)
        self.n_basis = len(self.basis_functions)

        # Clear cache when basis set changes
        self._cache = {}

    def add_functions(self, basis_functions: List[GaussianBasisFunction]):
        """
        Add multiple basis functions to the set.

        Parameters:
        -----------
        basis_functions : List[GaussianBasisFunction]
            List of basis functions to add
        """
        self.basis_functions.extend(basis_functions)
        self.n_basis = len(self.basis_functions)

        # Clear cache when basis set changes
        self._cache = {}

    def create_for_atom(
        self, element: str, position: np.ndarray, quality: str = 'standard'
    ):
        """
        Create basis functions for a given atom at specified position.

        Parameters:
        -----------
        element : str
            Element symbol
        position : np.ndarray
            Atomic position
        quality : str
            Basis set quality ('minimal', 'standard', 'extended', 'large')

        Returns:
        --------
        BasisSet
            Self reference for method chaining
        """
        # Get basis parameters based on quality
        basis_params = self._get_basis_params(element, quality)

        if not basis_params:
            warnings.warn(
                f"No basis parameters available for {element} with quality {quality}"
            )
            return self

        # Create basis functions from parameters
        new_functions = []
        for shell_type, exponents, coefficients in basis_params:
            # Process by shell type
            if shell_type == 's':
                # s-type function (angular momentum = 0,0,0)
                for exp, coef in zip(exponents, coefficients):
                    new_functions.append(
                        GaussianBasisFunction(
                            center=position, exponent=exp, angular_momentum=(0, 0, 0)
                        )
                    )
            elif shell_type == 'p':
                # p-type functions (px, py, pz)
                for exp, coef in zip(exponents, coefficients):
                    for am in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        new_functions.append(
                            GaussianBasisFunction(
                                center=position, exponent=exp, angular_momentum=am
                            )
                        )
            elif shell_type == 'd':
                # d-type functions
                # Standard definition: d_xy, d_xz, d_yz, d_x²-y², d_z²
                for exp, coef in zip(exponents, coefficients):
                    for am in [
                        (2, 0, 0),
                        (0, 2, 0),
                        (0, 0, 2),
                        (1, 1, 0),
                        (1, 0, 1),
                        (0, 1, 1),
                    ]:
                        new_functions.append(
                            GaussianBasisFunction(
                                center=position, exponent=exp, angular_momentum=am
                            )
                        )

        self.add_functions(new_functions)
        return self

    def create_for_molecule(
        self, atoms: List[Tuple[str, np.ndarray]], quality: str = 'standard'
    ):
        """
        Create basis set for a complete molecule.

        Parameters:
        -----------
        atoms : List[Tuple[str, np.ndarray]]
            List of (element, position) tuples
        quality : str
            Quality of basis set ('minimal', 'standard', 'extended', 'large')

        Returns:
        --------
        BasisSet
            Self reference for method chaining
        """
        # Clear any existing basis functions
        self.basis_functions = []
        self._cache = {}

        # Add basis functions for each atom
        for element, position in atoms:
            self.create_for_atom(element, position, quality)

        # Update count
        self.n_basis = len(self.basis_functions)
        return self

    def _get_basis_params(self, element: str, quality: str) -> List[Tuple]:
        """
        Get basis set parameters for a given element and quality.

        Parameters:
        -----------
        element : str
            Element symbol
        quality : str
            Basis set quality

        Returns:
        --------
        List[Tuple]
            List of (shell_type, exponents, coefficients) tuples
        """
        # Define basis parameters for common elements
        basis_params = {
            'minimal': {
                'H': [('s', [0.5], [1.0])],  # Minimal H basis with single s function
                'He': [('s', [0.8], [1.0])],  # Minimal He basis
                'O': [
                    ('s', [7.6], [1.0]),  # Minimal O basis
                    ('p', [2.0], [1.0]),  # Include p-orbitals for O
                ],
                'C': [
                    ('s', [5.0], [1.0]),  # Minimal C basis
                    ('p', [1.5], [1.0]),  # Include p-orbitals for C
                ],
                'N': [
                    ('s', [6.5], [1.0]),  # Minimal N basis
                    ('p', [1.8], [1.0]),  # Include p-orbitals for N
                ],
                'Li': [('s', [0.6], [1.0])],  # Minimal Li basis
                'Na': [('s', [0.4], [1.0])],  # Minimal Na basis
                # Additional elements can be added here
            },
            'standard': {
                'H': [('s', [13.0, 1.96, 0.4446], [0.0334, 0.2347, 0.8137])],  # STO-3G
                'He': [('s', [38.4, 5.77, 1.24], [0.0236, 0.1557, 0.4685])],
                'O': [
                    (
                        's',
                        [5909.0, 887.5, 204.7, 59.84, 19.14, 6.57],
                        [0.0018, 0.0139, 0.0684, 0.2321, 0.4679, 0.3620],
                    ),
                    ('s', [2.93, 0.93, 0.29], [0.4888, 0.5818, 0.1446]),
                    (
                        'p',
                        [35.18, 7.904, 2.305, 0.717],
                        [0.0597, 0.2392, 0.5082, 0.4754],
                    ),
                ],
                'C': [
                    (
                        's',
                        [3047.5, 457.4, 103.9, 29.21, 9.29, 3.16],
                        [0.0018, 0.0138, 0.0680, 0.2306, 0.4670, 0.3623],
                    ),
                    ('s', [1.22, 0.37, 0.11], [0.5566, 0.5328, 0.0988]),
                    (
                        'p',
                        [13.50, 3.067, 0.905, 0.276],
                        [0.0733, 0.2964, 0.5057, 0.3993],
                    ),
                ],
                'N': [
                    (
                        's',
                        [4173.5, 627.5, 142.6, 40.23, 12.82, 4.39],
                        [0.0018, 0.0137, 0.0678, 0.2307, 0.4685, 0.3603],
                    ),
                    ('s', [1.65, 0.46, 0.13], [0.5445, 0.5342, 0.0994]),
                    ('p', [17.68, 3.97, 1.14, 0.32], [0.0725, 0.2916, 0.5010, 0.4038]),
                ],
                # Additional elements with standard basis parameters
            },
            'extended': {
                'H': [
                    (
                        's',
                        [33.8650, 5.0947, 1.1587, 0.3258],
                        [0.0254, 0.1907, 0.5523, 0.5672],
                    ),
                    ('p', [1.0], [1.0]),  # Add p functions for polarization
                ],
                'O': [
                    (
                        's',
                        [9532.0, 1426.0, 326.0, 93.4, 30.4, 10.5, 3.72, 1.31],
                        [
                            0.0012,
                            0.0094,
                            0.0480,
                            0.1651,
                            0.3657,
                            0.4031,
                            0.1954,
                            0.0169,
                        ],
                    ),
                    ('s', [0.54, 0.20], [0.8071, 0.3184]),
                    (
                        'p',
                        [49.98, 11.42, 3.35, 1.03, 0.31],
                        [0.0339, 0.1868, 0.4640, 0.4112, 0.0621],
                    ),
                    ('d', [1.43, 0.36], [0.6667, 0.3333]),
                ],
                'C': [
                    (
                        's',
                        [6665.0, 1000.0, 228.0, 64.71, 21.06, 7.50, 2.80, 0.52],
                        [
                            0.0010,
                            0.0077,
                            0.0400,
                            0.1375,
                            0.3212,
                            0.4358,
                            0.1671,
                            -0.0059,
                        ],
                    ),
                    ('s', [0.52, 0.16], [0.5869, 0.4754]),
                    (
                        'p',
                        [30.63, 7.03, 2.11, 0.68, 0.21],
                        [0.0291, 0.1702, 0.4515, 0.4585, 0.1301],
                    ),
                    ('d', [1.09, 0.32], [0.7282, 0.3225]),
                ],
                # Additional elements with extended basis parameters
            },
            'large': {
                'H': [
                    (
                        's',
                        [82.64, 12.41, 2.824, 0.7977, 0.2581, 0.0898],
                        [0.0020, 0.0156, 0.0784, 0.2881, 0.5678, 0.2421],
                    ),
                    ('p', [1.5, 0.4], [0.7, 0.4]),
                    ('d', [0.8], [1.0]),
                ],
                # Large basis set definitions for other elements would follow
            },
        }

        return basis_params.get(quality, {}).get(element, [])

    def get_s_type_basis(self) -> 'BasisSet':
        """
        Get a basis set with only s-type functions.

        Returns:
        --------
        BasisSet
            New basis set with s-type functions only
        """
        s_functions = []

        for basis in self.basis_functions:
            if basis.angular_momentum == (0, 0, 0):
                s_functions.append(basis)

        new_basis = BasisSet(s_functions, f"{self.name}-s")
        return new_basis

    def get_functions_by_center(
        self, center: np.ndarray, tolerance: float = 1e-6
    ) -> List[GaussianBasisFunction]:
        """
        Get all basis functions centered at a specific position.

        Parameters:
        -----------
        center : np.ndarray
            Center position
        tolerance : float
            Position matching tolerance

        Returns:
        --------
        List[GaussianBasisFunction]
            List of basis functions at the specified center
        """
        functions = []
        center = np.asarray(center)

        for basis in self.basis_functions:
            if np.allclose(basis.center, center, atol=tolerance):
                functions.append(basis)

        return functions

    def get_functions_by_type(self, orbital_type: str) -> List[GaussianBasisFunction]:
        """
        Get all basis functions of a specific orbital type.

        Parameters:
        -----------
        orbital_type : str
            Orbital type ('s', 'p', 'd', etc.)

        Returns:
        --------
        List[GaussianBasisFunction]
            List of basis functions of the specified type
        """
        # Map orbital type to angular momentum sum
        type_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}

        if orbital_type not in type_map:
            raise ValueError(f"Unknown orbital type: {orbital_type}")

        target_L = type_map[orbital_type]

        # Filter functions by angular momentum sum
        functions = []
        for basis in self.basis_functions:
            L = sum(basis.angular_momentum)
            if L == target_L:
                functions.append(basis)

        return functions

    def get_unique_centers(self) -> List[np.ndarray]:
        """
        Get list of unique centers in the basis set.

        Returns:
        --------
        List[np.ndarray]
            List of unique centers
        """
        # Use a cache if available
        if 'unique_centers' in self._cache:
            return self._cache['unique_centers']

        # Find unique centers
        centers = set()
        for basis in self.basis_functions:
            centers.add(tuple(basis.center))

        # Convert back to numpy arrays
        unique_centers = [np.array(center) for center in centers]

        # Store in cache
        self._cache['unique_centers'] = unique_centers

        return unique_centers

    def get_function_types(self) -> Dict[str, int]:
        """
        Get count of each function type in the basis set.

        Returns:
        --------
        Dict[str, int]
            Dictionary mapping function types to counts
        """
        types = {'s': 0, 'p': 0, 'd': 0, 'f': 0, 'g': 0, 'h': 0, 'other': 0}

        for basis in self.basis_functions:
            basis_type = basis.get_type()
            if basis_type in types:
                types[basis_type] += 1
            else:
                types['other'] += 1

        return {k: v for k, v in types.items() if v > 0}

    def __len__(self) -> int:
        """Get number of basis functions."""
        return self.n_basis

    def __getitem__(self, index) -> GaussianBasisFunction:
        """Access basis function by index."""
        return self.basis_functions[index]

    def __iter__(self):
        """Iterate through basis functions."""
        return iter(self.basis_functions)

    def __str__(self) -> str:
        """String representation of the basis set."""
        types = self.get_function_types()
        centers = len(self.get_unique_centers())

        return (
            f"BasisSet({self.name}): {self.n_basis} functions "
            f"({', '.join(f'{count} {type}' for type, count in types.items())}) "
            f"at {centers} centers"
        )

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"BasisSet(name='{self.name}', n_basis={self.n_basis})"


class PositronBasis(BasisSet):
    """
    Specialized basis set for positrons with optimizations for antimatter calculations.

    This class extends the standard BasisSet with positron-specific features
    like more diffuse basis functions and specialized basis set combinations.
    """

    def __init__(
        self,
        basis_functions: List[GaussianBasisFunction] = None,
        name: str = "positron",
    ):
        """
        Initialize positron basis set.

        Parameters:
        -----------
        basis_functions : List[GaussianBasisFunction], optional
            List of basis functions
        name : str
            Name or identifier for the basis set
        """
        super().__init__(basis_functions, name)

    def create_for_atom(
        self, element: str, position: np.ndarray, quality: str = 'standard'
    ):
        """
        Create positron-specific basis functions for an atom.

        For positrons, we use more diffuse functions than for electrons.

        Parameters:
        -----------
        element : str
            Element symbol
        position : np.ndarray
            Position of the atom
        quality : str
            Quality of basis ('minimal', 'standard', 'extended', 'large')

        Returns:
        --------
        PositronBasis
            Self reference for method chaining
        """
        # Get positron-specific parameters
        params = self._get_positron_basis_params(element, quality)

        if not params:
            warnings.warn(
                f"No positron basis parameters for {element} with quality {quality}"
            )
            return self

        # Create basis functions
        new_functions = []

        for shell_type, exponents, coefficients in params:
            # Process by shell type
            if shell_type == 's':
                # s-type function (angular momentum = 0,0,0)
                for exp, coef in zip(exponents, coefficients):
                    new_functions.append(
                        GaussianBasisFunction(
                            center=position, exponent=exp, angular_momentum=(0, 0, 0)
                        )
                    )
            elif shell_type == 'p':
                # p-type functions (px, py, pz)
                for exp, coef in zip(exponents, coefficients):
                    for am in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        new_functions.append(
                            GaussianBasisFunction(
                                center=position, exponent=exp, angular_momentum=am
                            )
                        )
            elif shell_type == 'd':
                # d-type functions
                for exp, coef in zip(exponents, coefficients):
                    for am in [
                        (2, 0, 0),
                        (0, 2, 0),
                        (0, 0, 2),
                        (1, 1, 0),
                        (1, 0, 1),
                        (0, 1, 1),
                    ]:
                        new_functions.append(
                            GaussianBasisFunction(
                                center=position, exponent=exp, angular_momentum=am
                            )
                        )

        self.add_functions(new_functions)
        return self

    def _get_positron_basis_params(self, element: str, quality: str) -> List[Tuple]:
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
        List[Tuple]
            List of (shell_type, exponents, coefficients) tuples
        """
        # For positrons, we start with more diffuse basis functions than for electrons

        # Scale factors for positron basis compared to electron basis
        # Positrons need more diffuse functions to represent their wavefunctions
        scale_factor = {
            'minimal': 0.5,  # More diffuse for minimal basis
            'standard': 0.6,  # More diffuse for standard basis
            'extended': 0.7,  # More diffuse for extended basis
            'large': 0.8,  # More diffuse for large basis
        }.get(quality, 0.6)

        # Get regular basis parameters first
        electron_params = super()._get_basis_params(element, quality)
        positron_params = []

        # Adjust exponents for positrons (make them more diffuse)
        for shell_type, exponents, coefficients in electron_params:
            # Make exponents more diffuse by scaling
            positron_exponents = [exp * scale_factor for exp in exponents]
            positron_params.append((shell_type, positron_exponents, coefficients))

        # Add extra diffuse functions for positrons
        if quality != 'minimal':
            # Add an extra diffuse s function
            if positron_params and positron_params[0][0] == 's':
                # Get the most diffuse s exponent and make it even more diffuse
                min_s_exp = min(positron_params[0][1])
                extra_s_exp = min_s_exp * 0.3
                positron_params.append(('s', [extra_s_exp], [1.0]))

            # Add extra diffuse p functions for better description of positron distribution
            if quality in ['extended', 'large']:
                extra_p_exp = 0.2 if quality == 'extended' else 0.15
                positron_params.append(('p', [extra_p_exp], [1.0]))

        return positron_params

    def create_positron_orbital_basis(
        self, center: np.ndarray, quality: str = 'extended'
    ):
        """
        Create a specialized positron basis set for a free positron.

        Parameters:
        -----------
        center : np.ndarray
            Center position for the basis
        quality : str
            Quality of the basis set

        Returns:
        --------
        PositronBasis
            Self reference for method chaining
        """
        # Define exponents based on quality
        if quality == 'minimal':
            s_exponents = [0.2]
            p_exponents = []
        elif quality == 'standard':
            s_exponents = [0.4, 0.2, 0.1]
            p_exponents = [0.3, 0.15]
        elif quality == 'extended':
            s_exponents = [0.8, 0.4, 0.2, 0.1, 0.05]
            p_exponents = [0.6, 0.3, 0.15, 0.075]
            d_exponents = [0.4, 0.2]
        else:  # 'large'
            s_exponents = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
            p_exponents = [0.8, 0.4, 0.2, 0.1, 0.05]
            d_exponents = [0.6, 0.3, 0.15]

        # Create basis functions
        for exp in s_exponents:
            self.add_function(
                GaussianBasisFunction(
                    center=center, exponent=exp, angular_momentum=(0, 0, 0)
                )
            )

        for exp in p_exponents:
            for am in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                self.add_function(
                    GaussianBasisFunction(
                        center=center, exponent=exp, angular_momentum=am
                    )
                )

        if quality in ['extended', 'large']:
            for exp in d_exponents:
                for am in [
                    (2, 0, 0),
                    (0, 2, 0),
                    (0, 0, 2),
                    (1, 1, 0),
                    (1, 0, 1),
                    (0, 1, 1),
                ]:
                    self.add_function(
                        GaussianBasisFunction(
                            center=center, exponent=exp, angular_momentum=am
                        )
                    )

        return self


class MixedMatterBasis:
    """
    Combined basis set for mixed matter/antimatter calculations.

    This class combines electron and positron basis sets into a single
    comprehensive basis for mixed matter/antimatter systems.
    """

    def __init__(self, electron_basis=None, positron_basis=None):
        """
        Initialize mixed matter basis set.

        Parameters:
        -----------
        electron_basis : BasisSet, optional
            Basis set for electrons
        positron_basis : PositronBasis, optional
            Basis set for positrons
        """
        # Handle more flexible initialization
        if electron_basis is None:
            self.electron_basis = BasisSet()
        elif isinstance(electron_basis, BasisSet):
            self.electron_basis = electron_basis
        else:
            self.electron_basis = BasisSet()

        if positron_basis is None:
            self.positron_basis = PositronBasis()
        elif isinstance(positron_basis, PositronBasis):
            self.positron_basis = positron_basis
        else:
            self.positron_basis = PositronBasis()

        # Total number of basis functions
        self.n_electron_basis = self.electron_basis.n_basis
        self.n_positron_basis = self.positron_basis.n_basis
        self.n_total_basis = self.n_electron_basis + self.n_positron_basis

        # Cache for integral calculations
        self._integral_cache = {}

        # For integration with integral engine
        self.integral_engine = None

    def create_for_molecule(self, atoms, e_quality='standard', p_quality='standard'):
        """
        Create basis sets for a molecular system with both electron and positron basis functions.

        Parameters:
        -----------
        atoms : List[Tuple[str, np.ndarray]]
            List of (element, position) tuples for the molecular structure
        e_quality : str
            Quality of electron basis set ('minimal', 'standard', 'extended', 'large')
        p_quality : str
            Quality of positron basis set ('minimal', 'standard', 'extended', 'large')

        Returns:
        --------
        MixedMatterBasis
            Self reference for method chaining
        """
        # Create electron basis
        self.electron_basis = BasisSet()
        self.electron_basis.create_for_molecule(atoms, quality=e_quality)

        # Create positron basis
        self.positron_basis = PositronBasis()
        self.positron_basis.create_for_molecule(atoms, quality=p_quality)

        # Update counts
        self.n_electron_basis = self.electron_basis.n_basis
        self.n_positron_basis = self.positron_basis.n_basis
        self.n_total_basis = self.n_electron_basis + self.n_positron_basis

        # Clear cache
        self._integral_cache = {}

        return self

    def create_positronium_basis(self, quality='extended'):
        """
        Create specialized basis set optimized for positronium calculations.

        Parameters:
        -----------
        quality : str
            Quality level of the basis set ('standard', 'extended', 'large', 'positronium')

        Returns:
        --------
        MixedMatterBasis
            Self reference for method chaining
        """
        # Center at origin
        center = np.array([0.0, 0.0, 0.0])

        # Create electron and positron basis sets
        self.electron_basis = BasisSet(name="electron-positronium")
        self.positron_basis = PositronBasis(name="positron-positronium")

        if quality == 'positronium':
            # Specialized positronium basis with optimized exponents

            # Electron s-functions with exponents optimized for positronium
            e_s_exponents = [0.25, 0.5, 1.0, 2.0, 4.0]
            for exp in e_s_exponents:
                self.electron_basis.add_function(
                    GaussianBasisFunction(
                        center=center, exponent=exp, angular_momentum=(0, 0, 0)
                    )
                )

            # Electron p-functions for better correlation
            e_p_exponents = [0.3, 0.6, 1.2]
            for exp in e_p_exponents:
                for am in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    self.electron_basis.add_function(
                        GaussianBasisFunction(
                            center=center, exponent=exp, angular_momentum=am
                        )
                    )

            # Positron basis functions - more diffuse than electrons
            p_s_exponents = [0.2, 0.4, 0.8, 1.6, 3.2]
            for exp in p_s_exponents:
                self.positron_basis.add_function(
                    GaussianBasisFunction(
                        center=center, exponent=exp, angular_momentum=(0, 0, 0)
                    )
                )

            # Positron p-functions
            p_p_exponents = [0.25, 0.5, 1.0]
            for exp in p_p_exponents:
                for am in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    self.positron_basis.add_function(
                        GaussianBasisFunction(
                            center=center, exponent=exp, angular_momentum=am
                        )
                    )
        else:
            # Create standard basis sets
            atoms = [('H', center)]
            self.electron_basis.create_for_molecule(atoms, quality=quality)
            self.positron_basis.create_positron_orbital_basis(center, quality=quality)

            # Add extra diffuse functions for positronium
            if quality in ['extended', 'large']:
                self.electron_basis.add_function(
                    GaussianBasisFunction(
                        center=center, exponent=0.25, angular_momentum=(0, 0, 0)
                    )
                )
                self.positron_basis.add_function(
                    GaussianBasisFunction(
                        center=center, exponent=0.2, angular_momentum=(0, 0, 0)
                    )
                )

        # Update counts
        self.n_electron_basis = self.electron_basis.n_basis
        self.n_positron_basis = self.positron_basis.n_basis
        self.n_total_basis = self.n_electron_basis + self.n_positron_basis

        # Clear cache
        self._integral_cache = {}

        return self

    def set_integral_engine(self, engine):
        """
        Set the integral engine for integral calculations.

        Parameters:
        -----------
        engine : AntinatureIntegralEngine
            Integral engine instance
        """
        self.integral_engine = engine

    def get_basis_function(self, index):
        """
        Get a basis function by its global index.

        Parameters:
        -----------
        index : int
            Global index of the basis function

        Returns:
        --------
        GaussianBasisFunction
            The basis function at the given index
        """
        if index < 0 or index >= self.n_total_basis:
            raise IndexError(f"Index {index} out of range (0-{self.n_total_basis-1})")

        if index < self.n_electron_basis:
            return self.electron_basis.basis_functions[index]
        else:
            pos_index = index - self.n_electron_basis
            return self.positron_basis.basis_functions[pos_index]

    def overlap_integral(self, i, j):
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
        # Use cache if available
        key = ('overlap', i, j)
        if key in self._integral_cache:
            return self._integral_cache[key]

        # Check if integral engine is set
        if self.integral_engine is None:
            raise ValueError("Integral engine not set. Call set_integral_engine first.")

        # Get basis functions
        basis_i = self.get_basis_function(i)
        basis_j = self.get_basis_function(j)

        # Calculate integral
        result = self.integral_engine.overlap_integral(basis_i, basis_j)

        # Store in cache
        self._integral_cache[key] = result

        return result

    def kinetic_integral(self, i, j):
        """
        Calculate kinetic energy integral between basis functions.

        Parameters:
        -----------
        i, j : int
            Indices of basis functions in the combined basis

        Returns:
        --------
        float
            Kinetic energy integral <i|-∇²/2|j>
        """
        # Use cache if available
        key = ('kinetic', i, j)
        if key in self._integral_cache:
            return self._integral_cache[key]

        # Check if integral engine is set
        if self.integral_engine is None:
            raise ValueError("Integral engine not set. Call set_integral_engine first.")

        # Get basis functions
        basis_i = self.get_basis_function(i)
        basis_j = self.get_basis_function(j)

        # Calculate integral
        result = self.integral_engine.kinetic_integral(basis_i, basis_j)

        # Store in cache
        self._integral_cache[key] = result

        return result

    def nuclear_attraction_integral(self, i, j, nuclear_pos):
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
            Nuclear attraction integral <i|1/r|j>
        """
        # Use cache if available
        pos_tuple = tuple(nuclear_pos)
        key = ('nuclear', i, j, pos_tuple)
        if key in self._integral_cache:
            return self._integral_cache[key]

        # Check if integral engine is set
        if self.integral_engine is None:
            raise ValueError("Integral engine not set. Call set_integral_engine first.")

        # Get basis functions
        basis_i = self.get_basis_function(i)
        basis_j = self.get_basis_function(j)

        # Calculate integral
        result = self.integral_engine.nuclear_attraction_integral(
            basis_i, basis_j, nuclear_pos
        )

        # Store in cache
        self._integral_cache[key] = result

        return result
