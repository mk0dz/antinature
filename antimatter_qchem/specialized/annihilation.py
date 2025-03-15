import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.linalg import eigh
import time

class AnnihilationOperator:
    """
    Optimized implementation of electron-positron annihilation processes.
    """
    
    def __init__(self, 
                 basis_set,
                 wavefunction=None):
        """
        Initialize the annihilation operator.
        
        Parameters:
        -----------
        basis_set : MixedMatterBasis
            Combined basis set for electrons and positrons
        wavefunction : Dict, optional
            Wavefunction information
        """
        self.basis_set = basis_set
        self.wavefunction = wavefunction
        
        # Physical constants in atomic units
        self.c = 137.036  # Speed of light
        self.r0_squared = (1.0 / self.c)**2  # Classical electron radius squared
        self.pi_r0_squared_c = np.pi * self.r0_squared * self.c  # Common prefactor
        
        # Initialize annihilation matrix
        self.matrix = None
        
        # Performance tracking
        self.timing = {}
    
    def build_annihilation_operator(self, use_vectorized=True):
        """
        Construct annihilation operator efficiently.
        
        Parameters:
        -----------
        use_vectorized : bool
            Whether to use vectorized operations for speed
            
        Returns:
        --------
        np.ndarray
            Matrix representation of the annihilation operator
        """
        start_time = time.time()
        
        n_e_basis = self.basis_set.n_electron_basis
        n_p_basis = self.basis_set.n_positron_basis
        
        # Initialize annihilation matrix
        matrix = np.zeros((n_e_basis, n_p_basis))
        
        if use_vectorized:
            # Create evaluation points grid
            grid_points = 50  # Adjust based on accuracy needed
            limit = 8.0  # atomic units
            
            # Create 3D grid efficiently
            x = np.linspace(-limit, limit, grid_points)
            y = np.linspace(-limit, limit, grid_points)
            z = np.linspace(-limit, limit, grid_points)
            
            # Calculate volume element
            dv = (2*limit/grid_points)**3
            
            # Create meshgrid for vectorized evaluation
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
            
            # Evaluate all electron basis functions at all points
            e_values = np.zeros((n_e_basis, len(points)))
            for i, e_func in enumerate(self.basis_set.electron_basis.basis_functions):
                for j, point in enumerate(points):
                    e_values[i, j] = e_func.evaluate(point)
            
            # Evaluate all positron basis functions at all points
            p_values = np.zeros((n_p_basis, len(points)))
            for i, p_func in enumerate(self.basis_set.positron_basis.basis_functions):
                for j, point in enumerate(points):
                    p_values[i, j] = p_func.evaluate(point)
            
            # Calculate overlap efficiently
            for i in range(n_e_basis):
                for j in range(n_p_basis):
                    matrix[i, j] = np.sum(e_values[i] * p_values[j]) * dv
        else:
            # Compute using analytical formulas if available
            for i in range(n_e_basis):
                e_func = self.basis_set.electron_basis.basis_functions[i]
                for j in range(n_p_basis):
                    p_func = self.basis_set.positron_basis.basis_functions[j]
                    
                    # For Gaussian basis functions, this is an overlap integral
                    # with centers at different positions
                    matrix[i, j] = self.basis_set.integral_engine.overlap_integral(e_func, p_func)
        
        self.matrix = matrix
        
        end_time = time.time()
        self.timing['build_operator'] = end_time - start_time
        
        return matrix
    
    def calculate_annihilation_rate(self, electron_density=None, positron_density=None):
        """
        Calculate electron-positron annihilation rate from density matrices.
        
        Parameters:
        -----------
        electron_density : np.ndarray, optional
            Electron density matrix
        positron_density : np.ndarray, optional
            Positron density matrix
            
        Returns:
        --------
        float
            Annihilation rate in atomic units
        """
        start_time = time.time()
        
        # Use provided densities or extract from wavefunction
        P_e = electron_density
        if P_e is None and self.wavefunction is not None:
            P_e = self.wavefunction.get('P_electron')
        
        P_p = positron_density
        if P_p is None and self.wavefunction is not None:
            P_p = self.wavefunction.get('P_positron')
        
        if P_e is None or P_p is None:
            return 0.0
        
        # Ensure annihilation matrix is built
        if self.matrix is None:
            self.build_annihilation_operator()
        
        # Calculate rate using matrix multiplication for efficiency
        # Γ = Tr(P_e A P_p A^T) * π*r₀²*c
        rate = np.trace(P_e @ self.matrix @ P_p @ self.matrix.T)
        
        # Apply physical prefactor
        rate *= self.pi_r0_squared_c
        
        end_time = time.time()
        self.timing['calculate_rate'] = end_time - start_time
        
        return rate
    
    def analyze_annihilation_channels(self, wavefunction=None):
        """
        Analyze different annihilation channels (2γ vs. 3γ).
        
        Parameters:
        -----------
        wavefunction : Dict, optional
            Wavefunction information
            
        Returns:
        --------
        Dict
            Breakdown of annihilation channels
        """
        start_time = time.time()
        
        # Use provided wavefunction or the stored one
        if wavefunction is not None:
            self.wavefunction = wavefunction
        
        if self.wavefunction is None:
            return {'two_gamma': 0.0, 'three_gamma': 0.0, 'total': 0.0}
        
        # Extract MO coefficients
        C_e = self.wavefunction.get('C_electron')
        C_p = self.wavefunction.get('C_positron')
        
        if C_e is None or C_p is None:
            return {'two_gamma': 0.0, 'three_gamma': 0.0, 'total': 0.0}
        
        # Ensure annihilation matrix is built
        if self.matrix is None:
            self.build_annihilation_operator()
        
        # Transform annihilation operator to MO basis efficiently
        ann_mo = C_e.T @ self.matrix @ C_p
        
        # Extract occupied orbitals
        n_e_occ = self.wavefunction.get('n_electrons', 0) // 2
        if n_e_occ == 0 and self.wavefunction.get('n_electrons', 0) > 0:
            n_e_occ = 1  # Handle odd number
            
        n_p_occ = self.wavefunction.get('n_positrons', 0) // 2
        if n_p_occ == 0 and self.wavefunction.get('n_positrons', 0) > 0:
            n_p_occ = 1  # Handle odd number
        
        # Calculate annihilation rates for different channels
        
        # 2γ annihilation (singlet state, 75% probability)
        rate_two_gamma = 0.0
        for i in range(min(n_e_occ, C_e.shape[1])):
            for j in range(min(n_p_occ, C_p.shape[1])):
                rate_two_gamma += 0.75 * self.pi_r0_squared_c * ann_mo[i, j]**2
        
        # 3γ annihilation (triplet state, 25% probability)
        # Ratio of 3γ to 2γ rates from theoretical calculations
        triplet_factor = 1.0/1115.0
        rate_three_gamma = 0.0
        for i in range(min(n_e_occ, C_e.shape[1])):
            for j in range(min(n_p_occ, C_p.shape[1])):
                rate_three_gamma += 0.25 * self.pi_r0_squared_c * triplet_factor * ann_mo[i, j]**2
        
        # Total rate and ratio
        total_rate = rate_two_gamma + rate_three_gamma
        ratio = rate_two_gamma / rate_three_gamma if rate_three_gamma > 1e-30 else float('inf')
        
        end_time = time.time()
        self.timing['analyze_channels'] = end_time - start_time
        
        return {
            'two_gamma': rate_two_gamma,
            'three_gamma': rate_three_gamma,
            'total': total_rate,
            'ratio_2g_3g': ratio
        }
    
    def calculate_lifetime(self, annihilation_rate=None):
        """
        Calculate lifetime based on annihilation rate.
        
        Parameters:
        -----------
        annihilation_rate : float, optional
            Annihilation rate or calculate if None
            
        Returns:
        --------
        Dict
            Lifetime in various units
        """
        if annihilation_rate is None:
            annihilation_rate = self.calculate_annihilation_rate()
        
        if annihilation_rate <= 1e-30:
            return {
                'lifetime_au': float('inf'),
                'lifetime_s': float('inf'),
                'lifetime_ns': float('inf')
            }
        
        # Convert from atomic units to seconds and nanoseconds
        au_to_seconds = 2.4188843265e-17
        
        lifetime_au = 1.0 / annihilation_rate
        lifetime_s = lifetime_au * au_to_seconds
        lifetime_ns = lifetime_s * 1e9
        
        return {
            'lifetime_au': lifetime_au,
            'lifetime_s': lifetime_s,
            'lifetime_ns': lifetime_ns
        }
    
    def visualize_annihilation_density(self, grid_dims=(50, 50, 50), limits=(-5.0, 5.0)):
        """
        Calculate and visualize annihilation density.
        
        Parameters:
        -----------
        grid_dims : Tuple[int, int, int]
            Dimensions of the visualization grid
        limits : Tuple[float, float]
            Spatial limits for visualization
            
        Returns:
        --------
        Dict
            Grid and density data
        """
        if self.wavefunction is None:
            return None
        
        # Extract density matrices
        P_e = self.wavefunction.get('P_electron')
        P_p = self.wavefunction.get('P_positron')
        
        if P_e is None or P_p is None:
            return None
        
        # Create visualization grid
        nx, ny, nz = grid_dims
        xmin, xmax = limits
        
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(xmin, xmax, ny)
        z = np.linspace(xmin, xmax, nz)
        
        # For better performance, calculate on a coarser grid first
        density = np.zeros((nx, ny, nz))
        
        # Calculate electron and positron densities in real space efficiently
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                for k, zk in enumerate(z):
                    r = np.array([xi, yj, zk])
                    
                    # Calculate electron density at this point
                    e_density = self._calculate_density_at_point(r, P_e, self.basis_set.electron_basis)
                    
                    # Calculate positron density at this point
                    p_density = self._calculate_density_at_point(r, P_p, self.basis_set.positron_basis)
                    
                    # Annihilation density is proportional to product of densities
                    density[i, j, k] = e_density * p_density
        
        # Scale by annihilation constant
        density *= self.pi_r0_squared_c
        
        return {
            'x': x,
            'y': y,
            'z': z,
            'density': density
        }
    
    def _calculate_density_at_point(self, r, P, basis):
        """
        Calculate density at a given point efficiently.
        
        Parameters:
        -----------
        r : np.ndarray
            Position vector
        P : np.ndarray
            Density matrix
        basis : BasisSet
            Basis set
            
        Returns:
        --------
        float
            Density at the given point
        """
        # Evaluate all basis functions at this point
        basis_vals = np.array([func.evaluate(r) for func in basis.basis_functions])
        
        # Calculate density using matrix multiplication
        density = np.dot(basis_vals, np.dot(P, basis_vals))
        
        return density