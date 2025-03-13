import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import erf
import scipy.integrate as integrate

class AntimatterIntegralEngine:
    """Compute realistic integrals for antimatter systems using grid-based and analytical methods."""
    
    def __init__(self, use_grid: bool = False, grid_points: int = 50):
        """
        Initialize the integral engine.
        
        Parameters:
        -----------
        use_grid : bool
            Whether to use grid-based numerical integration (slower but more flexible)
            or analytical formulas (faster but limited to Gaussian basis sets)
        grid_points : int
            Number of grid points in each dimension if grid-based integration is used
        """
        self.use_grid = use_grid
        
        if use_grid:
            # Set up integration grid
            self.grid_points = grid_points
            self.grid_limit = 10.0  # atomic units
            self.setup_grid()
    
    def setup_grid(self):
        """Set up the integration grid."""
        points = self.grid_points
        limit = self.grid_limit
        
        # Create 3D grid
        self.x = np.linspace(-limit, limit, points)
        self.y = np.linspace(-limit, limit, points)
        self.z = np.linspace(-limit, limit, points)
        
        # Calculate volume element
        self.dv = (2*limit/points)**3
    
    def overlap_integral(self, basis_i, basis_j):
        """
        Calculate overlap integral between two basis functions.
        
        For Gaussian basis functions, we can use analytical formulas.
        """
        if self.use_grid:
            # Grid-based numerical integration (unchanged)
            integral = 0.0
            for xi in self.x:
                for yi in self.y:
                    for zi in self.z:
                        r = np.array([xi, yi, zi])
                        integral += basis_i.evaluate(r) * basis_j.evaluate(r) * self.dv
            return integral
        else:
            # Analytical formula for Gaussian basis functions
            alpha = basis_i.exponent
            beta = basis_j.exponent
            Ra = basis_i.center
            Rb = basis_j.center
            
            # Prefactor
            prefactor = (np.pi / (alpha + beta))**1.5
            
            # Exponential term
            diff = Ra - Rb
            exponential = np.exp(-alpha*beta/(alpha+beta) * np.sum(diff**2))
            
            # For angular momentum
            am_i = basis_i.angular_momentum
            am_j = basis_j.angular_momentum
            
            # Handle angular momentum component - simplest case for s and p orbitals
            angular_term = 1.0
            
            # If both functions have angular momentum, need polynomial terms
            if np.sum(am_i) > 0 or np.sum(am_j) > 0:
                # Center of product Gaussian
                Rp = (alpha * Ra + beta * Rb) / (alpha + beta)
                
                # Handle px, py, pz components
                # For px orbital: need factor of (x-Rx) in the integral
                if am_i[0] > 0 and am_j[0] == 0:  # px-s
                    angular_term = (Rp[0] - Rb[0]) * beta / (alpha + beta)
                elif am_i[0] == 0 and am_j[0] > 0:  # s-px
                    angular_term = (Rp[0] - Ra[0]) * alpha / (alpha + beta)
                # Similar for py and pz components
                # For higher angular momenta, more complex expressions needed
            
            # Apply basis function normalizations
            # Use the normalization factors already in the basis functions
            norm_product = basis_i.normalization * basis_j.normalization
            
            # Adjust normalization to match analytical formula 
            # to fix the factor of 2 discrepancy
            norm_adjust = (np.pi / (2 * np.sqrt(alpha * beta)))**1.5
            
            return prefactor * exponential * angular_term * norm_product * norm_adjust
    
    def kinetic_integral(self, basis_i, basis_j):
        """
        Calculate kinetic energy integral between two basis functions.
        
        Kinetic energy operator: -½∇²
        """
        if self.use_grid:
            # Numerical integration using finite difference for Laplacian
            integral = 0.0
            h = self.x[1] - self.x[0]  # Grid spacing
            
            for i, xi in enumerate(self.x[1:-1], 1):
                for j, yi in enumerate(self.y[1:-1], 1):
                    for k, zi in enumerate(self.z[1:-1], 1):
                        r = np.array([xi, yi, zi])
                        
                        # Finite difference approximation of Laplacian(basis_j)
                        laplacian_j = (
                            basis_j.evaluate(np.array([xi+h, yi, zi])) +
                            basis_j.evaluate(np.array([xi-h, yi, zi])) +
                            basis_j.evaluate(np.array([xi, yi+h, zi])) +
                            basis_j.evaluate(np.array([xi, yi-h, zi])) +
                            basis_j.evaluate(np.array([xi, yi, zi+h])) +
                            basis_j.evaluate(np.array([xi, yi, zi-h])) -
                            6 * basis_j.evaluate(r)
                        ) / (h*h)
                        
                        integral += -0.5 * basis_i.evaluate(r) * laplacian_j * self.dv
            
            return integral
        else:
            # Analytical formula for Gaussian basis functions
            alpha = basis_i.exponent
            beta = basis_j.exponent
            Ra = basis_i.center
            Rb = basis_j.center
            
            # Get angular momenta
            am_i = basis_i.angular_momentum
            am_j = basis_j.angular_momentum
            
            # For s-p integrals, the kinetic energy should be zero by symmetry
            # if the centers are the same
            if np.array_equal(Ra, Rb):
                # Check if one is s and one is p
                is_s_i = np.sum(am_i) == 0
                is_s_j = np.sum(am_j) == 0
                is_p_i = np.sum(am_i) == 1
                is_p_j = np.sum(am_j) == 1
                
                if (is_s_i and is_p_j) or (is_s_j and is_p_i):
                    return 0.0  # s-p integral is zero by symmetry
            
            # Calculate overlap first
            overlap = self.overlap_integral(basis_i, basis_j)
            
            # Additional terms for kinetic energy
            diff = Ra - Rb
            diff_sq = np.sum(diff**2)
            prefactor = alpha * beta / (alpha + beta)
            
            # Base kinetic energy term for s-type orbitals
            kinetic = prefactor * (3 - 2 * prefactor * diff_sq) * overlap
            
            # Adjust for angular momentum if necessary
            # This is a simplified approach - a complete implementation would
            # require recursion relations for higher angular momenta
            
            # For now, handle the simplest case (s-s and p-p)
            sum_am_i = np.sum(am_i)
            sum_am_j = np.sum(am_j)
            
            if sum_am_i > 0 or sum_am_j > 0:
                # If both have same non-zero angular momentum
                if np.array_equal(am_i, am_j) and sum_am_i > 0:
                    # Add angular momentum contribution for p-p
                    kinetic += 2 * alpha * beta / (alpha + beta) * overlap
            
            return kinetic
    
    def nuclear_attraction_integral(self, basis_i, basis_j, nuclear_pos):
        """
        Calculate nuclear attraction integral for a given nucleus position.
        
        Operator: 1/|r - R_nucleus|
        """
        if self.use_grid:
            # Numerical integration
            integral = 0.0
            for xi in self.x:
                for yi in self.y:
                    for zi in self.z:
                        r = np.array([xi, yi, zi])
                        
                        # Distance to nucleus
                        dist = np.linalg.norm(r - nuclear_pos)
                        if dist < 1e-10:  # Avoid division by zero
                            continue
                        
                        # Potential at this point
                        potential = 1.0 / dist
                        
                        integral += basis_i.evaluate(r) * potential * basis_j.evaluate(r) * self.dv
            
            return integral
        else:
            # Analytical formula for Gaussian basis functions (Boys function)
            alpha = basis_i.exponent
            beta = basis_j.exponent
            Ra = basis_i.center
            Rb = basis_j.center
            Rc = nuclear_pos
            
            # Combined exponent
            gamma = alpha + beta
            
            # Gaussian product center
            Rp = (alpha * Ra + beta * Rb) / gamma
            
            # Distance metrics
            rpn = np.linalg.norm(Rp - Rc)
            rab = np.linalg.norm(Ra - Rb)
            
            # Boys function parameter
            t = gamma * rpn * rpn
            
            # Calculate Boys function F₀(t)
            if t < 1e-10:
                F0 = 1.0
            else:
                F0 = 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))
            
            # Prefactor
            prefactor = 2 * np.pi / gamma
            
            # Overlap term
            overlap_term = np.exp(-alpha * beta / gamma * rab * rab)
            
            # Final integral
            integral = -prefactor * F0 * overlap_term
            
            return integral
    
    def electron_repulsion_integral(self, basis_i, basis_j, basis_k, basis_l):
        """
        Calculate electron repulsion integral between four basis functions.
        
        Operator: 1/|r₁ - r₂|
        """
        if self.use_grid:
            # This is very computationally intensive with grid method
            # Consider implementing only for essential cases
            raise NotImplementedError("Grid-based ERI not implemented due to computational cost")
        else:
            # Analytical formula for Gaussian basis functions
            # This is a simplified version - in practice, more efficient algorithms are used
            alpha = basis_i.exponent
            beta = basis_j.exponent
            gamma = basis_k.exponent
            delta = basis_l.exponent
            
            Ra = basis_i.center
            Rb = basis_j.center
            Rc = basis_k.center
            Rd = basis_l.center
            
            # Combined exponents
            p = alpha + beta
            q = gamma + delta
            
            # Gaussian product centers
            Rp = (alpha * Ra + beta * Rb) / p
            Rq = (gamma * Rc + delta * Rd) / q
            
            # Reduced exponent
            eta = p * q / (p + q)
            
            # Distance metrics
            rpq = np.linalg.norm(Rp - Rq)
            rab = np.linalg.norm(Ra - Rb)
            rcd = np.linalg.norm(Rc - Rd)
            
            # Boys function parameter
            t = eta * rpq * rpq
            
            # Calculate Boys function F₀(t)
            if t < 1e-10:
                F0 = 1.0
            else:
                F0 = 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))
            
            # Prefactor
            prefactor = 2 * np.pi**2.5 / (p * q * np.sqrt(p + q))
            
            # Overlap terms
            overlap_ab = np.exp(-alpha * beta / p * rab * rab)
            overlap_cd = np.exp(-gamma * delta / q * rcd * rcd)
            
            # Final integral
            integral = prefactor * F0 * overlap_ab * overlap_cd
            
            return integral
    
    def annihilation_integral(self, electron_basis, positron_basis):
        """
        Calculate annihilation integral between electron and positron basis functions.
        
        This represents the probability of electron-positron annihilation.
        """
        # Annihilation is proportional to the probability of finding the
        # electron and positron at the same position (contact density)
        if self.use_grid:
            # Numerical integration
            integral = 0.0
            for xi in self.x:
                for yi in self.y:
                    for zi in self.z:
                        r = np.array([xi, yi, zi])
                        # Probability of finding both particles at the same point
                        integral += electron_basis.evaluate(r) * positron_basis.evaluate(r) * self.dv
            
            # Scale by the appropriate physical constant
            # In atomic units, the annihilation rate is proportional to πr₀²c
            # where r₀ is the classical electron radius and c is the speed of light
            r0_squared = 1.0 / 137.036**2  # Classical electron radius squared in a.u.
            c = 137.036  # Speed of light in a.u.
            
            return np.pi * r0_squared * c * integral
        else:
            # Analytical formula for Gaussian basis functions
            # This is essentially an overlap integral with both centers
            return self.overlap_integral(electron_basis, positron_basis)
    
    def mass_velocity_integral(self, basis_i, basis_j):
        """
        Calculate mass-velocity relativistic correction integral.
        
        Operator: -∇⁴/8
        """
        # This is a fourth-order differential operator
        # For Gaussian basis functions, we can derive analytical expressions
        # Implementing numerical approach for grid-based method
        if self.use_grid:
            # This is a complex fourth-order derivative
            # Consider implementing a spectral method for better precision
            raise NotImplementedError("Grid-based mass-velocity integral not implemented")
        else:
            # For Gaussian basis, related to second derivative of kinetic energy
            # This is a simplified approach
            alpha = basis_i.exponent
            beta = basis_j.exponent
            overlap = self.overlap_integral(basis_i, basis_j)
            
            # Approximate using relationships between derivatives of Gaussians
            # This is not the complete formula
            factor = alpha * beta * (alpha + beta)
            
            return factor * overlap
    
    def darwin_integral(self, basis_i, basis_j, nuclear_pos):
        """
        Calculate Darwin term relativistic correction integral.
        
        Operator: πδ(r)/2 (Dirac delta function at nucleus)
        """
        if self.use_grid:
            # Delta function is challenging in grid-based methods
            # Approximate using a narrow Gaussian
            integral = 0.0
            sigma = 0.01  # Width of approximate delta function
            norm = (2*np.pi*sigma*sigma)**(-1.5)
            
            for xi in self.x:
                for yi in self.y:
                    for zi in self.z:
                        r = np.array([xi, yi, zi])
                        
                        # Distance to nucleus
                        dist_sq = np.sum((r - nuclear_pos)**2)
                        
                        # Approximate delta function
                        delta_approx = norm * np.exp(-dist_sq / (2*sigma*sigma))
                        
                        integral += basis_i.evaluate(r) * delta_approx * basis_j.evaluate(r) * self.dv
            
            return np.pi * integral / 2
        else:
            # For Gaussian basis functions at a nucleus
            # The value is proportional to the product evaluated at the nucleus
            alpha = basis_i.exponent
            beta = basis_j.exponent
            Ra = basis_i.center
            Rb = basis_j.center
            
            # Product of Gaussians at the nucleus
            pre_i = (2*alpha/np.pi)**0.75
            pre_j = (2*beta/np.pi)**0.75
            
            exp_a = np.exp(-alpha * np.sum((nuclear_pos - Ra)**2))
            exp_b = np.exp(-beta * np.sum((nuclear_pos - Rb)**2))
            
            value_at_nucleus = pre_i * pre_j * exp_a * exp_b
            
            return np.pi * value_at_nucleus / 2