import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.linalg import eigh, inv, sqrtm
import time

class AntimatterSCF:
    """
    Optimized Self-Consistent Field solver for antimatter systems.
    """
    
    def __init__(self, 
                 hamiltonian,
                 basis_set,
                 molecular_data,
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-6,
                 use_diis: bool = True,
                 damping_factor: float = 0.5):
        """
        Initialize the SCF solver with improved convergence techniques.
        
        Parameters:
        -----------
        hamiltonian : Dict
            Dictionary of Hamiltonian matrices
        basis_set : MixedMatterBasis
            Basis set for the calculation
        molecular_data : MolecularData
            Molecular structure information
        max_iterations : int
            Maximum number of SCF iterations
        convergence_threshold : float
            Threshold for convergence checking
        use_diis : bool
            Whether to use DIIS acceleration
        damping_factor : float
            Damping factor for density matrix updates (0-1)
        """
        self.hamiltonian = hamiltonian
        self.basis_set = basis_set
        self.molecular_data = molecular_data
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_diis = use_diis
        self.damping_factor = damping_factor
        
        # Extract key information
        self.n_electrons = molecular_data.n_electrons
        self.n_positrons = molecular_data.n_positrons
        
        # Extract matrices from hamiltonian
        self.S = hamiltonian.get('overlap')
        self.H_core_e = hamiltonian.get('H_core_electron')
        self.H_core_p = hamiltonian.get('H_core_positron')
        self.V_nuc = molecular_data.get_nuclear_repulsion_energy()
        
        # Get ERI matrices (or functions)
        self.ERI_e = hamiltonian.get('electron_repulsion')
        self.ERI_p = hamiltonian.get('positron_repulsion')
        self.ERI_ep = hamiltonian.get('electron_positron_attraction')
        
        # Initialize density matrices and energies
        self.P_e = None
        self.P_p = None
        self.E_e = None  # Orbital energies
        self.E_p = None
        self.C_e = None  # Orbital coefficients
        self.C_p = None
        self.energy = 0.0
        
        # For DIIS acceleration
        if use_diis:
            self.diis_start = 3
            self.diis_dim = 6
            self.diis_error_vectors_e = []
            self.diis_fock_matrices_e = []
            self.diis_error_vectors_p = []
            self.diis_fock_matrices_p = []
    
    def initial_guess(self):
        """
        Generate an efficient initial guess for the SCF procedure.
        """
        # For electrons
        if self.n_electrons > 0 and self.H_core_e is not None:
            # Use core Hamiltonian eigenvalues for initial guess
            n_e_basis = self.basis_set.n_electron_basis
            S_e = self.S[:n_e_basis, :n_e_basis]
            e_vals, e_vecs = eigh(self.H_core_e, S_e)
            
            # Store orbital energies and coefficients
            self.E_e = e_vals
            self.C_e = e_vecs
            
            # Form initial density matrix
            self.P_e = np.zeros((n_e_basis, n_e_basis))
            n_occ = self.n_electrons // 2  # Assuming closed-shell
            for i in range(n_occ):
                self.P_e += 2.0 * np.outer(e_vecs[:, i], e_vecs[:, i])
        
        # For positrons
        if self.n_positrons > 0 and self.H_core_p is not None:
            # Similar process for positrons
            n_p_basis = self.basis_set.n_positron_basis
            n_e_basis = self.basis_set.n_electron_basis
            S_p = self.S[n_e_basis:, n_e_basis:]
            p_vals, p_vecs = eigh(self.H_core_p, S_p)
            
            # Store orbital energies and coefficients
            self.E_p = p_vals
            self.C_p = p_vecs
            
            # Form initial density matrix
            self.P_p = np.zeros((n_p_basis, n_p_basis))
            n_occ = self.n_positrons // 2  # Assuming closed-shell
            for i in range(n_occ):
                self.P_p += 2.0 * np.outer(p_vecs[:, i], p_vecs[:, i])
    
    def build_fock_matrix_e(self):
        """
        Build electron Fock matrix efficiently.
        """
        if self.H_core_e is None:
            return None
        
        n_e_basis = self.basis_set.n_electron_basis
        F_e = self.H_core_e.copy()
        
        # Add two-electron contributions if available
        if self.ERI_e is not None and self.P_e is not None:
            # Compute J and K matrices
            J = np.zeros((n_e_basis, n_e_basis))
            K = np.zeros((n_e_basis, n_e_basis))
            
            for mu in range(n_e_basis):
                for nu in range(n_e_basis):
                    for lambda_ in range(n_e_basis):
                        for sigma in range(n_e_basis):
                            J[mu, nu] += self.P_e[lambda_, sigma] * self.ERI_e[mu, nu, lambda_, sigma]
                            K[mu, nu] += self.P_e[lambda_, sigma] * self.ERI_e[mu, lambda_, nu, sigma]
            
            F_e += 2.0 * J - K
        
        # Add electron-positron interaction if available
        if self.ERI_ep is not None and self.P_p is not None:
            # Add electron-positron attraction to electron Fock matrix
            for mu in range(n_e_basis):
                for nu in range(n_e_basis):
                    for lambda_ in range(self.basis_set.n_positron_basis):
                        for sigma in range(self.basis_set.n_positron_basis):
                            F_e[mu, nu] += self.P_p[lambda_, sigma] * self.ERI_ep[mu, nu, lambda_, sigma]
        
        return F_e
    
    def build_fock_matrix_p(self):
        """
        Build positron Fock matrix efficiently.
        """
        if self.H_core_p is None:
            return None
        
        n_p_basis = self.basis_set.n_positron_basis
        F_p = self.H_core_p.copy()
        
        # Add two-positron contributions if available
        if self.ERI_p is not None and self.P_p is not None:
            # Compute J and K matrices for positrons
            J = np.zeros((n_p_basis, n_p_basis))
            K = np.zeros((n_p_basis, n_p_basis))
            
            for mu in range(n_p_basis):
                for nu in range(n_p_basis):
                    for lambda_ in range(n_p_basis):
                        for sigma in range(n_p_basis):
                            J[mu, nu] += self.P_p[lambda_, sigma] * self.ERI_p[mu, nu, lambda_, sigma]
                            K[mu, nu] += self.P_p[lambda_, sigma] * self.ERI_p[mu, lambda_, nu, sigma]
            
            F_p += 2.0 * J - K
        
        # Add electron-positron interaction if available
        if self.ERI_ep is not None and self.P_e is not None:
            # Add electron-positron attraction to positron Fock matrix
            n_e_basis = self.basis_set.n_electron_basis
            for mu in range(n_p_basis):
                for nu in range(n_p_basis):
                    for lambda_ in range(n_e_basis):
                        for sigma in range(n_e_basis):
                            F_p[mu, nu] += self.P_e[lambda_, sigma] * self.ERI_ep[lambda_, sigma, mu, nu]
        
        return F_p
    
    def compute_energy(self):
        """
        Calculate the total SCF energy efficiently.
        """
        energy = self.V_nuc  # Start with nuclear repulsion
        
        # Add electronic contribution
        if self.P_e is not None and self.H_core_e is not None:
            energy += np.sum(self.P_e * (self.H_core_e + self.build_fock_matrix_e())) / 2.0
        
        # Add positronic contribution
        if self.P_p is not None and self.H_core_p is not None:
            energy += np.sum(self.P_p * (self.H_core_p + self.build_fock_matrix_p())) / 2.0
        
        # Store energy
        self.energy = energy
        return energy
    
    def diis_extrapolation(self, F, P, S, error_vectors, fock_matrices):
        """
        Apply DIIS (Direct Inversion of Iterative Subspace) to accelerate convergence.
        """
        # Calculate error vector: FPS - SPF
        error = F @ P @ S - S @ P @ F
        error_norm = np.linalg.norm(error)
        error_vector = error.flatten()
        
        # Add to history
        error_vectors.append(error_vector)
        fock_matrices.append(F.copy())
        
        # Keep only the last diis_dim matrices
        if len(error_vectors) > self.diis_dim:
            error_vectors.pop(0)
            fock_matrices.pop(0)
        
        n_diis = len(error_vectors)
        
        # Build B matrix for DIIS
        B = np.zeros((n_diis + 1, n_diis + 1))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        
        for i in range(n_diis):
            for j in range(n_diis):
                B[i, j] = np.dot(error_vectors[i], error_vectors[j])
        
        # Solve DIIS equations
        rhs = np.zeros(n_diis + 1)
        rhs[-1] = -1
        
        try:
            coeffs = np.linalg.solve(B, rhs)
            
            # Form extrapolated Fock matrix
            F_diis = np.zeros_like(F)
            for i in range(n_diis):
                F_diis += coeffs[i] * fock_matrices[i]
            
            return F_diis, error_norm
        except np.linalg.LinAlgError:
            # If DIIS fails, return original matrix
            return F, error_norm
    
    def solve_scf(self):
        """
        Perform SCF calculation with optimized algorithms and convergence acceleration.
        """
        # Generate initial guess
        self.initial_guess()
        
        # Main SCF loop
        energy_prev = 0.0
        converged = False
        iterations = 0
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            iterations = iteration + 1
            
            # Build Fock matrices
            F_e = self.build_fock_matrix_e()
            F_p = self.build_fock_matrix_p()
            
            # DIIS acceleration for electrons
            max_error = 0.0
            if self.use_diis and iteration >= self.diis_start and F_e is not None:
                n_e_basis = self.basis_set.n_electron_basis
                S_e = self.S[:n_e_basis, :n_e_basis]
                F_e, error_e = self.diis_extrapolation(
                    F_e, self.P_e, S_e, self.diis_error_vectors_e, self.diis_fock_matrices_e
                )
                max_error = max(max_error, error_e)
            
            # DIIS acceleration for positrons
            if self.use_diis and iteration >= self.diis_start and F_p is not None:
                n_p_basis = self.basis_set.n_positron_basis
                n_e_basis = self.basis_set.n_electron_basis
                S_p = self.S[n_e_basis:, n_e_basis:]
                F_p, error_p = self.diis_extrapolation(
                    F_p, self.P_p, S_p, self.diis_error_vectors_p, self.diis_fock_matrices_p
                )
                max_error = max(max_error, error_p)
            
            # Solve eigenvalue problem for electrons
            if F_e is not None:
                n_e_basis = self.basis_set.n_electron_basis
                S_e = self.S[:n_e_basis, :n_e_basis]
                
                # Prepare orthogonalization matrix
                X = sqrtm(inv(S_e))
                
                # Transform Fock matrix
                F_ortho = X.T @ F_e @ X
                
                # Solve eigenvalue problem
                e_vals, C_ortho = eigh(F_ortho)
                
                # Back-transform coefficients
                C_e_new = X @ C_ortho
                
                # Store orbital energies and coefficients
                self.E_e = e_vals
                self.C_e = C_e_new
                
                # Form new density matrix
                P_e_new = np.zeros_like(self.P_e)
                n_occ = self.n_electrons // 2
                for i in range(n_occ):
                    P_e_new += 2.0 * np.outer(C_e_new[:, i], C_e_new[:, i])
                
                # Apply damping for improved convergence
                if iteration > 0:
                    self.P_e = self.damping_factor * P_e_new + (1 - self.damping_factor) * self.P_e
                else:
                    self.P_e = P_e_new
            
            # Solve eigenvalue problem for positrons
            if F_p is not None:
                n_p_basis = self.basis_set.n_positron_basis
                n_e_basis = self.basis_set.n_electron_basis
                S_p = self.S[n_e_basis:, n_e_basis:]
                
                # Prepare orthogonalization matrix
                X = sqrtm(inv(S_p))
                
                # Transform Fock matrix
                F_ortho = X.T @ F_p @ X
                
                # Solve eigenvalue problem
                p_vals, C_ortho = eigh(F_ortho)
                
                # Back-transform coefficients
                C_p_new = X @ C_ortho
                
                # Store orbital energies and coefficients
                self.E_p = p_vals
                self.C_p = C_p_new
                
                # Form new density matrix
                P_p_new = np.zeros_like(self.P_p)
                n_occ = self.n_positrons // 2
                for i in range(n_occ):
                    P_p_new += 2.0 * np.outer(C_p_new[:, i], C_p_new[:, i])
                
                # Apply damping for improved convergence
                if iteration > 0:
                    self.P_p = self.damping_factor * P_p_new + (1 - self.damping_factor) * self.P_p
                else:
                    self.P_p = P_p_new
            
            # Compute energy
            energy = self.compute_energy()
            
            # Check convergence
            energy_diff = abs(energy - energy_prev)
            energy_prev = energy
            
            # Print progress
            print(f"Iteration {iteration+1}: Energy = {energy:.10f}, ΔE = {energy_diff:.10f}, Error = {max_error:.10f}")
            
            if energy_diff < self.convergence_threshold and max_error < self.convergence_threshold * 10:
                converged = True
                break
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        print(f"SCF {'converged' if converged else 'not converged'} in {iterations} iterations")
        print(f"Final energy: {energy:.10f} Hartree")
        print(f"Calculation time: {computation_time:.2f} seconds")
        
        # Prepare results
        results = {
            'energy': energy,
            'converged': converged,
            'iterations': iterations,
            'E_electron': self.E_e,
            'E_positron': self.E_p,
            'C_electron': self.C_e,
            'C_positron': self.C_p,
            'P_electron': self.P_e,
            'P_positron': self.P_p,
            'computation_time': computation_time
        }
        
        return results