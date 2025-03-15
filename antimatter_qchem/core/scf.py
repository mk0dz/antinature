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
                 use_diis: bool = True):
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
        """
        self.hamiltonian = hamiltonian
        self.basis_set = basis_set
        self.molecular_data = molecular_data
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_diis = use_diis
        
        # Extract key information
        self.n_electrons = molecular_data.n_electrons
        self.n_positrons = molecular_data.n_positrons
        
        # Extract matrices from hamiltonian
        self.S = hamiltonian.get('overlap')
        self.H_core_e = hamiltonian.get('H_core_electron')
        self.H_core_p = hamiltonian.get('H_core_positron')
        self.ERI = hamiltonian.get('electron_repulsion')
        
        # Initialize density matrices
        self.P_e = None
        self.P_p = None
        
        # For DIIS acceleration
        if use_diis:
            self.diis_start = 3
            self.diis_dim = 6
            self.diis_error_vectors = []
            self.diis_fock_matrices = []
    
    def initial_guess(self):
        """
        Generate an efficient initial guess for the SCF procedure.
        """
        # For electrons
        if self.n_electrons > 0:
            # Use core Hamiltonian eigenvalues for initial guess
            e_vals, e_vecs = eigh(self.H_core_e, self.S[:self.basis_set.n_electron_basis, 
                                                       :self.basis_set.n_electron_basis])
            
            # Form initial density matrix
            self.P_e = np.zeros((self.basis_set.n_electron_basis, self.basis_set.n_electron_basis))
            n_occ = self.n_electrons // 2
            for i in range(n_occ):
                self.P_e += 2.0 * np.outer(e_vecs[:, i], e_vecs[:, i])
        
        # For positrons
        if self.n_positrons > 0:
            # Similar process for positrons
            pass
    
    def build_fock_matrix(self, P, H_core, ERI=None, is_positron=False):
        """
        Build the Fock matrix with optimized algorithms.
        """
        if P is None:
            return H_core.copy()
        
        F = H_core.copy()
        
        # Add two-electron contributions
        if ERI is not None:
            n = P.shape[0]
            
            # Vectorized version when possible
            if isinstance(ERI, np.ndarray):
                # Direct algorithm
                J = np.einsum('pqrs,rs->pq', ERI, P)
                K = np.einsum('prqs,rs->pq', ERI, P)
                F += 2.0 * J - K
            else:
                # On-the-fly calculation for larger systems
                for p in range(n):
                    for q in range(n):
                        val = 0.0
                        for r in range(n):
                            for s in range(n):
                                val += P[r, s] * (2.0 * ERI[p, q, r, s] - ERI[p, r, q, s])
                        F[p, q] += val
        
        return F
    
    def compute_energy(self, P_e, P_p, H_core_e, H_core_p, F_e, F_p):
        """
        Calculate the SCF energy efficiently.
        """
        energy = 0.0
        
        # Electronic contribution
        if P_e is not None and H_core_e is not None and F_e is not None:
            energy += 0.5 * np.sum((H_core_e + F_e) * P_e)
        
        # Positronic contribution 
        if P_p is not None and H_core_p is not None and F_p is not None:
            energy += 0.5 * np.sum((H_core_p + F_p) * P_p)
        
        # Add nuclear repulsion
        energy += self.molecular_data.get_nuclear_repulsion_energy()
        
        return energy
    
    def diis_extrapolation(self, F, error_vector):
        """
        Apply DIIS (Direct Inversion of Iterative Subspace) to accelerate convergence.
        """
        # Add current Fock matrix and error to history
        self.diis_fock_matrices.append(F.copy())
        self.diis_error_vectors.append(error_vector.copy())
        
        # Keep only the last diis_dim matrices
        if len(self.diis_fock_matrices) > self.diis_dim:
            self.diis_fock_matrices.pop(0)
            self.diis_error_vectors.pop(0)
        
        n_diis = len(self.diis_fock_matrices)
        
        # Build B matrix for DIIS
        B = np.zeros((n_diis + 1, n_diis + 1))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        
        for i in range(n_diis):
            for j in range(n_diis):
                B[i, j] = np.einsum('ij,ij->', self.diis_error_vectors[i], self.diis_error_vectors[j])
        
        # Solve DIIS equations
        rhs = np.zeros(n_diis + 1)
        rhs[-1] = -1
        
        try:
            coeffs = np.linalg.solve(B, rhs)
            
            # Form extrapolated Fock matrix
            F_diis = np.zeros_like(F)
            for i in range(n_diis):
                F_diis += coeffs[i] * self.diis_fock_matrices[i]
            
            return F_diis
        except np.linalg.LinAlgError:
            # If DIIS fails, return original matrix
            return F
    
    def solve_scf(self):
        """
        Perform SCF calculation with optimized algorithms and convergence acceleration.
        """
        # Generate initial guess
        self.initial_guess()
        
        # Main SCF loop
        energy_prev = 0.0
        converged = False
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # Build Fock matrices
            F_e = self.build_fock_matrix(self.P_e, self.H_core_e, self.ERI)
            F_p = self.build_fock_matrix(self.P_p, self.H_core_p, None, is_positron=True) if self.n_positrons > 0 else None
            
            # Compute energy
            energy = self.compute_energy(self.P_e, self.P_p, self.H_core_e, self.H_core_p, F_e, F_p)
            
            # Check convergence
            energy_diff = abs(energy - energy_prev)
            energy_prev = energy
            
            if energy_diff < self.convergence_threshold:
                converged = True
                break
            
            # DIIS acceleration for electrons
            if self.use_diis and iteration >= self.diis_start:
                # Calculate error vector
                error_e = (self.S @ self.P_e @ F_e - F_e @ self.P_e @ self.S).flatten()
                
                # Apply DIIS
                F_e = self.diis_extrapolation(F_e, error_e)
            
            # Same for positrons if needed
            
            # Diagonalize Fock matrix with orthogonalization
            X = sqrtm(inv(self.S[:self.basis_set.n_electron_basis, :self.basis_set.n_electron_basis]))
            F_ortho = X.T @ F_e @ X
            e_vals, C_ortho = eigh(F_ortho)
            C_e = X @ C_ortho
            
            # Form new density matrix
            P_e_new = np.zeros_like(self.P_e)
            n_occ = self.n_electrons // 2
            for i in range(n_occ):
                P_e_new += 2.0 * np.outer(C_e[:, i], C_e[:, i])
            
            # Density damping to improve convergence
            damping_factor = 0.5 if iteration < 5 else 0.3
            self.P_e = damping_factor * P_e_new + (1 - damping_factor) * self.P_e
            
            # Similar for positrons
            
            print(f"Iteration {iteration+1}: Energy = {energy:.10f}, ΔE = {energy_diff:.10f}")
        
        end_time = time.time()
        
        print(f"SCF {'converged' if converged else 'not converged'} in {iteration+1} iterations")
        print(f"Final energy: {energy:.10f} Hartree")
        print(f"Calculation time: {end_time - start_time:.2f} seconds")
        
        # Prepare results
        results = {
            'energy': energy,
            'converged': converged,
            'iterations': iteration + 1,
            'P_electron': self.P_e,
            'P_positron': self.P_p,
            'computation_time': end_time - start_time
        }
        
        return results