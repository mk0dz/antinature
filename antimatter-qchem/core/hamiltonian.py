import numpy as np
from typing import List, Tuple, Optional

class AntimatterHamiltonian:
    """Fully-featured Hamiltonian for antimatter molecular systems."""
    
    def __init__(self, 
                 nuclei: List[Tuple[str, float, np.ndarray]], 
                 n_electrons: int, 
                 n_positrons: int,
                 include_annihilation: bool = True,
                 include_relativistic: bool = True):
        """
        Initialize an antimatter Hamiltonian.
        
        Parameters:
        -----------
        nuclei : List of tuples (element, charge, position)
            Nuclear coordinates and charges
        n_electrons : int
            Number of electrons
        n_positrons : int
            Number of positrons
        include_annihilation : bool
            Whether to include annihilation terms
        include_relativistic : bool
            Whether to include relativistic corrections
        """
        self.nuclei = nuclei
        self.n_electrons = n_electrons
        self.n_positrons = n_positrons
        self.include_annihilation = include_annihilation
        self.include_relativistic = include_relativistic
        
        # Initialize integral storage
        self.overlap = None
        self.kinetic = None
        self.nuclear_attraction = None
        self.electron_repulsion = None
        self.positron_repulsion = None
        self.electron_positron_attraction = None
        self.annihilation = None
        
    def compute_integrals(self, basis_set):
        """
        Compute all necessary integrals using the provided basis set.
        """
        if hasattr(basis_set, 'n_total_basis'):
    # For MixedMatterBasis objects 
            n_basis = basis_set.n_total_basis
        else:
            # For regular BasisSet objects
            n_basis = basis_set.n_basis
        # Initialize integral matrices
        self.overlap = np.zeros((n_basis, n_basis))
        self.kinetic = np.zeros((n_basis, n_basis))
        self.nuclear_attraction = np.zeros((n_basis, n_basis))
        self.electron_repulsion = np.zeros((n_basis, n_basis, n_basis, n_basis))
        
        # For mixed electron-positron systems
        if self.n_positrons > 0:
            self.positron_kinetic = np.zeros((n_basis, n_basis))
            self.positron_nuclear = np.zeros((n_basis, n_basis))
            self.positron_repulsion = np.zeros((n_basis, n_basis, n_basis, n_basis))
            self.electron_positron_attraction = np.zeros((n_basis, n_basis, n_basis, n_basis))
            
            if self.include_annihilation:
                self.annihilation = np.zeros((n_basis, n_basis))
        
        # Compute standard integrals first
        for i in range(n_basis):
            for j in range(n_basis):
                self.overlap[i, j] = basis_set.overlap_integral(i, j)
                self.kinetic[i, j] = basis_set.kinetic_integral(i, j)
                
                # Nuclear attraction with sign change for positrons
                nuc_sum = 0.0
                for atom, charge, pos in self.nuclei:
                    integral = basis_set.nuclear_attraction_integral(i, j, pos)
                    nuc_sum += charge * integral
                
                self.nuclear_attraction[i, j] = nuc_sum
                
                # Compute electron repulsion integrals (ERI)
                for k in range(n_basis):
                    for l in range(n_basis):
                        self.electron_repulsion[i, j, k, l] = basis_set.electron_repulsion_integral(i, j, k, l)
        
        # For positronic systems, modify the integrals
        if self.n_positrons > 0:
            # Positron kinetic energy (mass difference)
            positron_mass_ratio = 1.0  # Actually equal to electron mass
            self.positron_kinetic = self.kinetic * positron_mass_ratio
            
            # Nuclear attraction (sign change)
            self.positron_nuclear = -self.nuclear_attraction
            
            # Positron repulsion integrals (same as electrons)
            self.positron_repulsion = self.electron_repulsion
            
            # Electron-positron attraction (opposite sign of repulsion)
            self.electron_positron_attraction = -self.electron_repulsion
            
            # Compute annihilation integrals if requested
            if self.include_annihilation:
                for i in range(n_basis):
                    for j in range(n_basis):
                        self.annihilation[i, j] = basis_set.annihilation_integral(i, j)
        
        # Add relativistic corrections if requested
        if self.include_relativistic:
            self.add_relativistic_corrections(basis_set)
    
    def add_relativistic_corrections(self, basis_set):
        """
        Add relativistic corrections to the Hamiltonian.
        
        Includes:
        1. Mass-velocity correction
        2. Darwin term
        3. Spin-orbit coupling (for open-shell systems)
        """
        if hasattr(basis_set, 'n_total_basis'):
        # For MixedMatterBasis objects
            n_basis = basis_set.n_total_basis
        else:
            # For regular BasisSet objects
            n_basis = basis_set.n_basis
        c = 137.036  # Speed of light in atomic units
        
        # Mass-velocity correction
        self.mass_velocity = np.zeros((n_basis, n_basis))
        
        # Darwin term
        self.darwin = np.zeros((n_basis, n_basis))
        
        # Compute the corrections
        for i in range(n_basis):
            for j in range(n_basis):
                # Mass-velocity term: -1/(8c²) ∇⁴
                self.mass_velocity[i, j] = basis_set.mass_velocity_integral(i, j) / (8 * c * c)
                
                # Darwin term: (π/2c²) ∑_A Z_A δ(r_A)
                darwin_sum = 0.0
                for atom, charge, pos in self.nuclei:
                    integral = basis_set.darwin_integral(i, j, pos)
                    darwin_sum += charge * integral
                
                self.darwin[i, j] = (np.pi / (2 * c * c)) * darwin_sum
        
        # Apply corrections to the Hamiltonian
        self.relativistic_correction = self.mass_velocity + self.darwin
    
    def build_hamiltonian(self):
        """
        Construct the complete Hamiltonian for the antimatter system.
        
        Returns:
        --------
        dict
            Dictionary containing all Hamiltonian components
        """
        # Ensure integrals have been computed
        if self.overlap is None:
            raise ValueError("Integrals must be computed before building the Hamiltonian")
        
        # Build core Hamiltonian (one-electron terms)
        H_core = self.kinetic + self.nuclear_attraction
        
        # Add relativistic corrections if included
        if self.include_relativistic and hasattr(self, 'relativistic_correction'):
            H_core += self.relativistic_correction
        
        # For positronic systems, construct specialized Hamiltonians
        if self.n_positrons > 0:
            # Separate Hamiltonians for electrons and positrons
            H_core_electron = H_core
            H_core_positron = self.positron_kinetic + self.positron_nuclear
            
            # If relativistic corrections included, add to positron Hamiltonian
            if self.include_relativistic and hasattr(self, 'relativistic_correction'):
                H_core_positron += self.relativistic_correction
            
            # Return both Hamiltonians for mixed systems
            return {
                'overlap': self.overlap,
                'H_core_electron': H_core_electron,
                'H_core_positron': H_core_positron,
                'electron_repulsion': self.electron_repulsion,
                'positron_repulsion': self.positron_repulsion,
                'electron_positron_attraction': self.electron_positron_attraction,
                'annihilation': self.annihilation if self.include_annihilation else None
            }
        else:
            # For pure electronic systems, return a consistent dictionary format
            return {
                'overlap': self.overlap,
                'H_core_electron': H_core,
                'H_core_positron': None,  # No positrons
                'electron_repulsion': self.electron_repulsion,
                'positron_repulsion': None,  # No positrons
                'electron_positron_attraction': None,  # No positrons
                'annihilation': None  # No positrons
            }