"""
Antimatter Quantum Chemistry Core Module
========================================

This module integrates all components of the antimatter quantum chemistry 
framework, providing a unified interface for performing calculations on
electron-positron systems.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from datetime import datetime
import os
import sys

class AntimatterQuantumChemistry:
    """
    Integrated antimatter quantum chemistry framework.
    
    This class provides a unified interface for all components of the
    antimatter quantum chemistry framework, allowing seamless execution
    of calculations on electron-positron systems.
    """
    
    def __init__(self, molecule=None, basis_type='positron-aug-cc-pVDZ',
                 include_relativistic=True, include_annihilation=True,
                 scf_max_iterations=100, scf_convergence=1e-6,
                 integral_grid_level=3, visualization_level='standard'):
        """
        Initialize the antimatter quantum chemistry framework.
        
        Parameters:
        -----------
        molecule : object, optional
            Molecule object with atom positions and types
        basis_type : str
            Type of basis set to use
        include_relativistic : bool
            Whether to include relativistic effects
        include_annihilation : bool
            Whether to include electron-positron annihilation
        scf_max_iterations : int
            Maximum number of SCF iterations
        scf_convergence : float
            Convergence threshold for SCF
        integral_grid_level : int
            Level of accuracy for numerical integration
        visualization_level : str
            Level of detail for visualization (minimal, standard, detailed)
        """
        self.molecule = molecule
        self.basis_type = basis_type
        self.include_relativistic = include_relativistic
        self.include_annihilation = include_annihilation
        self.scf_max_iterations = scf_max_iterations
        self.scf_convergence = scf_convergence
        self.integral_grid_level = integral_grid_level
        self.visualization_level = visualization_level
        
        # Components (to be initialized)
        self.basis_set = None
        self.integral_engine = None
        self.operator_engine = None
        self.scf_solver = None
        self.visualizer = None
        
        # Results
        self.integrals = None
        self.scf_results = None
        self.analysis_results = None
        self.computation_time = None
        
        # Initialize if molecule is provided
        if molecule is not None:
            self.initialize_components()
    
    def initialize_components(self):
        """Initialize all components of the framework."""
        print("Initializing Antimatter Quantum Chemistry framework...")
        
        start_time = time.time()
        
        # Import components
        try:
            print("Importing specialized basis module...")
            # Try different import approaches in case one fails
            try:
                # Try relative import first
                from .positron_basis import GaussianPositronBasis
            except ImportError:
                try:
                    # Try absolute import
                    from antimatter_qc.positron_basis import GaussianPositronBasis
                except ImportError:
                    # Try direct import
                    from positron_basis import GaussianPositronBasis
            
            print("Importing integral calculation module...")
            try:
                from .antimatter_integrals import AntimatterIntegrals
            except ImportError:
                try:
                    from antimatter_qc.antimatter_integrals import AntimatterIntegrals
                except ImportError:
                    from antimatter_integrals import AntimatterIntegrals
            
            print("Importing antimatter operators module...")
            try:
                from .antimatter_operators import AntimatterOperators
            except ImportError:
                try:
                    from antimatter_qc.antimatter_operators import AntimatterOperators
                except ImportError:
                    from antimatter_operators import AntimatterOperators
            
            print("Importing SCF module...")
            try:
                from .antimatter_scf import AntimatterSCF
            except ImportError:
                try:
                    from antimatter_qc.antimatter_scf import AntimatterSCF
                except ImportError:
                    from antimatter_scf import AntimatterSCF
            
            print("Importing visualization module...")
            try:
                from .antimatter_visualization import AntimatterVisualizer
            except ImportError:
                try:
                    from antimatter_qc.antimatter_visualization import AntimatterVisualizer
                except ImportError:
                    from antimatter_visualization import AntimatterVisualizer
        except ImportError as e:
            print(f"Error importing modules: {str(e)}")
            print("Make sure all required modules are available.")
            return False
        
        # Initialize basis set
        print(f"Creating specialized {self.basis_type} basis set...")
        self.basis_set = GaussianPositronBasis(self.molecule, basis_type=self.basis_type)
        
        # Initialize integral engine
        print("Setting up integral calculation engine...")
        self.integral_engine = AntimatterIntegrals(
            self.molecule, 
            self.basis_set, 
            grid_level=self.integral_grid_level,
            include_relativistic=self.include_relativistic,
            use_adaptive_grid=True
        )
        
        # Initialize operator engine
        print("Setting up antimatter operator engine...")
        self.operator_engine = AntimatterOperators(
            include_relativistic=self.include_relativistic,
            include_annihilation=self.include_annihilation
        )
        
        # Initialize SCF solver
        print("Setting up SCF solver...")
        hamiltonian_gen = lambda D_e, D_p: self._create_hamiltonian(D_e, D_p)
        
        if self.molecule is not None:
            # Determine number of electrons and positrons from molecule
            # Here we're providing placeholder calculation logic 
            n_electrons = sum(atom.get('Z', atom.get('charge', 0)) for atom in self.molecule.atoms)
            # For simplicity, assume 1 positron for now
            n_positrons = 1
        else:
            # Default values for testing
            n_electrons = 2
            n_positrons = 1
        
        self.scf_solver = AntimatterSCF(
            hamiltonian_gen=hamiltonian_gen,
            basis_set=self.basis_set,
            n_electrons=n_electrons,
            n_positrons=n_positrons,
            max_iterations=self.scf_max_iterations,
            convergence_threshold=self.scf_convergence,
            include_annihilation=self.include_annihilation
        )
        
        # Initialize visualizer
        print("Setting up visualization tools...")
        self.visualizer = AntimatterVisualizer(
            basis_set=self.basis_set,
            molecule=self.molecule,
            show_annihilation=self.include_annihilation
        )
        
        self.computation_time = time.time() - start_time
        print(f"Initialization completed in {self.computation_time:.2f} seconds.")
        
        return True
    
    def _create_hamiltonian(self, D_e, D_p):
        """
        Create Hamiltonian from density matrices.
        
        Parameters:
        -----------
        D_e : np.ndarray
            Electron density matrix
        D_p : np.ndarray
            Positron density matrix
            
        Returns:
        --------
        hamiltonian : dict
            Hamiltonian matrices for electrons and positrons
        """
        # Use operators to construct Hamiltonian
        if self.integrals is None:
            # Calculate integrals if not already done
            self.calculate_integrals()
        
        # Extract necessary integrals
        h_core_e = self.integrals.get('h_core_e')
        h_core_p = self.integrals.get('h_core_p')
        two_e = self.integrals.get('two_electron')
        two_p = self.integrals.get('two_electron_positron')
        two_ep = self.integrals.get('electron_positron')
        
        # Use operator engine to construct Hamiltonian
        nuclei_repulsion = 0.0  # Placeholder, would calculate from molecule
        
        # Create Hamiltonian components
        H_e = self.operator_engine.construct_hamiltonian(
            h_core_e, two_e, h_core_p, two_ep, nuclei_repulsion)
        
        H_p = self.operator_engine.construct_hamiltonian(
            h_core_p, two_p, h_core_e, two_ep, nuclei_repulsion, reverse_spin_mapping=True)
        
        return {'electron': H_e, 'positron': H_p}
    
    def calculate_integrals(self):
        """
        Calculate all necessary integrals for the system.
        
        Returns:
        --------
        integrals : dict
            Dictionary of calculated integrals
        """
        if self.integral_engine is None:
            raise ValueError("Integral engine not initialized. Call initialize_components() first.")
        
        print("Calculating integrals...")
        start_time = time.time()
        
        # Calculate all integrals
        self.integrals = self.integral_engine.calculate_all_integrals()
        
        calculation_time = time.time() - start_time
        print(f"Integral calculation completed in {calculation_time:.2f} seconds.")
        
        return self.integrals
    
    def run_scf(self):
        """
        Run the SCF procedure to solve for the ground state.
        
        Returns:
        --------
        results : dict
            Results of the SCF calculation
        """
        if self.scf_solver is None:
            raise ValueError("SCF solver not initialized. Call initialize_components() first.")
        
        print("Initializing SCF procedure...")
        if self.integrals is None:
            print("Integrals not calculated yet. Calculating now...")
            self.calculate_integrals()
        
        print("\nStarting SCF iterations...")
        start_time = time.time()
        
        # Initialize SCF solver
        self.scf_solver.initialize()
        
        # Run SCF procedure
        self.scf_results = self.scf_solver.run_scf()
        
        calculation_time = time.time() - start_time
        self.scf_results['calculation_time'] = calculation_time
        
        if self.scf_results['converged']:
            print(f"\nSCF converged in {self.scf_results['iterations']} iterations.")
            print(f"Final energy: {self.scf_results['energy']:.10f} Hartree")
        else:
            print(f"\nWARNING: SCF did not converge in {self.scf_max_iterations} iterations.")
            print(f"Last energy: {self.scf_results['energy']:.10f} Hartree")
        
        print(f"SCF calculation completed in {calculation_time:.2f} seconds.")
        
        return self.scf_results
    
    def analyze_results(self, properties=None):
        """
        Analyze SCF results to extract molecular properties.
        
        Parameters:
        -----------
        properties : list, optional
            List of properties to calculate
            
        Returns:
        --------
        analysis : dict
            Dictionary of calculated properties
        """
        if self.scf_results is None:
            raise ValueError("SCF not run yet. Call run_scf() first.")
        
        print("Analyzing calculation results...")
        start_time = time.time()
        
        # Default properties if none specified
        if properties is None:
            properties = [
                'energy_components', 
                'dipole_moment', 
                'charges', 
                'orbital_analysis',
                'annihilation_rate'
            ]
        
        # Initialize results dictionary
        self.analysis_results = {}
        
        # Extract energy components
        if 'energy_components' in properties:
            self.analysis_results['energy_components'] = self.scf_results['energy_components']
        
        # Calculate dipole moment
        if 'dipole_moment' in properties:
            self.analysis_results['dipole_moment'] = self._calculate_dipole_moment()
        
        # Calculate atomic charges
        if 'charges' in properties:
            self.analysis_results['charges'] = self._calculate_charges()
        
        # Analyze orbitals
        if 'orbital_analysis' in properties:
            self.analysis_results['orbital_analysis'] = self._analyze_orbitals()
        
        # Calculate annihilation rate
        if 'annihilation_rate' in properties and self.include_annihilation:
            self.analysis_results['annihilation_rate'] = self._calculate_annihilation_rate()
        
        calculation_time = time.time() - start_time
        print(f"Analysis completed in {calculation_time:.2f} seconds.")
        
        return self.analysis_results
    
    def _calculate_dipole_moment(self):
        """Calculate molecular dipole moment."""
        # Placeholder for actual calculation
        return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'total': 0.0}
    
    def _calculate_charges(self):
        """Calculate atomic charges."""
        # Placeholder for actual calculation
        if self.molecule is None:
            return None
        
        charges = []
        for i, atom in enumerate(self.molecule.atoms):
            charges.append({
                'atom': i,
                'element': atom['element'],
                'charge': 0.0  # Placeholder
            })
        
        return charges
    
    def _analyze_orbitals(self):
        """Analyze molecular orbitals."""
        # Placeholder for actual calculation
        if self.scf_results is None:
            return None
        
        # Extract orbital information
        orbitals_e = {
            'energies': self.scf_results['orbital_energies_e'],
            'coefficients': self.scf_results['mo_coefficients_e'],
            'homo': self.scf_results.get('n_electrons', 0) // 2,
            'lumo': self.scf_results.get('n_electrons', 0) // 2 + 1,
            'homo_lumo_gap': 0.0  # Placeholder
        }
        
        orbitals_p = {
            'energies': self.scf_results['orbital_energies_p'],
            'coefficients': self.scf_results['mo_coefficients_p'],
            'homo': self.scf_results.get('n_positrons', 0) // 2,
            'lumo': self.scf_results.get('n_positrons', 0) // 2 + 1,
            'homo_lumo_gap': 0.0  # Placeholder
        }
        
        # Calculate HOMO-LUMO gaps
        if (orbitals_e['homo'] < len(orbitals_e['energies']) and 
            orbitals_e['lumo'] < len(orbitals_e['energies'])):
            orbitals_e['homo_lumo_gap'] = (
                orbitals_e['energies'][orbitals_e['lumo']] - 
                orbitals_e['energies'][orbitals_e['homo']]
            )
        
        if (orbitals_p['homo'] < len(orbitals_p['energies']) and 
            orbitals_p['lumo'] < len(orbitals_p['energies'])):
            orbitals_p['homo_lumo_gap'] = (
                orbitals_p['energies'][orbitals_p['lumo']] - 
                orbitals_p['energies'][orbitals_p['homo']]
            )
        
        return {
            'electron': orbitals_e,
            'positron': orbitals_p
        }
    
    def _calculate_annihilation_rate(self):
        """Calculate positron-electron annihilation rate."""
        # Placeholder for actual calculation
        if not self.include_annihilation:
            return 0.0
        
        # Simple approximation based on density overlap
        density_e = self.scf_results['density_matrix_e']
        density_p = self.scf_results['density_matrix_p']
        
        # Use operator engine to calculate rate
        # This is a placeholder
        rate = 0.1 * np.sum(density_e * density_p)
        
        return rate
    
    def visualize_results(self, plot_type='density', save_dir=None, show=True):
        """
        Create visualizations of calculation results.
        
        Parameters:
        -----------
        plot_type : str
            Type of plot to generate (density, orbital, spectrum, convergence)
        save_dir : str, optional
            Directory to save plots
        show : bool
            Whether to display the plots
            
        Returns:
        --------
        figures : list
            List of generated matplotlib figures
        """
        if self.visualizer is None:
            raise ValueError("Visualizer not initialized. Call initialize_components() first.")
        
        if self.scf_results is None:
            raise ValueError("SCF not run yet. Call run_scf() first.")
        
        # Get density matrices
        density_e = self.scf_results['density_matrix_e']
        density_p = self.scf_results['density_matrix_p']
        
        figures = []
        
        if plot_type == 'density':
            print("Generating density plots...")
            
            # 2D density plots
            fig = self.visualizer.plot_density_2d(
                density_matrix_e=density_e,
                density_matrix_p=density_p,
                plane='xy',
                resolution=100
            )
            figures.append(fig)
            
            # 3D density plots (if detailed visualization level)
            if self.visualization_level in ['detailed']:
                fig = self.visualizer.plot_density_3d(
                    density_matrix_e=density_e,
                    density_matrix_p=density_p,
                    resolution=30
                )
                figures.append(fig)
        
        elif plot_type == 'orbital':
            print("Generating orbital plots...")
            
            # Get MO coefficients
            mo_coeff_e = self.scf_results['mo_coefficients_e']
            mo_coeff_p = self.scf_results['mo_coefficients_p']
            
            # Find HOMO and LUMO indexes
            n_electrons = self.scf_results.get('n_electrons', 0)
            n_positrons = self.scf_results.get('n_positrons', 0)
            
            homo_e = (n_electrons // 2) - 1
            lumo_e = n_electrons // 2
            homo_p = (n_positrons // 2) - 1
            lumo_p = n_positrons // 2
            
            # Plot HOMOs
            if homo_e >= 0:
                fig = self.visualizer.plot_orbital(
                    mo_coeff_e, homo_e, resolution=50, contour=True)
                fig.suptitle(f"Electron HOMO (Orbital {homo_e})")
                figures.append(fig)
            
            if homo_p >= 0:
                fig = self.visualizer.plot_orbital(
                    mo_coeff_p, homo_p, resolution=50, contour=True)
                fig.suptitle(f"Positron HOMO (Orbital {homo_p})")
                figures.append(fig)
            
            # Plot LUMOs
            if lumo_e < mo_coeff_e.shape[1]:
                fig = self.visualizer.plot_orbital(
                    mo_coeff_e, lumo_e, resolution=50, contour=True)
                fig.suptitle(f"Electron LUMO (Orbital {lumo_e})")
                figures.append(fig)
            
            if lumo_p < mo_coeff_p.shape[1]:
                fig = self.visualizer.plot_orbital(
                    mo_coeff_p, lumo_p, resolution=50, contour=True)
                fig.suptitle(f"Positron LUMO (Orbital {lumo_p})")
                figures.append(fig)
        
        elif plot_type == 'spectrum':
            print("Generating orbital energy spectrum...")
            
            # Get orbital energies
            e_energies = self.scf_results['orbital_energies_e']
            p_energies = self.scf_results['orbital_energies_p']
            
            # Find occupied orbitals
            n_electrons = self.scf_results.get('n_electrons', 0)
            n_positrons = self.scf_results.get('n_positrons', 0)
            
            fig = self.visualizer.plot_spectrum(
                orbital_energies_e=e_energies,
                orbital_energies_p=p_energies,
                occupied_e=n_electrons // 2,
                occupied_p=n_positrons // 2
            )
            figures.append(fig)
        
        elif plot_type == 'convergence':
            print("Generating convergence plots...")
            
            # Get convergence data
            energy_history = self.scf_results['energy_history']
            conv_history = self.scf_results['convergence_history']
            
            fig = self.visualizer.plot_convergence(
                energy_history=energy_history,
                conv_history=conv_history
            )
            figures.append(fig)
        
        else:
            print(f"Unknown plot type: {plot_type}")
            return figures
        
        # Save figures if directory provided
        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            for i, fig in enumerate(figures):
                filename = f"{save_dir}/{plot_type}_{i}_{timestamp}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved plot to {filename}")
        
        # Show figures if requested
        if show:
            plt.show()
        
        return figures
    
    def save_results(self, filename, format='npz'):
        """
        Save calculation results to a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to save
        format : str
            File format ('npz', 'json', 'pickle')
            
        Returns:
        --------
        success : bool
            Whether the save was successful
        """
        if self.scf_results is None:
            raise ValueError("No results to save. Run calculations first.")
        
        print(f"Saving results to {filename}...")
        
        try:
            if format == 'npz':
                # Convert SCF results to arrays for numpy format
                np.savez(
                    filename,
                    energy=self.scf_results['energy'],
                    density_matrix_e=self.scf_results['density_matrix_e'],
                    density_matrix_p=self.scf_results['density_matrix_p'],
                    mo_coefficients_e=self.scf_results['mo_coefficients_e'],
                    mo_coefficients_p=self.scf_results['mo_coefficients_p'],
                    orbital_energies_e=self.scf_results['orbital_energies_e'],
                    orbital_energies_p=self.scf_results['orbital_energies_p'],
                    energy_history=np.array(self.scf_results['energy_history']),
                    converged=np.array([self.scf_results['converged']])
                )
            
            elif format == 'json':
                import json
                
                # Create a JSON-compatible dictionary
                json_data = {
                    'energy': float(self.scf_results['energy']),
                    'converged': bool(self.scf_results['converged']),
                    'iterations': int(self.scf_results['iterations']),
                    'energy_history': [float(e) for e in self.scf_results['energy_history']],
                    'energy_components': {
                        k: float(v) for k, v in self.scf_results['energy_components'].items()
                    }
                }
                
                # Save as JSON
                with open(filename, 'w') as f:
                    json.dump(json_data, f, indent=2)
            
            elif format == 'pickle':
                import pickle
                
                # Save all results
                with open(filename, 'wb') as f:
                    pickle.dump(self.scf_results, f)
            
            else:
                print(f"Unknown format: {format}")
                return False
            
            print(f"Results saved successfully to {filename}")
            return True
        
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False
    
    def load_results(self, filename, format='npz'):
        """
        Load calculation results from a file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to load
        format : str
            File format ('npz', 'json', 'pickle')
            
        Returns:
        --------
        success : bool
            Whether the load was successful
        """
        print(f"Loading results from {filename}...")
        
        try:
            if format == 'npz':
                # Load numpy format
                data = np.load(filename)
                
                # Reconstruct SCF results
                self.scf_results = {
                    'energy': float(data['energy']),
                    'converged': bool(data['converged'][0]),
                    'density_matrix_e': data['density_matrix_e'],
                    'density_matrix_p': data['density_matrix_p'],
                    'mo_coefficients_e': data['mo_coefficients_e'],
                    'mo_coefficients_p': data['mo_coefficients_p'],
                    'orbital_energies_e': data['orbital_energies_e'],
                    'orbital_energies_p': data['orbital_energies_p'],
                    'energy_history': data['energy_history'].tolist()
                }
            
            elif format == 'json':
                import json
                
                # Load JSON format
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct SCF results
                self.scf_results = data
            
            elif format == 'pickle':
                import pickle
                
                # Load pickle format
                with open(filename, 'rb') as f:
                    self.scf_results = pickle.load(f)
            
            else:
                print(f"Unknown format: {format}")
                return False
            
            print(f"Results loaded successfully from {filename}")
            return True
        
        except Exception as e:
            print(f"Error loading results: {str(e)}")
            return False
        
    def get_summary(self):
        """
        Get a text summary of calculation results.
        
        Returns:
        --------
        summary : str
            Text summary of results
        """
        if self.scf_results is None:
            return "No calculations performed yet."
        
        lines = []
        lines.append("=" * 80)
        lines.append("ANTIMATTER QUANTUM CHEMISTRY CALCULATION SUMMARY")
        lines.append("=" * 80)
        
        # System information
        lines.append("\nSYSTEM INFORMATION:")
        if self.molecule is not None:
            atoms = [f"{atom['element']} at {atom['position']}" 
                    for atom in self.molecule.atoms]
            lines.append(f"Molecule: {len(atoms)} atoms")
            for atom in atoms:
                lines.append(f"  {atom}")
        else:
            lines.append("Molecule: Not specified")
        
        lines.append(f"Basis set: {self.basis_type}")
        lines.append(f"Include relativistic effects: {self.include_relativistic}")
        lines.append(f"Include annihilation: {self.include_annihilation}")
        
        # Calculation settings
        lines.append("\nCALCULATION SETTINGS:")
        lines.append(f"SCF max iterations: {self.scf_max_iterations}")
        lines.append(f"SCF convergence threshold: {self.scf_convergence}")
        lines.append(f"Integral grid level: {self.integral_grid_level}")
        
        # SCF results
        lines.append("\nSCF RESULTS:")
        lines.append(f"Converged: {self.scf_results['converged']}")
        lines.append(f"Iterations: {self.scf_results['iterations']}")
        lines.append(f"Final energy: {self.scf_results['energy']:.10f} Hartree")
        
        # Energy components
        if 'energy_components' in self.scf_results:
            lines.append("\nENERGY COMPONENTS (Hartree):")
            components = self.scf_results['energy_components']
            for component, value in components.items():
                lines.append(f"  {component.replace('_', ' ').title()}: {value:.10f}")
        
        # Orbital information
        lines.append("\nORBITAL INFORMATION:")
        n_electrons = self.scf_results.get('n_electrons', 0)
        n_positrons = self.scf_results.get('n_positrons', 0)
        
        e_energies = self.scf_results['orbital_energies_e']
        p_energies = self.scf_results['orbital_energies_p']
        
        lines.append(f"Number of electron orbitals: {len(e_energies)}")
        lines.append(f"Number of positron orbitals: {len(p_energies)}")
        
        # Show a few orbital energies
        lines.append("\nElectron orbital energies (Hartree):")
        for i, energy in enumerate(e_energies[:min(10, len(e_energies))]):
            occ = " (occupied)" if i < n_electrons // 2 else ""
            lines.append(f"  Orbital {i}: {energy:.6f}{occ}")
        
        lines.append("\nPositron orbital energies (Hartree):")
        for i, energy in enumerate(p_energies[:min(10, len(p_energies))]):
            occ = " (occupied)" if i < n_positrons // 2 else ""
            lines.append(f"  Orbital {i}: {energy:.6f}{occ}")
        
        # Computation time
        if 'calculation_time' in self.scf_results:
            lines.append(f"\nSCF calculation time: {self.scf_results['calculation_time']:.2f} seconds")
        
        # Analysis results
        if self.analysis_results is not None:
            lines.append("\nANALYSIS RESULTS:")
            
            # Dipole moment
            if 'dipole_moment' in self.analysis_results:
                dipole = self.analysis_results['dipole_moment']
                lines.append(f"Dipole moment (Debye): {dipole['total']:.4f}")
                lines.append(f"  Components: X={dipole['x']:.4f}, Y={dipole['y']:.4f}, Z={dipole['z']:.4f}")
            
            # Annihilation rate
            if 'annihilation_rate' in self.analysis_results:
                rate = self.analysis_results['annihilation_rate']
                lines.append(f"Positron-electron annihilation rate: {rate:.6e} ns^-1")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)

# Molecule class
class Molecule:
    """Simple molecule container for antimatter calculations."""
    
    def __init__(self, atoms=None):
        """
        Initialize a molecule.
        
        Parameters:
        -----------
        atoms : list, optional
            List of atom dictionaries
        """
        self.atoms = atoms if atoms is not None else []
    
    @classmethod
    def from_xyz(cls, xyz_string):
        """
        Create a molecule from an XYZ format string.
        
        Parameters:
        -----------
        xyz_string : str
            XYZ format string
            
        Returns:
        --------
        molecule : Molecule
            Molecule object
        """
        lines = xyz_string.strip().split('\n')
        n_atoms = int(lines[0])
        comment = lines[1]
        
        atoms = []
        for i in range(2, 2 + n_atoms):
            if i >= len(lines):
                break
                
            parts = lines[i].split()
            if len(parts) >= 4:
                element = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                
                atoms.append({
                    'element': element,
                    'position': np.array([x, y, z])
                })
        
        molecule = cls(atoms)
        molecule.comment = comment
        return molecule
    
    def to_xyz(self):
        """
        Convert the molecule to XYZ format string.
        
        Returns:
        --------
        xyz_string : str
            XYZ format string
        """
        lines = [str(len(self.atoms))]
        lines.append(getattr(self, 'comment', 'Created by AntimatterQuantumChemistry'))
        
        for atom in self.atoms:
            element = atom['element']
            x, y, z = atom['position']
            lines.append(f"{element} {x:.6f} {y:.6f} {z:.6f}")
        
        return '\n'.join(lines)
    
    def add_atom(self, element, position):
        """
        Add an atom to the molecule.
        
        Parameters:
        -----------
        element : str
            Element symbol
        position : array_like
            3D coordinates
        """
        self.atoms.append({
            'element': element,
            'position': np.array(position)
        })
    
    def get_atoms(self):
        """
        Get the list of atoms.
        
        Returns:
        --------
        atoms : list
            List of atom dictionaries
        """
        return self.atoms