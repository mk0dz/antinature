"""
Mock Visualization Module for Testing
===================================

This is a simplified version used for testing the framework structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings

class AntimatterVisualizer:
    """Mock implementation of the AntimatterVisualizer class."""
    
    def __init__(self, basis_set, molecule=None, fig_size=(12, 10), dpi=100,
                 color_scheme='spectral', log_scale=False, show_annihilation=True):
        """Initialize the visualizer."""
        self.basis_set = basis_set
        self.molecule = molecule
        self.fig_size = fig_size
        self.dpi = dpi
        self.color_scheme = color_scheme
        self.log_scale = log_scale
        self.show_annihilation = show_annihilation
        
        # Set color maps
        self.electron_cmap = cm.Blues
        self.positron_cmap = cm.Reds
        self.annihilation_cmap = cm.hot
        self.diff_cmap = cm.seismic
    
    def create_grid(self, resolution=50, padding=3.0, dimension=3):
        """Create a grid for visualization."""
        # Create a simple grid
        x = np.linspace(-5.0, 5.0, resolution)
        y = np.linspace(-5.0, 5.0, resolution)
        
        if dimension == 3:
            z = np.linspace(-5.0, 5.0, resolution)
            X, Y, Z = np.meshgrid(x, y, z)
            points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        else:
            z = np.array([0.0])
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        
        grid = {
            'x': x,
            'y': y,
            'z': z if dimension == 3 else None,
            'X': X,
            'Y': Y,
            'Z': Z if dimension == 3 else None,
            'points': points,
            'resolution': resolution,
            'dimension': dimension
        }
        
        return grid
    
    def calculate_density(self, density_matrix, grid=None, resolution=50):
        """Calculate mock electron or positron density."""
        # Create grid if not provided
        if grid is None:
            grid = self.create_grid(resolution=resolution, dimension=2)
        
        # Create a mock density function (Gaussian)
        if grid['dimension'] == 3:
            shape = grid['X'].shape
        else:
            shape = grid['X'].shape
        
        # Create a mock density centered at the origin
        X, Y = grid['X'], grid['Y']
        density = np.exp(-(X**2 + Y**2) / 10.0)
        
        return density, grid
    
    def calculate_annihilation_density(self, density_matrix_e, density_matrix_p, 
                                       grid=None, resolution=50, method='geometric'):
        """Calculate mock annihilation density."""
        # Get electron and positron densities
        electron_density, grid = self.calculate_density(density_matrix_e, grid, resolution)
        positron_density, _ = self.calculate_density(density_matrix_p, grid, resolution)
        
        # Calculate annihilation density based on method
        if method == 'product':
            return electron_density * positron_density, grid
        elif method == 'geometric':
            return np.sqrt(electron_density * positron_density), grid
        else:
            return 0.5 * (electron_density + positron_density), grid
    
    def plot_density_2d(self, density_matrix_e=None, density_matrix_p=None, 
                       plane='xy', offset=0.0, resolution=100, 
                       show_atoms=True, show_contours=True):
        """Create mock 2D density plots."""
        # Create a figure with a grid of subplots
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        # Determine how many plots to make
        n_plots = sum([density_matrix_e is not None, 
                      density_matrix_p is not None, 
                      density_matrix_e is not None and density_matrix_p is not None,
                      density_matrix_e is not None and density_matrix_p is not None and self.show_annihilation])
        
        if n_plots == 0:
            warnings.warn("No density matrices provided for plotting.")
            return fig
        
        if n_plots <= 2:
            n_rows, n_cols = 1, n_plots
        else:
            n_rows, n_cols = 2, 2
        
        # Create grid
        grid = self.create_grid(resolution=resolution, dimension=2)
        
        # Plot mock densities
        plot_idx = 1
        
        if density_matrix_e is not None:
            ax = fig.add_subplot(n_rows, n_cols, plot_idx)
            electron_density, _ = self.calculate_density(density_matrix_e, grid)
            
            im = ax.pcolormesh(grid['X'], grid['Y'], electron_density, cmap=self.electron_cmap)
            plt.colorbar(im, ax=ax)
            ax.set_title('Electron Density')
            ax.set_xlabel(f'{plane[0]} (bohr)')
            ax.set_ylabel(f'{plane[1]} (bohr)')
            
            plot_idx += 1
        
        if density_matrix_p is not None:
            ax = fig.add_subplot(n_rows, n_cols, plot_idx)
            positron_density, _ = self.calculate_density(density_matrix_p, grid)
            
            im = ax.pcolormesh(grid['X'], grid['Y'], positron_density, cmap=self.positron_cmap)
            plt.colorbar(im, ax=ax)
            ax.set_title('Positron Density')
            ax.set_xlabel(f'{plane[0]} (bohr)')
            ax.set_ylabel(f'{plane[1]} (bohr)')
            
            plot_idx += 1
        
        if density_matrix_e is not None and density_matrix_p is not None:
            # Difference plot
            ax = fig.add_subplot(n_rows, n_cols, plot_idx)
            electron_density, _ = self.calculate_density(density_matrix_e, grid)
            positron_density, _ = self.calculate_density(density_matrix_p, grid)
            
            diff_density = electron_density - positron_density
            abs_max = max(abs(diff_density.min()), abs(diff_density.max()))
            
            im = ax.pcolormesh(grid['X'], grid['Y'], diff_density, 
                              cmap=self.diff_cmap, vmin=-abs_max, vmax=abs_max)
            plt.colorbar(im, ax=ax)
            ax.set_title('Electron-Positron Density Difference')
            ax.set_xlabel(f'{plane[0]} (bohr)')
            ax.set_ylabel(f'{plane[1]} (bohr)')
            
            plot_idx += 1
            
            # Annihilation plot
            if self.show_annihilation:
                ax = fig.add_subplot(n_rows, n_cols, plot_idx)
                ann_density, _ = self.calculate_annihilation_density(
                    density_matrix_e, density_matrix_p, grid)
                
                im = ax.pcolormesh(grid['X'], grid['Y'], ann_density, cmap=self.annihilation_cmap)
                plt.colorbar(im, ax=ax)
                ax.set_title('Positron-Electron Annihilation')
                ax.set_xlabel(f'{plane[0]} (bohr)')
                ax.set_ylabel(f'{plane[1]} (bohr)')
        
        plt.tight_layout()
        return fig
    
    def plot_density_3d(self, density_matrix_e=None, density_matrix_p=None,
                       resolution=30, iso_level=0.01, show_atoms=True):
        """Create mock 3D density plots."""
        # Create a simple figure
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Just plot a sphere as a placeholder
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        if density_matrix_e is not None:
            ax.plot_surface(x, y, z, color='blue', alpha=0.3)
        
        if density_matrix_p is not None:
            ax.plot_surface(x*0.7, y*0.7, z*0.7, color='red', alpha=0.3)
        
        ax.set_title('3D Density Visualization (Mock)')
        ax.set_xlabel('X (bohr)')
        ax.set_ylabel('Y (bohr)')
        ax.set_zlabel('Z (bohr)')
        
        return fig
    
    def plot_orbital(self, mo_coefficients, orbital_idx, grid=None, resolution=50,
                    contour=True, abs_phase=False):
        """Create a mock orbital plot."""
        # Create a grid if not provided
        if grid is None:
            grid = self.create_grid(resolution=resolution, dimension=2)
        
        # Create a mock orbital (simple harmonic oscillator)
        x = grid['X']
        y = grid['Y']
        
        if orbital_idx == 0:
            # Ground state: exp(-r^2/2)
            orbital = np.exp(-(x**2 + y**2) / 10.0)
        elif orbital_idx == 1:
            # First excited state: x*exp(-r^2/2)
            orbital = x * np.exp(-(x**2 + y**2) / 10.0)
        elif orbital_idx == 2:
            # Second excited state: y*exp(-r^2/2)
            orbital = y * np.exp(-(x**2 + y**2) / 10.0)
        else:
            # Higher states: (x^2 - y^2)*exp(-r^2/2)
            orbital = (x**2 - y**2) * np.exp(-(x**2 + y**2) / 10.0)
        
        # Create figure
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        
        if contour:
            ax = fig.add_subplot(111)
            
            # Find max absolute value for symmetric color scale
            max_val = np.max(np.abs(orbital))
            
            if abs_phase:
                im = ax.contourf(x, y, np.abs(orbital), levels=20, cmap=self.electron_cmap)
            else:
                im = ax.contourf(x, y, orbital, levels=20, cmap=self.diff_cmap, 
                               vmin=-max_val, vmax=max_val)
            
            ax.contour(x, y, orbital, colors='k', alpha=0.3)
            plt.colorbar(im, ax=ax)
            
            ax.set_title(f'Molecular Orbital {orbital_idx}')
            ax.set_xlabel('X (bohr)')
            ax.set_ylabel('Y (bohr)')
        else:
            ax = fig.add_subplot(111, projection='3d')
            
            if abs_phase:
                surf = ax.plot_surface(x, y, np.abs(orbital), cmap=self.electron_cmap)
            else:
                max_val = np.max(np.abs(orbital))
                surf = ax.plot_surface(x, y, orbital, cmap=self.diff_cmap, 
                                     vmin=-max_val, vmax=max_val)
            
            plt.colorbar(surf, ax=ax, shrink=0.7)
            
            ax.set_title(f'Molecular Orbital {orbital_idx}')
            ax.set_xlabel('X (bohr)')
            ax.set_ylabel('Y (bohr)')
            ax.set_zlabel('Orbital Value')
        
        plt.tight_layout()
        return fig
    
    def plot_spectrum(self, orbital_energies_e=None, orbital_energies_p=None,
                     occupied_e=None, occupied_p=None, overlay=True):
        """Create a mock orbital energy spectrum plot."""
        fig = plt.figure(figsize=(10, 6), dpi=self.dpi)
        
        if orbital_energies_e is None and orbital_energies_p is None:
            warnings.warn("No orbital energies provided for spectrum plot.")
            return fig
        
        # Plot on a single axis if overlay is True and both sets are present
        if overlay and orbital_energies_e is not None and orbital_energies_p is not None:
            ax = fig.add_subplot(111)
            
            # Plot electron orbitals
            n_e = len(orbital_energies_e)
            x_e = np.arange(n_e)
            
            if occupied_e is None:
                occupied_e = n_e // 2
            
            ax.scatter(x_e[:occupied_e], orbital_energies_e[:occupied_e], 
                      color='blue', marker='o', label='Electron (Occupied)')
            ax.scatter(x_e[occupied_e:], orbital_energies_e[occupied_e:], 
                      color='lightblue', marker='o', label='Electron (Virtual)')
            
            # Plot positron orbitals with an offset
            n_p = len(orbital_energies_p)
            x_p = np.arange(n_p) + 0.3
            
            if occupied_p is None:
                occupied_p = n_p // 2
            
            ax.scatter(x_p[:occupied_p], orbital_energies_p[:occupied_p], 
                      color='red', marker='s', label='Positron (Occupied)')
            ax.scatter(x_p[occupied_p:], orbital_energies_p[occupied_p:], 
                      color='pink', marker='s', label='Positron (Virtual)')
            
            ax.set_title('Orbital Energy Spectrum')
            ax.set_xlabel('Orbital Index')
            ax.set_ylabel('Energy (Hartree)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        else:
            # Plot separately
            if orbital_energies_e is not None:
                ax1 = fig.add_subplot(121 if orbital_energies_p is not None else 111)
                
                n_e = len(orbital_energies_e)
                x_e = np.arange(n_e)
                
                if occupied_e is None:
                    occupied_e = n_e // 2
                
                ax1.scatter(x_e[:occupied_e], orbital_energies_e[:occupied_e], 
                           color='blue', marker='o', label='Occupied')
                ax1.scatter(x_e[occupied_e:], orbital_energies_e[occupied_e:], 
                           color='lightblue', marker='o', label='Virtual')
                
                ax1.set_title('Electron Orbital Energies')
                ax1.set_xlabel('Orbital Index')
                ax1.set_ylabel('Energy (Hartree)')
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.7)
            
            if orbital_energies_p is not None:
                ax2 = fig.add_subplot(122 if orbital_energies_e is not None else 111)
                
                n_p = len(orbital_energies_p)
                x_p = np.arange(n_p)
                
                if occupied_p is None:
                    occupied_p = n_p // 2
                
                ax2.scatter(x_p[:occupied_p], orbital_energies_p[:occupied_p], 
                           color='red', marker='s', label='Occupied')
                ax2.scatter(x_p[occupied_p:], orbital_energies_p[occupied_p:], 
                           color='pink', marker='s', label='Virtual')
                
                ax2.set_title('Positron Orbital Energies')
                ax2.set_xlabel('Orbital Index')
                ax2.set_ylabel('Energy (Hartree)')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_convergence(self, energy_history, conv_history=None):
        """Create a mock convergence plot."""
        fig = plt.figure(figsize=(10, 6), dpi=self.dpi)
        
        # Plot energy convergence
        if conv_history is not None:
            ax1 = fig.add_subplot(211)
        else:
            ax1 = fig.add_subplot(111)
        
        iterations = np.arange(len(energy_history))
        ax1.plot(iterations, energy_history, 'o-', color='blue')
        
        ax1.set_title('SCF Energy Convergence')
        ax1.set_xlabel('' if conv_history is not None else 'Iteration')
        ax1.set_ylabel('Energy (Hartree)')
        
        ax1.axhline(y=energy_history[-1], color='r', linestyle='--', alpha=0.5)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot convergence metric if provided
        if conv_history is not None:
            ax2 = fig.add_subplot(212)
            
            ax2.semilogy(np.arange(1, len(conv_history)+1), conv_history, 'o-', color='green')
            
            ax2.set_title('Convergence Metric')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Error')
            
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig