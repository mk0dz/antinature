"""
Visualization functions for antinature quantum chemistry.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def visualize_annihilation_density(
    density_data: Dict, slice_dim: str = 'z', slice_idx: int = None
) -> plt.Figure:
    """
    Visualize electron-positron annihilation density.

    Parameters:
    -----------
    density_data : Dict
        Dictionary containing grid data and density values with keys:
        - 'x', 'y', 'z': numpy arrays of grid coordinates
        - 'density': 3D numpy array of density values
    slice_dim : str
        Dimension to slice ('x', 'y', 'z', or '3d' for 3D visualization)
    slice_idx : int, optional
        Index of slice to visualize. If None, uses the middle slice.

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the visualization
    """
    if density_data is None or 'density' not in density_data:
        print("No valid density data provided")
        return None

    x = density_data['x']
    y = density_data['y']
    z = density_data['z']
    density = density_data['density']

    fig = plt.figure(figsize=(10, 8))

    if slice_dim == '3d':
        # 3D visualization with isosurfaces
        ax = fig.add_subplot(111, projection='3d')

        # Create a meshgrid for 3D plotting
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Find a suitable isosurface level (e.g., 10% of max density)
        max_density = np.max(density)
        if max_density > 0:
            level = max_density * 0.1
        else:
            level = 0.001

        # Try to create an isosurface
        try:
            from skimage import measure

            verts, faces, _, _ = measure.marching_cubes(density, level)

            # Scale the vertices to match the actual coordinates
            verts[:, 0] = x[0] + (x[-1] - x[0]) * verts[:, 0] / (len(x) - 1)
            verts[:, 1] = y[0] + (y[-1] - y[0]) * verts[:, 1] / (len(y) - 1)
            verts[:, 2] = z[0] + (z[-1] - z[0]) * verts[:, 2] / (len(z) - 1)

            # Plot the isosurface
            mesh = ax.plot_trisurf(
                verts[:, 0],
                verts[:, 1],
                faces,
                verts[:, 2],
                cmap=cm.plasma,
                lw=0,
                alpha=0.7,
            )

            plt.colorbar(mesh, ax=ax, shrink=0.5, aspect=5)

        except (ImportError, ValueError) as e:
            # Fall back to volumetric rendering if marching cubes fails
            print(f"Could not create isosurface: {str(e)}")
            print("Falling back to volumetric rendering")

            # Downsample for performance
            skip = max(1, len(x) // 20)

            # Plot only points with significant density
            mask = density > max_density * 0.05
            points = np.argwhere(mask)

            if len(points) > 0:
                # Limit to 1000 points for performance
                if len(points) > 1000:
                    idx = np.random.choice(len(points), 1000, replace=False)
                    points = points[idx]

                # Scale color by density
                colors = density[mask][idx] if len(points) > 1000 else density[mask]
                normalized_colors = colors / max_density

                # Convert indices to actual coordinates
                x_coords = x[points[:, 0]]
                y_coords = y[points[:, 1]]
                z_coords = z[points[:, 2]]

                scatter = ax.scatter(
                    x_coords,
                    y_coords,
                    z_coords,
                    c=colors,
                    alpha=0.7,
                    cmap=cm.plasma,
                    s=50 * normalized_colors + 5,
                )

                plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
            else:
                ax.text(
                    0,
                    0,
                    0,
                    "No significant density to display",
                    ha='center',
                    va='center',
                    color='red',
                )

        ax.set_xlabel('X (Bohr)')
        ax.set_ylabel('Y (Bohr)')
        ax.set_zlabel('Z (Bohr)')
        ax.set_title('Electron-Positron Annihilation Density')

    else:
        # 2D slice visualization
        if slice_dim not in ['x', 'y', 'z']:
            print(f"Invalid slice_dim: {slice_dim}. Using 'z' instead.")
            slice_dim = 'z'

        # Set up default slice if not specified
        if slice_idx is None:
            if slice_dim == 'x':
                slice_idx = len(x) // 2
            elif slice_dim == 'y':
                slice_idx = len(y) // 2
            else:  # z
                slice_idx = len(z) // 2

        # Extract the slice
        if slice_dim == 'x':
            slice_data = density[slice_idx, :, :]
            extent = [y[0], y[-1], z[0], z[-1]]
            xlabel, ylabel = 'Y (Bohr)', 'Z (Bohr)'
            title = f'Annihilation Density (X = {x[slice_idx]:.2f} Bohr)'
        elif slice_dim == 'y':
            slice_data = density[:, slice_idx, :]
            extent = [x[0], x[-1], z[0], z[-1]]
            xlabel, ylabel = 'X (Bohr)', 'Z (Bohr)'
            title = f'Annihilation Density (Y = {y[slice_idx]:.2f} Bohr)'
        else:  # z
            slice_data = density[:, :, slice_idx]
            extent = [x[0], x[-1], y[0], y[-1]]
            xlabel, ylabel = 'X (Bohr)', 'Y (Bohr)'
            title = f'Annihilation Density (Z = {z[slice_idx]:.2f} Bohr)'

        # Create the plot
        ax = fig.add_subplot(111)
        im = ax.imshow(
            slice_data.T,
            origin='lower',
            extent=extent,
            cmap=cm.viridis,
            interpolation='bilinear',
        )

        plt.colorbar(im, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    return fig


def plot_wavefunction(
    wavefunction: Dict,
    particle_type: str = 'electron',
    orbitals: List[int] = [0],
    grid_dims: Tuple[int, int, int] = (50, 50, 50),
    limits: Tuple[float, float] = (-5.0, 5.0),
) -> plt.Figure:
    """
    Plot wavefunction orbitals in 3D.

    Parameters:
    -----------
    wavefunction : Dict
        Wavefunction data containing MO coefficients, etc.
    particle_type : str
        Type of particle ('electron' or 'positron')
    orbitals : List[int]
        List of orbital indices to plot
    grid_dims : Tuple[int, int, int]
        Dimensions of the visualization grid
    limits : Tuple[float, float]
        Spatial limits for visualization

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the visualization
    """
    if wavefunction is None:
        print("No wavefunction data provided")
        return None

    # Extract MO coefficients
    if particle_type == 'electron':
        mo_coeff = wavefunction.get('C_electron')
    elif particle_type == 'positron':
        mo_coeff = wavefunction.get('C_positron')
    else:
        print(f"Invalid particle type: {particle_type}")
        return None

    if mo_coeff is None:
        print(f"No MO coefficients found for {particle_type}")
        return None

    # Check if orbitals exist
    max_orbital = mo_coeff.shape[1] - 1 if len(mo_coeff.shape) > 1 else 0
    valid_orbitals = [i for i in orbitals if i <= max_orbital]

    if not valid_orbitals:
        print(f"No valid orbitals to plot. Max orbital is {max_orbital}")
        return None

    # Create a simple visualization of the orbital shape
    nx, ny, nz = grid_dims
    xmin, xmax = limits

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(xmin, xmax, ny)
    z = np.linspace(xmin, xmax, nz)

    # For simplicity, just show the contour of each orbital at z=0
    X, Y = np.meshgrid(x, y, indexing='ij')
    z_slice = nz // 2

    # Create a figure with subplots for each orbital
    n_orbitals = len(valid_orbitals)
    fig = plt.figure(figsize=(6 * min(n_orbitals, 3), 5 * ((n_orbitals + 2) // 3)))

    for i, orbital_idx in enumerate(valid_orbitals):
        # Get orbital coefficient
        orbital_coeff = (
            mo_coeff[:, orbital_idx] if len(mo_coeff.shape) > 1 else mo_coeff
        )

        # Create a simple representation of the orbital (this is just a placeholder)
        # In a real implementation, this would involve basis function evaluation
        orbital = np.zeros((nx, ny))

        # Simple placeholder: create a Gaussian centered at the origin
        for ix, xi in enumerate(x):
            for iy, yi in enumerate(y):
                r2 = xi**2 + yi**2
                # Use the orbital coefficient to modulate the size
                orbital[ix, iy] = (
                    np.sum(orbital_coeff) * np.exp(-r2) + 0.1 * np.random.randn()
                )

        # Plot the orbital
        ax = fig.add_subplot(((n_orbitals + 2) // 3), min(n_orbitals, 3), i + 1)

        # Plot contour
        contour = ax.contourf(X, Y, orbital, cmap=cm.RdBu_r)
        plt.colorbar(contour, ax=ax)

        ax.set_xlabel('X (Bohr)')
        ax.set_ylabel('Y (Bohr)')
        ax.set_title(f'{particle_type.capitalize()} Orbital {orbital_idx}')

    plt.tight_layout()
    return fig


class AntinatureVisualizer:
    """
    Visualization tools for antinature systems and calculation results.
    """

    def __init__(self, style: str = 'default'):
        """
        Initialize visualizer with specified style.

        Parameters:
        -----------
        style : str
            Matplotlib style to use ('default', 'dark_background', etc.)
        """
        self.style = style
        plt.style.use(style)

    def plot_positronium_density(self, molecule, basis, scf_result, resolution=30):
        """
        Plot positronium density in 3D.

        Parameters:
        -----------
        molecule : MolecularData
            Positronium system
        basis : MixedMatterBasis
            Basis set used for calculation
        scf_result : Dict
            Result from SCF calculation
        resolution : int
            Grid resolution

        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        # Create a 3D grid
        x = np.linspace(-5, 5, resolution)
        y = np.linspace(-5, 5, resolution)
        z = np.linspace(-5, 5, resolution)
        X, Y, Z = np.meshgrid(x, y, z)

        # Calculate electron and positron densities on the grid
        electron_density = self._calculate_density(
            X, Y, Z, 'electron', basis, scf_result
        )
        positron_density = self._calculate_density(
            X, Y, Z, 'positron', basis, scf_result
        )

        # Create plots
        fig = plt.figure(figsize=(12, 5))

        # Electron density
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.contour3D(
            X[:, :, resolution // 2],
            Y[:, :, resolution // 2],
            electron_density[:, :, resolution // 2],
            50,
            cmap='Blues',
        )
        ax1.set_title('Electron Density')

        # Positron density
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.contour3D(
            X[:, :, resolution // 2],
            Y[:, :, resolution // 2],
            positron_density[:, :, resolution // 2],
            50,
            cmap='Reds',
        )
        ax2.set_title('Positron Density')

        plt.tight_layout()
        return fig

    def plot_positron_density(self, molecule, basis, scf_result, resolution=30):
        """
        Plot positron density for anti-hydrogen or similar.

        Parameters:
        -----------
        molecule : MolecularData
            Anti-hydrogen or similar system
        basis : MixedMatterBasis
            Basis set used for calculation
        scf_result : Dict
            Result from SCF calculation
        resolution : int
            Grid resolution

        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        # Create a 3D grid
        x = np.linspace(-5, 5, resolution)
        y = np.linspace(-5, 5, resolution)
        z = np.linspace(-5, 5, resolution)
        X, Y, Z = np.meshgrid(x, y, z)

        # Calculate positron density
        positron_density = self._calculate_density(
            X, Y, Z, 'positron', basis, scf_result
        )

        # Create plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot a slice of the 3D density
        ax.contour3D(
            X[:, :, resolution // 2],
            Y[:, :, resolution // 2],
            positron_density[:, :, resolution // 2],
            50,
            cmap='Reds',
        )

        # Plot density isosurfaces
        # We need a specific value for the isosurface
        iso_value = np.max(positron_density) * 0.3  # 30% of maximum density
        ax.contour3D(X, Y, Z, positron_density, [iso_value], cmap='Reds', alpha=0.6)

        # Add nuclei position if known
        for atom, position in molecule.atoms:
            ax.scatter(position[0], position[1], position[2], color='black', s=50)

        ax.set_title(f'Positron Density - {molecule.name}')
        ax.set_xlabel('X (bohr)')
        ax.set_ylabel('Y (bohr)')
        ax.set_zlabel('Z (bohr)')

        plt.tight_layout()
        return fig

    def _calculate_density(self, X, Y, Z, particle_type, basis, scf_result):
        """
        Calculate electron or positron density on a grid.

        Parameters:
        -----------
        X, Y, Z : np.ndarray
            Grid coordinates
        particle_type : str
            'electron' or 'positron'
        basis : MixedMatterBasis
            Basis set
        scf_result : Dict
            SCF calculation result

        Returns:
        --------
        np.ndarray
            Density values on the grid
        """
        # Placeholder implementation - in a real implementation, this would
        # use the wavefunction from scf_result and basis functions

        # Simple Gaussian density for demonstration
        density = np.zeros_like(X)

        # Create a density centered at the origin with some spread
        density = np.exp(-(X**2 + Y**2 + Z**2) / 2)

        # Normalize
        density /= np.sum(density)

        return density

    def plot_annihilation_probability(self, antihydrogen_results):
        """
        Plot annihilation probability for an anti-hydrogen system.

        Parameters:
        -----------
        antihydrogen_results : Dict
            Results from anti-hydrogen calculation

        Returns:
        --------
        Figure
            Matplotlib figure
        """
        # Placeholder for actual implementation
        fig, ax = plt.subplots(figsize=(8, 6))

        # Sample data for demonstration
        r = np.linspace(0, 5, 100)
        prob = np.exp(-r)

        ax.plot(r, prob, 'r-', linewidth=2)
        ax.set_xlabel('Distance from nucleus (bohr)')
        ax.set_ylabel('Annihilation probability')
        ax.set_title('Positron-Antiproton Annihilation Probability')
        ax.grid(True, alpha=0.3)

        return fig
    
    def plot_energy_convergence(self, convergence_data, save_path=None, show_theoretical=True):
        """
        Plot energy convergence from SCF or optimization procedure.
        
        Parameters:
        -----------
        convergence_data : Dict or List
            Convergence data containing energies and iterations
        save_path : str, optional
            Path to save the figure
        show_theoretical : bool
            Whether to show theoretical reference lines
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Handle different data formats
        if isinstance(convergence_data, dict):
            if 'energies' in convergence_data:
                energies = convergence_data['energies']
                iterations = convergence_data.get('iterations', range(len(energies)))
            else:
                # Assume keys are iteration numbers and values are energies
                iterations = list(convergence_data.keys())
                energies = list(convergence_data.values())
        elif isinstance(convergence_data, list):
            energies = convergence_data
            iterations = range(len(energies))
        else:
            # Generate sample data for testing
            iterations = range(50)
            energies = [-0.25 + 0.1 * np.exp(-0.1 * i) + 0.01 * np.random.randn() for i in iterations]
        
        # Plot convergence
        ax.plot(iterations, energies, 'o-', linewidth=2, markersize=4, 
                color='blue', label='Energy')
        
        # Add theoretical lines if requested
        if show_theoretical:
            if min(energies) < -0.2 and max(energies) > -0.3:
                # Likely positronium
                ax.axhline(y=-0.25, color='r', linestyle='--', 
                          label='Positronium theoretical (-0.25 Ha)')
            
        # Formatting
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Energy (Hartree)')
        ax.set_title('Energy Convergence')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add final energy annotation
        if energies:
            final_energy = energies[-1]
            ax.annotate(f'Final: {final_energy:.6f}', 
                       xy=(iterations[-1], final_energy), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_density(self, density_data, slice_type='2d', save_path=None):
        """
        Plot electron or positron density.
        
        Parameters:
        -----------
        density_data : Dict or np.ndarray
            Density data to plot
        slice_type : str
            Type of plot ('2d', '3d', 'contour')
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Handle different data types
        if isinstance(density_data, dict):
            if 'density' in density_data:
                density = density_data['density']
                x = density_data.get('x', np.linspace(-5, 5, density.shape[0]))
                y = density_data.get('y', np.linspace(-5, 5, density.shape[1]))
            else:
                # Create sample density for testing
                x = np.linspace(-5, 5, 50)
                y = np.linspace(-5, 5, 50)
                X, Y = np.meshgrid(x, y)
                density = np.exp(-(X**2 + Y**2)/2)
        elif isinstance(density_data, np.ndarray):
            density = density_data
            x = np.linspace(-5, 5, density.shape[0])
            y = np.linspace(-5, 5, density.shape[1])
        else:
            # Create sample density for testing
            x = np.linspace(-5, 5, 50)
            y = np.linspace(-5, 5, 50)
            X, Y = np.meshgrid(x, y)
            density = np.exp(-(X**2 + Y**2)/2)
        
        # Create meshgrid for plotting
        X, Y = np.meshgrid(x, y)
        
        if slice_type == '2d':
            # 2D density plot
            if len(density.shape) == 3:
                # Take middle slice if 3D
                density_2d = density[:, :, density.shape[2]//2]
            else:
                density_2d = density
                
            im = ax.imshow(density_2d, extent=[x.min(), x.max(), y.min(), y.max()],
                          origin='lower', cmap='viridis', aspect='equal')
            plt.colorbar(im, ax=ax, label='Density')
            
        elif slice_type == 'contour':
            # Contour plot
            if len(density.shape) == 3:
                density_2d = density[:, :, density.shape[2]//2]
            else:
                density_2d = density
                
            contour = ax.contourf(X, Y, density_2d, levels=20, cmap='viridis')
            ax.contour(X, Y, density_2d, levels=20, colors='black', alpha=0.5, linewidths=0.5)
            plt.colorbar(contour, ax=ax, label='Density')
            
        elif slice_type == '3d':
            # 3D surface plot
            ax.remove()
            ax = fig.add_subplot(111, projection='3d')
            
            if len(density.shape) == 3:
                density_2d = density[:, :, density.shape[2]//2]
            else:
                density_2d = density
                
            surf = ax.plot_surface(X, Y, density_2d, cmap='viridis', alpha=0.8)
            plt.colorbar(surf, ax=ax, label='Density', shrink=0.5)
            ax.set_zlabel('Density')
        
        ax.set_xlabel('X (Bohr)')
        ax.set_ylabel('Y (Bohr)')
        ax.set_title(f'Density Plot ({slice_type})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
