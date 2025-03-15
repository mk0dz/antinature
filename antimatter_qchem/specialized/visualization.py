"""
Visualization functions for antimatter quantum chemistry.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union

def visualize_annihilation_density(density_data: Dict, slice_dim: str = 'z', slice_idx: int = None) -> plt.Figure:
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
                verts[:, 0], verts[:, 1], faces, verts[:, 2],
                cmap=cm.plasma, lw=0, alpha=0.7
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
                    x_coords, y_coords, z_coords,
                    c=colors, alpha=0.7, cmap=cm.plasma,
                    s=50 * normalized_colors + 5
                )
                
                plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
            else:
                ax.text(0, 0, 0, "No significant density to display", 
                       ha='center', va='center', color='red')
        
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
            interpolation='bilinear'
        )
        
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    
    return fig

def plot_wavefunction(wavefunction: Dict, particle_type: str = 'electron', orbitals: List[int] = [0], 
                     grid_dims: Tuple[int, int, int] = (50, 50, 50), limits: Tuple[float, float] = (-5.0, 5.0)) -> plt.Figure:
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
        orbital_coeff = mo_coeff[:, orbital_idx] if len(mo_coeff.shape) > 1 else mo_coeff
        
        # Create a simple representation of the orbital (this is just a placeholder)
        # In a real implementation, this would involve basis function evaluation
        orbital = np.zeros((nx, ny))
        
        # Simple placeholder: create a Gaussian centered at the origin
        for ix, xi in enumerate(x):
            for iy, yi in enumerate(y):
                r2 = xi**2 + yi**2
                # Use the orbital coefficient to modulate the size
                orbital[ix, iy] = np.sum(orbital_coeff) * np.exp(-r2) + 0.1 * np.random.randn()
        
        # Plot the orbital
        ax = fig.add_subplot(((n_orbitals + 2) // 3), min(n_orbitals, 3), i+1)
        
        # Plot contour
        contour = ax.contourf(X, Y, orbital, cmap=cm.RdBu_r)
        plt.colorbar(contour, ax=ax)
        
        ax.set_xlabel('X (Bohr)')
        ax.set_ylabel('Y (Bohr)')
        ax.set_title(f'{particle_type.capitalize()} Orbital {orbital_idx}')
    
    plt.tight_layout()
    return fig 