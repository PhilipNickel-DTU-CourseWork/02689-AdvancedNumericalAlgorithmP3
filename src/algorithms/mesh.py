"""
Mesh generation and handling for numerical methods.
"""

import numpy as np


class Mesh2D:
    """
    2D structured mesh for finite difference and finite volume methods.
    
    Parameters
    ----------
    nx : int
        Number of grid points in x-direction
    ny : int
        Number of grid points in y-direction
    xmin : float, optional
        Minimum x-coordinate (default: 0.0)
    xmax : float, optional
        Maximum x-coordinate (default: 1.0)
    ymin : float, optional
        Minimum y-coordinate (default: 0.0)
    ymax : float, optional
        Maximum y-coordinate (default: 1.0)
    """
    
    def __init__(self, nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
        self.nx = nx
        self.ny = ny
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        
        # Grid spacing
        self.dx = (xmax - xmin) / (nx - 1)
        self.dy = (ymax - ymin) / (ny - 1)
        
        # Generate mesh grids
        self.x = np.linspace(xmin, xmax, nx)
        self.y = np.linspace(ymin, ymax, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
    
    def get_interior_points(self):
        """
        Get indices of interior points (excluding boundaries).
        
        Returns
        -------
        tuple
            Tuple of (i_indices, j_indices) for interior points
        """
        i = np.arange(1, self.ny - 1)
        j = np.arange(1, self.nx - 1)
        return i, j
    
    def get_boundary_mask(self):
        """
        Get boolean mask for boundary points.
        
        Returns
        -------
        ndarray
            Boolean array where True indicates boundary points
        """
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        mask[0, :] = True   # Bottom boundary
        mask[-1, :] = True  # Top boundary
        mask[:, 0] = True   # Left boundary
        mask[:, -1] = True  # Right boundary
        return mask
