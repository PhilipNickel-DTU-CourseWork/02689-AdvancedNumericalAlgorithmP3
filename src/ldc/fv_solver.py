"""Finite volume solver for lid-driven cavity.

This module implements a collocated finite volume solver using
SIMPLE algorithm for pressure-velocity coupling.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Tuple

from .base_solver import LidDrivenCavitySolver
from .datastructures import FVConfig, Results


class FVSolver(LidDrivenCavitySolver):
    """Finite volume solver for lid-driven cavity problem.

    This solver uses a collocated grid arrangement with Rhie-Chow interpolation
    for pressure-velocity coupling using the SIMPLE algorithm.

    Parameters
    ----------
    config : FVConfig
        Configuration with physics (Re, lid velocity, domain size) and
        FV-specific parameters (nx, ny, convection scheme, etc.).
    """

    def __init__(self, config: FVConfig):
        """Initialize FV solver.

        Parameters
        ----------
        config : FVConfig
            Finite volume configuration with physics and numerics.
        """
        # Initialize base solver (will call _setup_solver_specifics)
        super().__init__(config)

    def _setup_solver_specifics(self):
        """Create FV mesh structure from base class grid.

        This method is called by base class after grid creation.
        Creates MeshData2D directly in memory (no files).
        """
        from meshing.structured_inmemory import create_structured_mesh_2d

        # Create MeshData2D object (numba jitclass)
        self.mesh = create_structured_mesh_2d(
            nx=self.nx,
            ny=self.ny,
            Lx=self.config.Lx,
            Ly=self.config.Ly,
            lid_velocity=self.config.lid_velocity
        )

        # Initialize solution fields
        n_cells = self.nx * self.ny
        n_faces = self.mesh.face_areas.shape[0]
        self.U = np.zeros((n_cells, 2))
        self.p = np.zeros(n_cells)
        self.mdot = np.zeros(n_faces)

    def solve(self, tolerance: float = 1e-6, max_iter: int = 1000) -> Results:
        """Run the SIMPLE algorithm until convergence.

        Parameters
        ----------
        tolerance : float
            Convergence tolerance for residual norm.
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        Results
            Solution data including velocity, pressure, and convergence info.
        """
        from fv.core.simple_algorithm import simple_algorithm

        n_cells = self.mesh.cell_volumes.shape[0]
        print(f"Starting FV solver with SIMPLE algorithm (Re={self.config.Re}, "
              f"n_cells={n_cells})")

        # Run SIMPLE algorithm (returns dictionary)
        result = simple_algorithm(
            mesh=self.mesh,
            alpha_uv=self.config.alpha_uv,
            alpha_p=self.config.alpha_p,
            rho=self.rho,
            mu=self.mu,
            max_iter=max_iter,
            tol=tolerance,
            convection_scheme=self.config.convection_scheme,
            limiter=self.config.limiter,
        )

        # Extract data from result dictionary
        fields = result['fields']
        time_series = result['time_series']
        metadata = result['metadata']

        # Store solution
        self.p = fields['p']
        self.U = np.column_stack([fields['u'], fields['v']])
        self.mdot = result['mdot']
        self.converged = metadata['converged']
        self.iterations = metadata['iterations']
        self.residual_history = time_series['residual']

        # Add coordinates to fields
        fields['x'] = self.mesh.cell_centers[:, 0]
        fields['y'] = self.mesh.cell_centers[:, 1]
        fields['grid_points'] = self.mesh.cell_centers

        # Add solver-specific metadata
        metadata.update({
            'convection_scheme': self.config.convection_scheme,
            'limiter': self.config.limiter,
            'alpha_uv': self.config.alpha_uv,
            'alpha_p': self.config.alpha_p
        })

        return self._build_results(fields, time_series, metadata)
