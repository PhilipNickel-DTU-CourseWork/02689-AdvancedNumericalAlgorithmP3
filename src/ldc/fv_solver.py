"""Finite volume solver for lid-driven cavity.

This module implements a collocated finite volume solver using
SIMPLE algorithm for pressure-velocity coupling.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Tuple

from .base_solver import LidDrivenCavitySolver
from .datastructures import FVConfig


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

    # Make config class accessible via solver
    Config = FVConfig

    def __init__(self, **kwargs):
        """Initialize FV solver.

        Parameters
        ----------
        **kwargs
            Configuration parameters passed to FVConfig.
            Can also pass config=FVConfig(...) directly.
        """
        super().__init__(**kwargs)

    def solve(self, tolerance: float = 1e-6, max_iter: int = 1000):
        """Run the SIMPLE algorithm until convergence.

        Parameters
        ----------
        tolerance : float
            Convergence tolerance for residual norm.
        max_iter : int
            Maximum number of iterations.
        """
        from fv.core.simple_algorithm import simple_algorithm

        n_cells = self.mesh.cell_volumes.shape[0]
        print(f"Starting FV solver with SIMPLE algorithm (Re={self.config.Re}, "
              f"n_cells={n_cells})")

        # Run SIMPLE algorithm - unpack directly into solver state
        self.fields, self.time_series, self.metadata = simple_algorithm(
            mesh=self.mesh,
            config=self.config,
            rho=self.rho,
            mu=self.mu,
            max_iter=max_iter,
            tol=tolerance,
        )
