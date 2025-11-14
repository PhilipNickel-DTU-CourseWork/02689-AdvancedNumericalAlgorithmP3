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

        # Store solution in solver state
        fields = result['fields']
        self.p = fields['p']
        self.U = np.column_stack([fields['u'], fields['v']])
        self.mdot = result['mdot']
        self.converged = result['metadata']['converged']
        self.iterations = result['metadata']['iterations']
        self.residual_history = result['time_series']['residual']

        # Return results directly (simple_algorithm already includes everything)
        return self._build_results(
            fields=result['fields'],
            time_series=result['time_series'],
            solver_metadata=result['metadata']
        )
