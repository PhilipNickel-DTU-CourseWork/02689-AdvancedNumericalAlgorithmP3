"""Finite volume solver for lid-driven cavity.

This module implements a collocated finite volume solver using
SIMPLE/PISO algorithms for pressure-velocity coupling.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Tuple

from .base_solver import LidDrivenCavitySolver
from .datastructures import SolverConfig, FVConfig, Results


class FVSolver(LidDrivenCavitySolver):
    """Finite volume solver for lid-driven cavity problem.

    This solver uses a collocated grid arrangement with Rhie-Chow interpolation
    for pressure-velocity coupling. Supports both SIMPLE and PISO algorithms.

    Parameters
    ----------
    solver_config : SolverConfig
        Physical problem configuration (Re, lid velocity, domain size).
    runtime_config : RuntimeConfig
        Runtime configuration (tolerance, max iterations).
    fv_config : FVConfig
        Finite volume specific configuration (mesh resolution, schemes, etc.).
    """

    def __init__(
        self,
        Re: float,
        mesh_path: str = "data/meshes/structured/fine.msh",
        solver_config: SolverConfig = None,
        fv_config: FVConfig = None,
    ):
        """Initialize FV solver.

        Parameters
        ----------
        Re : float
            Reynolds number.
        mesh_path : str, optional
            Path to mesh file. Default: fine mesh.
        solver_config : SolverConfig, optional
            Advanced solver configuration. If None, uses defaults with specified Re.
        fv_config : FVConfig, optional
            Advanced FV configuration. If None, uses defaults with specified mesh_path.
        """
        # Create configs if not provided
        if solver_config is None:
            solver_config = SolverConfig(Re=Re)
        if fv_config is None:
            fv_config = FVConfig(mesh_path=mesh_path)

        super().__init__(solver_config)
        self.fv_config = fv_config

        # Initialize solution fields
        self.U = None  # Velocity field (n_cells x 2)
        self.p = None  # Pressure field (n_cells)
        self.mdot = None  # Mass flux at faces

        # Set up mesh using base class method
        self._setup_mesh(mesh_path=fv_config.mesh_path)

    def step(self) -> float:
        """Single iteration step.

        Note: The underlying simple_algorithm runs its own iteration loop,
        so this method is not used directly. Instead, solve() is overridden.
        """
        raise NotImplementedError(
            "FVSolver uses simple_algorithm which manages its own iteration loop. "
            "Call solve() instead of step()."
        )

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
        import sys
        from pathlib import Path

        # Ensure fv module is importable
        src_dir = Path(__file__).parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from fv.core.simple_algorithm import simple_algorithm

        n_cells = self.mesh.cell_volumes.shape[0]
        print(f"Starting FV solver with SIMPLE algorithm (Re={self.solver_config.Re}, "
              f"n_cells={n_cells})")

        # Run SIMPLE algorithm
        p, U, mdot, residuals, iterations, converged, u_res, v_res, cont_res = (
            simple_algorithm(
                mesh=self.mesh,
                alpha_uv=self.fv_config.alpha_uv,
                alpha_p=self.fv_config.alpha_p,
                rho=self.rho,
                mu=self.mu,
                max_iter=max_iter,
                tol=tolerance,
                convection_scheme=self.fv_config.convection_scheme,
                limiter=self.fv_config.limiter,
            )
        )

        # Store solution
        self.U = U
        self.p = p
        self.mdot = mdot
        self.converged = converged
        self.iteration_count = iterations

        # Extract residual history (u, v, continuity)
        u_l2norm, v_l2norm, continuity_l2norm = residuals
        # Combine residuals (use max of u, v, continuity as overall residual)
        combined_residuals = np.maximum(
            np.maximum(u_l2norm, v_l2norm), continuity_l2norm
        )
        self.residual_history = combined_residuals.tolist()

        # Package results
        results = Results(
            u=U[:, 0],
            v=U[:, 1],
            p=p,
            res_his=self.residual_history,
            iterations=iterations,
            converged=converged,
            final_alg_residual=combined_residuals[-1] if len(combined_residuals) > 0 else float('inf'),
            wall_time=0.0,  # simple_algorithm doesn't return wall time directly
        )

        return results

    def get_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return velocity field components.

        Returns
        -------
        u : np.ndarray
            x-component of velocity (shape: n_cells).
        v : np.ndarray
            y-component of velocity (shape: n_cells).
        """
        if self.U is None:
            raise RuntimeError("Solver has not been run yet. Call solve() first.")
        return self.U[:, 0], self.U[:, 1]

    def get_pressure_field(self) -> np.ndarray:
        """Return pressure field.

        Returns
        -------
        p : np.ndarray
            Pressure field (shape: n_cells).
        """
        if self.p is None:
            raise RuntimeError("Solver has not been run yet. Call solve() first.")
        return self.p

    def get_cell_centers(self) -> np.ndarray:
        """Return mesh cell centers for plotting.

        Returns
        -------
        cell_centers : np.ndarray
            Cell center coordinates (shape: n_cells x 2).
        """
        return self.mesh.cell_centers
