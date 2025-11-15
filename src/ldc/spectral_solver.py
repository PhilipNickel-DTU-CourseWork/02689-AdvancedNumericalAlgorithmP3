"""Spectral solver for lid-driven cavity.

This module provides a concrete spectral solver with basic implementation.
Subclasses can override specific methods to add features like multigrid
acceleration or alternative discretization schemes.
"""

import numpy as np
import time

from .base_solver import LidDrivenCavitySolver
from .datastructures import SpectralConfig


class SpectralSolver(LidDrivenCavitySolver):
    # Make config class accessible via solver
    Config = SpectralConfig
    """Spectral solver for lid-driven cavity problem.

    Provides common infrastructure:
    - Grid generation (uniform, Chebyshev, etc.)
    - Time-stepping framework
    - Boundary condition handling
    - Solution state management

    Subclasses can override:
    - _setup_differentiation() : Custom differentiation operators
    - _compute_rhs() : Alternative RHS computation
    - _solve_pressure() : Different pressure solvers
    - _time_step() : Custom time integration schemes

    Parameters
    ----------
    config : SpectralConfig
        Spectral solver configuration (physics + numerics).
    """

    def __init__(self, **kwargs):
        """Initialize pseudo-spectral solver.

        Parameters
        ----------
        **kwargs
            Configuration parameters passed to SpectralConfig.
            Can also pass config=SpectralConfig(...) directly.
        """
        # Base class handles config creation from kwargs
        super().__init__(**kwargs)

        # Store spectral-specific parameters
        self.dt = self.config.dt

        # Store grid dimensions from base class for convenience
        self.Nx = self.nx
        self.Ny = self.ny

        # Initialize 2D working arrays for spectral computations
        self.u_2d = np.zeros((self.Ny, self.Nx))
        self.v_2d = np.zeros((self.Ny, self.Nx))
        self.p_2d = np.zeros((self.Ny, self.Nx))

    def _get_grid_size(self):
        """Return grid dimensions from spectral config.

        Returns
        -------
        nx, ny : int
            Number of grid points in x and y directions.
        """
        return self.config.Nx, self.config.Ny

    def _create_mesh(self):
        """Spectral solvers don't need FV mesh."""
        self.mesh = None

    def _setup_solver_specifics(self):
        """Setup spectral differentiation operators.

        This method is called by base class after grid creation.
        Override _setup_differentiation() in subclasses to create
        FFT plans, differentiation matrices, wavenumber vectors, etc.
        """
        self._setup_differentiation()

    def _setup_differentiation(self):
        """Setup spatial differentiation operators.

        Override this to create:
        - FFT plans
        - Differentiation matrices
        - Wavenumber vectors
        etc.

        Base implementation does nothing (placeholder).
        """
        pass

    def _compute_rhs(self, u, v, p):
        """Compute right-hand side of momentum equations.

        Parameters
        ----------
        u, v, p : np.ndarray
            Current velocity and pressure fields.

        Returns
        -------
        rhs_u, rhs_v : np.ndarray
            Right-hand side for u and v momentum equations.
        """
        raise NotImplementedError("Subclasses must implement _compute_rhs()")

    def _solve_pressure(self, u, v):
        """Solve pressure Poisson equation.

        Parameters
        ----------
        u, v : np.ndarray
            Intermediate velocity fields.

        Returns
        -------
        p : np.ndarray
            Pressure field.
        """
        raise NotImplementedError("Subclasses must implement _solve_pressure()")

    def _apply_boundary_conditions(self):
        """Apply boundary conditions to velocity fields.

        Lid-driven cavity BCs:
        - Top (y = Ly): u = lid_velocity, v = 0
        - Bottom, left, right: u = 0, v = 0
        """
        # Top boundary (lid)
        self.u_2d[-1, :] = self.config.lid_velocity
        self.v_2d[-1, :] = 0.0

        # Bottom boundary
        self.u_2d[0, :] = 0.0
        self.v_2d[0, :] = 0.0

        # Left boundary
        self.u_2d[:, 0] = 0.0
        self.v_2d[:, 0] = 0.0

        # Right boundary
        self.u_2d[:, -1] = 0.0
        self.v_2d[:, -1] = 0.0

    def _compute_residual(self):
        """Compute L2 norm of velocity residual for convergence check.

        Returns
        -------
        residual : float
            L2 norm of velocity change.
        """
        # Simple L2 norm of velocity magnitude
        vel_norm = np.sqrt(np.sum(self.u_2d**2 + self.v_2d**2))
        return vel_norm

    def solve(self, tolerance=1e-6, max_iter=1000):
        """Solve lid-driven cavity using pseudo-spectral time-stepping.

        Parameters
        ----------
        tolerance : float
            Convergence tolerance for steady-state residual.
        max_iter : int
            Maximum number of time steps.

        Returns
        -------
        Results
            Solution data with velocity, pressure, and convergence info.
        """
        start_time = time.time()
        self.residual_history = []

        print(f"Starting {self.__class__.__name__} (Re={self.config.Re:.0f}, "
              f"grid={self.Ny}x{self.Nx}, dt={self.dt})")

        # Time-stepping loop
        for n in range(max_iter):
            # Store previous solution
            u_old = self.u_2d.copy()
            v_old = self.v_2d.copy()

            # Time integration step (subclass implements specific scheme)
            self._time_step()

            # Apply boundary conditions
            self._apply_boundary_conditions()

            # Compute residual
            du = self.u_2d - u_old
            dv = self.v_2d - v_old
            residual = np.sqrt(np.sum(du**2 + dv**2)) / (self.Nx * self.Ny)
            self.residual_history.append(residual)

            # Check convergence every 10 steps
            if n % 10 == 0:
                print(f"  Step {n}: residual = {residual:.6e}")

            if residual < tolerance:
                self.converged = True
                self.iterations = n + 1
                print(f"Converged at step {n}")
                break

        else:
            self.converged = False
            self.iterations = max_iter
            print(f"Did not converge within {max_iter} steps")

        wall_time = time.time() - start_time
        print(f"Solver finished in {wall_time:.2f} seconds.")

        # Build and return results using base class helper
        return self._build_results(
            fields={
                'u': self.u_2d.flatten(),
                'v': self.v_2d.flatten(),
                'p': self.p_2d.flatten(),
            },
            time_series={
                'residual': self.residual_history,
            },
            solver_metadata={
                'iterations': self.iterations,
                'converged': self.converged,
                'final_residual': self.residual_history[-1] if self.residual_history else float('inf'),
                'wall_time': wall_time,
            }
        )

    def _time_step(self):
        """Advance solution by one time step.

        Override this to implement specific time integration schemes (RK4, AB2, etc.)
        and differentiation methods (FFT, matrix, etc.).
        """
        raise NotImplementedError("Subclasses must implement _time_step()")
