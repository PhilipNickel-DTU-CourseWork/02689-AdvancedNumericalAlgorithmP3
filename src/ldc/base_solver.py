"""Abstract base solver for lid-driven cavity problem."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from .datastructures import Config, Results


class LidDrivenCavitySolver(ABC):
    """Abstract base solver for lid-driven cavity problem.

    This base class handles:
    - Physics parameters (Re, viscosity, density)
    - Uniform structured grid creation
    - Common grid properties (dx, dy, X, Y, grid_points)

    Subclasses only need to:
    - Specify grid size via _get_grid_size()
    - Do solver-specific setup via _setup_solver_specifics()
    - Implement solve() method

    Parameters
    ----------
    config : Config (or subclass like FVConfig, SpectralConfig)
        Configuration with physics (Re, Lx, Ly, lid_velocity) and numerics (nx, ny, etc).
    """

    def __init__(self, config: Config):
        """Initialize solver with configuration.

        Parameters
        ----------
        config : Config
            Configuration object (FVConfig, SpectralConfig, etc).
        """
        self.config = config

        # Compute fluid properties from Reynolds number
        self.rho = 1.0  # Normalized density
        self.mu = self.rho * config.lid_velocity * config.Lx / config.Re

        # Get grid size from subclass
        nx, ny = self._get_grid_size()

        # Create uniform structured grid (common for all solvers)
        self._create_uniform_grid(nx, ny)

        # Solver state
        self.converged = False
        self.iterations = 0
        self.residual_history = []

        # Let subclass do solver-specific initialization
        self._setup_solver_specifics()

    @abstractmethod
    def _get_grid_size(self) -> Tuple[int, int]:
        """Return grid dimensions (nx, ny) for this solver.

        Returns
        -------
        nx, ny : int
            Number of grid points/cells in x and y directions.
        """
        pass

    def _create_uniform_grid(self, nx: int, ny: int):
        """Create uniform structured grid (shared by all solvers).

        Creates grid arrays that are accessible to subclasses:
        - self.x, self.y : 1D coordinate arrays
        - self.X, self.Y : 2D meshgrid arrays
        - self.dx, self.dy : Grid spacing
        - self.grid_points : Flattened (N, 2) array of coordinates
        - self.nx, self.ny : Grid dimensions

        Parameters
        ----------
        nx, ny : int
            Number of grid points in x and y.
        """
        self.nx = nx
        self.ny = ny

        # Create 1D coordinates
        self.x = np.linspace(0, self.config.Lx, nx)
        self.y = np.linspace(0, self.config.Ly, ny)

        # Create 2D meshgrid
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Grid spacing
        self.dx = self.x[1] - self.x[0] if nx > 1 else self.config.Lx
        self.dy = self.y[1] - self.y[0] if ny > 1 else self.config.Ly

        # Flattened grid points for compatibility
        self.grid_points = np.column_stack([self.X.flatten(), self.Y.flatten()])

    @abstractmethod
    def _setup_solver_specifics(self):
        """Solver-specific initialization.

        Called after grid creation. Use this to:
        - Create solver-specific data structures (FV mesh, spectral operators, etc.)
        - Initialize solution fields
        - Setup any other solver-specific state
        """
        pass

    @abstractmethod
    def solve(self, tolerance: float = 1e-6, max_iter: int = 1000) -> Results:
        """Solve the lid-driven cavity problem.

        Parameters
        ----------
        tolerance : float
            Convergence tolerance.
        max_iter : int
            Maximum iterations.

        Returns
        -------
        Results
            Solution data with fields, time_series, and metadata.
        """
        pass

    def _build_results(self, fields: dict, time_series: dict, solver_metadata: dict) -> Results:
        """Helper to build Results object with config metadata.

        Parameters
        ----------
        fields : dict
            Spatial fields (u, v, p, etc.) as arrays.
        time_series : dict
            Time series data (residuals, etc.) as lists.
        solver_metadata : dict
            Solver-specific metadata (iterations, converged, etc.).

        Returns
        -------
        Results
            Complete results object.
        """
        # Combine config and solver metadata
        metadata = {
            'Re': self.config.Re,
            'Lx': self.config.Lx,
            'Ly': self.config.Ly,
            'lid_velocity': self.config.lid_velocity,
            'nx': self.nx,
            'ny': self.ny,
            **solver_metadata
        }

        return Results(
            fields=fields,
            time_series=time_series,
            metadata=metadata
        )

    def save(self, filepath, results: Results):
        """Save results to HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        results : Results
            Results object to save.
        """
        import h5py
        from pathlib import Path

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            # Save metadata as root-level attributes
            for key, val in results.metadata.items():
                f.attrs[key] = val

            # Save fields in a fields group
            fields_grp = f.create_group('fields')
            for key, val in results.fields.items():
                fields_grp.create_dataset(key, data=val)

            # Add velocity magnitude if u and v are present
            if 'u' in results.fields and 'v' in results.fields:
                import numpy as np
                vel_mag = np.sqrt(results.fields['u']**2 + results.fields['v']**2)
                fields_grp.create_dataset('velocity_magnitude', data=vel_mag)

            # Save grid_points at root level for compatibility
            if 'grid_points' in results.fields:
                f.create_dataset('grid_points', data=results.fields['grid_points'])

            # Save time series in a group
            if results.time_series:
                ts_grp = f.create_group('time_series')
                for key, val in results.time_series.items():
                    ts_grp.create_dataset(key, data=val)
