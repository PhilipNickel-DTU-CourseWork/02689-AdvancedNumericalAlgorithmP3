"""Abstract base solver for lid-driven cavity problem."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from .datastructures import Info


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

    # Subclasses should override this with their config class
    Config = None

    def __init__(self, config: Config = None, **kwargs):
        """Initialize solver with configuration.

        Parameters
        ----------
        config : Config, optional
            Configuration object (FVConfig, SpectralConfig, etc).
            If not provided, kwargs are used to create config.
        **kwargs
            Configuration parameters passed to Config class if config is None.
        """
        # Create config from kwargs if not provided
        if config is None:
            if self.Config is None:
                raise ValueError("Subclass must define Config class attribute")
            config = self.Config(**kwargs)

        self.config = config

        # Compute fluid properties from Reynolds number
        self.rho = 1.0  # Normalized density
        self.mu = self.rho * config.lid_velocity * config.Lx / config.Re

        # Get grid size from subclass
        nx, ny = self._get_grid_size()

        # Create uniform structured grid (common for all solvers)
        self._create_uniform_grid(nx, ny)

        # Create mesh (for FV solvers) or None (for spectral solvers)
        self._create_mesh()


        # Let subclass do solver-specific initialization
        self._setup_solver_specifics()

    def _get_grid_size(self) -> Tuple[int, int]:
        """Return grid dimensions (nx, ny) for this solver.

        Default implementation returns config.nx and config.ny.
        Subclasses can override if they use different naming (e.g., Nx, Ny).

        Returns
        -------
        nx, ny : int
            Number of grid points/cells in x and y directions.
        """
        return self.config.nx, self.config.ny

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

    def _create_mesh(self):
        """Create FV mesh structure.

        Default implementation creates a structured FV mesh using gmsh.
        Spectral solvers can override this to set self.mesh = None.
        """
        from meshing.simple_structured import create_structured_mesh_2d

        self.mesh = create_structured_mesh_2d(
            nx=self.nx,
            ny=self.ny,
            Lx=self.config.Lx,
            Ly=self.config.Ly,
            lid_velocity=self.config.lid_velocity
        )

    def _setup_solver_specifics(self):
        """Solver-specific initialization (optional).

        Called after grid creation. Override this to:
        - Create solver-specific data structures (spectral operators, etc.)
        - Initialize additional solver-specific state

        Default implementation does nothing.
        """
        pass

    @abstractmethod
    def solve(self, tolerance: float = 1e-6, max_iter: int = 1000):
        """Solve the lid-driven cavity problem.

        Stores results in solver attributes:
        - self.fields : Fields dataclass with solution fields
        - self.time_series : TimeSeries dataclass with time series data
        - self.metadata : Metadata dataclass with solver metadata

        Parameters
        ----------
        tolerance : float
            Convergence tolerance.
        max_iter : int
            Maximum iterations.
        """
        pass


    def save(self, filepath):
        """Save results to HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        """
        from dataclasses import asdict
        import h5py
        from pathlib import Path

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dicts
        fields_dict = asdict(self.fields)
        time_series_dict = asdict(self.time_series)
        metadata_dict = asdict(self.metadata)

        with h5py.File(filepath, 'w') as f:
            # Save metadata as root-level attributes
            for key, val in metadata_dict.items():
                # Skip None values and convert to appropriate types
                if val is None:
                    continue
                # Convert strings to bytes for HDF5 compatibility
                if isinstance(val, str):
                    f.attrs[key] = val
                else:
                    f.attrs[key] = val

            # Save fields in a fields group
            fields_grp = f.create_group('fields')
            for key, val in fields_dict.items():
                fields_grp.create_dataset(key, data=val)

            # Add velocity magnitude if u and v are present
            if 'u' in fields_dict and 'v' in fields_dict:
                import numpy as np
                vel_mag = np.sqrt(fields_dict['u']**2 + fields_dict['v']**2)
                fields_grp.create_dataset('velocity_magnitude', data=vel_mag)

            # Save grid_points at root level for compatibility
            if 'grid_points' in fields_dict:
                f.create_dataset('grid_points', data=fields_dict['grid_points'])

            # Save time series in a group
            if time_series_dict:
                ts_grp = f.create_group('time_series')
                for key, val in time_series_dict.items():
                    if val is not None:
                        ts_grp.create_dataset(key, data=val)
