"""Abstract base solver for lid-driven cavity problem."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import pyvista as pv
from dataclasses import asdict

from .datastructures import SolverConfig, Results


class LidDrivenCavitySolver(ABC):
    """Abstract base solver defining the lid-driven cavity problem.

    This class handles:
    - Problem definition (Re, lid velocity, domain size)
    - Fluid properties computation (ρ, μ)
    - Mesh/grid generation and loading
    - Boundary condition definition

    Subclasses implement solver-specific methods:
    - How to solve (FV: SIMPLE, Spectral: time-stepping)
    - How to apply BCs (FV: face values, Spectral: Fourier)
    - How to extract results
    """

    def __init__(self, solver_config: SolverConfig):
        """Initialize base solver with problem configuration.

        Parameters
        ----------
        solver_config : SolverConfig
            Physical problem configuration (Re, lid velocity, domain).
        """
        self.solver_config = solver_config

        # Compute fluid properties from Reynolds number
        # Re = ρ * U * L / μ  =>  μ = ρ * U * L / Re
        self.rho = 1.0  # Density (normalized)
        self.mu = (
            self.rho * solver_config.lid_velocity * solver_config.Lx / solver_config.Re
        )

        # Solver state
        self.converged = False
        self.iterations = 0
        self.residual_history = []

        # Mesh/grid (populated by subclass via _setup_mesh)
        self.mesh = None

    # ---- Abstract methods: subclasses must implement ----

    @abstractmethod
    def solve(self, tolerance: float = 1e-6, max_iter: int = 1000) -> Results:
        """Solve the lid-driven cavity problem.

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
        pass

    @abstractmethod
    def get_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return velocity field components.

        Returns
        -------
        u, v : np.ndarray
            x and y components of velocity.
        """
        pass

    @abstractmethod
    def get_pressure_field(self) -> np.ndarray:
        """Return pressure field.

        Returns
        -------
        p : np.ndarray
            Pressure field.
        """
        pass

    @abstractmethod
    def get_cell_centers(self) -> np.ndarray:
        """Return spatial coordinates for solution fields.

        Returns
        -------
        coordinates : np.ndarray
            Spatial coordinates (shape: n_points x 2).
        """
        pass

    # ---- Shared methods: mesh/grid and BC setup ----

    def _create_lid_driven_cavity_bc_config(self) -> dict:
        """Create boundary condition configuration for lid-driven cavity.

        Returns
        -------
        dict
            BC configuration with:
            - Top wall: u = lid_velocity, v = 0 (moving lid)
            - Other walls: u = 0, v = 0 (no-slip)
            - All walls: zero gradient for pressure

        Notes
        -----
        This defines the physical BCs. How they're applied is solver-specific.
        """
        return {
            "boundaries": {
                "top": {
                    "velocity": {
                        "bc": "dirichlet",
                        "value": [float(self.solver_config.lid_velocity), 0.0],
                    },
                    "pressure": {"bc": "neumann", "value": 0.0},
                },
                "bottom": {
                    "velocity": {"bc": "dirichlet", "value": [0.0, 0.0]},
                    "pressure": {"bc": "neumann", "value": 0.0},
                },
                "left": {
                    "velocity": {"bc": "dirichlet", "value": [0.0, 0.0]},
                    "pressure": {"bc": "neumann", "value": 0.0},
                },
                "right": {
                    "velocity": {"bc": "dirichlet", "value": [0.0, 0.0]},
                    "pressure": {"bc": "neumann", "value": 0.0},
                },
            }
        }

    def _setup_mesh(self, mesh_path: str):
        """Load mesh file with boundary conditions.

        Parameters
        ----------
        mesh_path : str
            Path to the mesh file (.msh format).

        Notes
        -----
        - FV solver uses full MeshData2D (faces, connectivity, volumes)
        - Spectral solver uses just grid points (uniform spacing)
        """
        # Ensure meshing module is importable
        src_dir = Path(__file__).parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from meshing import load_mesh

        # Create BC config
        bc_config = self._create_lid_driven_cavity_bc_config()
        cache_dir = Path.home() / ".cache" / "ldc_solver"
        cache_dir.mkdir(parents=True, exist_ok=True)
        bc_config_file = cache_dir / "lid_driven_cavity_bc.yaml"
        with open(bc_config_file, "w") as f:
            yaml.dump(bc_config, f)

        # Load mesh with BC config
        self.mesh = load_mesh(mesh_path, bc_config_file=str(bc_config_file))

    def get_solver_specific_config(self) -> Dict[str, Any]:
        """Return solver-specific configuration.

        Returns
        -------
        dict
            Solver-specific configuration. Override in subclasses.
        """
        return {}

    def save_fields(self, output_path: Path, results: Results):
        """Save spatial solution fields to VTK file.

        Parameters
        ----------
        output_path : Path
            Output file path (should end in .vtp).
        results : Results
            Solution results containing fields.
        """
        # Get spatial coordinates
        cell_centers = self.get_cell_centers()
        points = np.column_stack([
            cell_centers[:, 0],
            cell_centers[:, 1],
            np.zeros(len(cell_centers))
        ])

        # Create point cloud
        cloud = pv.PolyData(points)
        cloud['u'] = results.fields.u
        cloud['v'] = results.fields.v
        cloud['p'] = results.fields.p
        cloud['velocity_magnitude'] = np.sqrt(
            results.fields.u**2 + results.fields.v**2
        )

        # Save to VTK
        cloud.save(output_path)

    def save_data(self, output_path: Path, results: Results):
        """Save metadata and time-series to Parquet file.

        Parameters
        ----------
        output_path : Path
            Output file path (should end in .parquet).
        results : Results
            Solution results containing convergence info and residuals.
        """
        # Config - combine solver and solver-specific config
        config_df = pd.concat([
            pd.DataFrame([asdict(self.solver_config)]),
            pd.DataFrame([self.get_solver_specific_config()])
        ], axis=1).assign(data_type='config', index=0)

        # Results - convergence info
        results_df = pd.DataFrame([asdict(results.convergence)]).assign(
            data_type='results', index=0
        )

        # Residuals - time-series
        residuals_df = pd.DataFrame(
            results.res_his, columns=['residual']
        ).reset_index().assign(data_type='residuals')

        # Combine and set MultiIndex
        all_data = pd.concat([config_df, results_df, residuals_df], ignore_index=True)
        all_data = all_data.set_index(['data_type', 'index'])

        # Save to Parquet
        all_data.to_parquet(output_path)
