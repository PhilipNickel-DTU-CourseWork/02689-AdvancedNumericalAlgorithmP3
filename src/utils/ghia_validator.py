"""Ghia benchmark validator for lid-driven cavity simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pyvista as pv


class GhiaValidator:
    """Validator for lid-driven cavity results against Ghia et al. (1982) benchmark.

    Parameters
    ----------
    Re : float
        Reynolds number of the simulation.
    fields_path : Path or str
        Path to VTK file with solution fields.
    validation_data_dir : Path or str, optional
        Directory containing Ghia CSV files. If None, uses default location.
    """

    AVAILABLE_RE = [100, 400, 1000, 3200, 5000, 7500, 10000]

    def __init__(
        self,
        Re: float,
        fields_path: Path | str,
        validation_data_dir: Optional[Path | str] = None
    ):
        """Initialize validator and load solution fields from VTK file."""
        self.Re = Re
        self.fields_path = Path(fields_path)

        # Load solution fields from VTK
        mesh = pv.read(self.fields_path)
        self.cell_centers = mesh.points[:, :2]  # Extract x, y coordinates
        self.u = mesh['u']
        self.v = mesh['v']

        # Find closest available Reynolds number
        self.Re_closest = min(self.AVAILABLE_RE, key=lambda x: abs(x - Re))
        if abs(self.Re_closest - Re) > 0.1 * Re:
            print(f"Warning: Using Ghia data for Re={self.Re_closest}, "
                  f"requested Re={Re}")

        # Set validation data directory
        if validation_data_dir is None:
            # Default: project_root/data/validation/ghia
            from utils import get_project_root
            validation_data_dir = get_project_root() / "data" / "validation" / "ghia"
        self.validation_data_dir = Path(validation_data_dir)

        # Load Ghia benchmark data
        self._load_ghia_data()

    def _load_ghia_data(self):
        """Load Ghia benchmark data from CSV files."""
        # U velocity along vertical centerline
        u_file = self.validation_data_dir / f"ghia_Re{self.Re_closest}_u_centerline.csv"
        u_df = pd.read_csv(u_file)
        self.ghia_y = u_df['y'].values
        self.ghia_u = u_df['u'].values

        # V velocity along horizontal centerline
        v_file = self.validation_data_dir / f"ghia_Re{self.Re_closest}_v_centerline.csv"
        v_df = pd.read_csv(v_file)
        self.ghia_x = v_df['x'].values
        self.ghia_v = v_df['v'].values

    def _extract_centerline_u(self):
        """Extract u velocity along vertical centerline (x=0.5)."""
        x = self.cell_centers[:, 0]
        y = self.cell_centers[:, 1]

        # Find cells near vertical centerline (x=0.5)
        tolerance = 0.05
        centerline_mask = np.abs(x - 0.5) < tolerance

        # Sort by y coordinate
        y_centerline = y[centerline_mask]
        u_centerline = self.u[centerline_mask]
        sorted_indices = np.argsort(y_centerline)

        return y_centerline[sorted_indices], u_centerline[sorted_indices]

    def _extract_centerline_v(self):
        """Extract v velocity along horizontal centerline (y=0.5)."""
        x = self.cell_centers[:, 0]
        y = self.cell_centers[:, 1]

        # Find cells near horizontal centerline (y=0.5)
        tolerance = 0.05
        centerline_mask = np.abs(y - 0.5) < tolerance

        # Sort by x coordinate
        x_centerline = x[centerline_mask]
        v_centerline = self.v[centerline_mask]
        sorted_indices = np.argsort(x_centerline)

        return x_centerline[sorted_indices], v_centerline[sorted_indices]

    def plot_validation(self, output_path: Optional[Path | str] = None, show: bool = False):
        """Plot velocity validation against Ghia benchmark using seaborn.

        Creates a two-panel figure with u and v velocity validation side-by-side.

        Parameters
        ----------
        output_path : Path or str, optional
            Path to save figure. If None, figure is not saved.
        show : bool, default False
            Whether to show the plot.
        """
        # Extract centerline data
        y_sim, u_sim = self._extract_centerline_u()
        x_sim, v_sim = self._extract_centerline_v()

        # Prepare simulation data only (for lines)
        u_sim_df = pd.DataFrame({
            'position': u_sim,
            'coordinate': y_sim,
            'component': 'U velocity\n(vertical centerline)'
        })
        v_sim_df = pd.DataFrame({
            'position': v_sim,
            'coordinate': x_sim,
            'component': 'V velocity\n(horizontal centerline)'
        })
        sim_df = pd.concat([u_sim_df, v_sim_df], ignore_index=True)

        # Create faceted plot with simulation data
        g = sns.relplot(
            data=sim_df,
            x='position',
            y='coordinate',
            col='component',
            kind='line',
            marker='o',
            height=5,
            aspect=0.8,
            facet_kws={'sharey': False, 'sharex': False},
            linewidth=1.5,
            markersize=4,
            label='Simulation'
        )

        # Add Ghia data as scatter (no lines)
        # Left panel: U velocity
        g.axes[0, 0].scatter(self.ghia_u, self.ghia_y, marker='s', s=36,
                            facecolors='none', edgecolors='C1', linewidths=1.5,
                            label='Ghia et al. (1982)', zorder=3)

        # Right panel: V velocity
        g.axes[0, 1].scatter(self.ghia_v, self.ghia_x, marker='s', s=36,
                            facecolors='none', edgecolors='C1', linewidths=1.5,
                            label='Ghia et al. (1982)', zorder=3)

        # Set axis labels
        g.axes[0, 0].set_xlabel('u')
        g.axes[0, 0].set_ylabel('y')
        g.axes[0, 1].set_xlabel('v')
        g.axes[0, 1].set_ylabel('x')

        # Add grid and legend to both subplots
        for ax in g.axes.flat:
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=True, loc='best')

        # Set overall title
        g.fig.suptitle(f'Centerline Velocity Validation (Re = {self.Re:.0f})',
                       fontweight='bold', y=1.02)

        plt.tight_layout()

        if output_path:
            g.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Validation plot saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()
