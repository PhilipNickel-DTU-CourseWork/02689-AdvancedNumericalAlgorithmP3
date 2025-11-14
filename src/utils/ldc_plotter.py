"""LDC results plotter for single and multiple runs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyvista as pv


class LDCPlotter:
    """Plotter for lid-driven cavity simulation results.

    Handles both single and multiple runs for comparison.

    Parameters
    ----------
    runs : dict or list of dict
        Single run dict or list of run dicts. Each dict should contain:
        - 'data_path': Path to parquet file
        - 'fields_path': Path to VTK file (optional)
        - 'label': Label for this run (required for multiple runs)
    """

    def __init__(self, runs: dict | list[dict]):
        """Initialize plotter and load data."""
        # Normalize to list
        if isinstance(runs, dict):
            self.runs = [runs]
            self.single_run = True
        else:
            self.runs = runs
            self.single_run = False

        # Validate and load all runs
        self.run_data = []
        for i, run in enumerate(self.runs):
            run_info = self._load_run(run, run_idx=i)
            self.run_data.append(run_info)

        # For single run, expose data at top level for backward compatibility
        if self.single_run:
            rd = self.run_data[0]
            self.data = rd['data']
            self.Re = rd['Re']
            self.residuals = rd['residuals']
            self.points = rd['points']
            self.x = rd['x']
            self.y = rd['y']
            self.u = rd['u']
            self.v = rd['v']
            self.p = rd['p']
            self.vel_mag = rd['vel_mag']
            self.fields_path = rd['fields_path']

    def _load_run(self, run: dict, run_idx: int) -> dict:
        """Load data for a single run."""
        data_path = Path(run['data_path'])
        fields_path = Path(run['fields_path']) if 'fields_path' in run else None

        # Load metadata and residuals
        data = pd.read_parquet(data_path)
        Re = data.loc['config', 'Re'].iloc[0]
        residuals = data.loc['residuals', 'residual'].values

        # Load spatial fields if available
        if fields_path and fields_path.exists():
            mesh = pv.read(fields_path)
            points = mesh.points
            x = points[:, 0]
            y = points[:, 1]
            u = mesh['u']
            v = mesh['v']
            p = mesh['p']
            vel_mag = mesh['velocity_magnitude']
        else:
            points = x = y = u = v = p = vel_mag = None

        # Get label (use provided or generate from run index)
        if not self.single_run and 'label' not in run:
            raise ValueError(f"Run {run_idx} missing 'label' key (required for multiple runs)")
        label = run.get('label', 'Run')

        return {
            'data': data,
            'Re': Re,
            'residuals': residuals,
            'points': points,
            'x': x,
            'y': y,
            'u': u,
            'v': v,
            'p': p,
            'vel_mag': vel_mag,
            'label': label,
            'data_path': data_path,
            'fields_path': fields_path
        }

    def plot_convergence(self, output_path: Optional[Path | str] = None, show: bool = False):
        """Plot convergence history using seaborn.

        Automatically handles single or multiple runs using hue.

        Parameters
        ----------
        output_path : Path or str, optional
            Path to save figure. If None, figure is not saved.
        show : bool, default False
            Whether to show the plot.
        """
        # Create DataFrame for all runs
        dfs = []
        for rd in self.run_data:
            df = pd.DataFrame({
                'Iteration': range(len(rd['residuals'])),
                'Residual': rd['residuals'],
                'Run': rd['label']
            })
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        # Create plot using seaborn relplot
        g = sns.relplot(
            data=combined_df,
            x='Iteration',
            y='Residual',
            hue='Run' if not self.single_run else None,
            kind='line',
            height=5,
            aspect=1.6,
            linewidth=2,
            legend='auto' if not self.single_run else False
        )

        # Set log scale for y-axis
        g.ax.set_yscale('log')
        g.ax.grid(True, alpha=0.3)

        # Set title based on single/multi run
        if self.single_run:
            g.ax.set_title(f'Convergence History (Re = {self.run_data[0]["Re"]:.0f})', fontweight='bold')
        else:
            g.ax.set_title('Convergence Comparison', fontweight='bold')

        if output_path:
            g.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Convergence plot saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_velocity_fields(self, output_path: Optional[Path | str] = None, show: bool = False):
        """Plot velocity components using matplotlib tricontourf.

        Parameters
        ----------
        output_path : Path or str, optional
            Path to save figure. If None, figure is not saved.
        show : bool, default False
            Whether to show the plot.
        """
        if self.points is None:
            raise ValueError("Fields data not available. Provide fields_path during initialization.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # U velocity
        axes[0].tricontourf(self.x, self.y, self.u, levels=20, cmap='RdBu_r')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title('U velocity')
        axes[0].set_aspect('equal')

        # V velocity
        axes[1].tricontourf(self.x, self.y, self.v, levels=20, cmap='RdBu_r')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title('V velocity')
        axes[1].set_aspect('equal')

        # Velocity magnitude
        axes[2].tricontourf(self.x, self.y, self.vel_mag, levels=20, cmap='viridis')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].set_title('Velocity magnitude')
        axes[2].set_aspect('equal')

        plt.suptitle(f'Lid-Driven Cavity: Re = {self.Re:.0f}', fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Velocity plot saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_pressure(self, output_path: Optional[Path | str] = None, show: bool = False):
        """Plot pressure field using matplotlib tricontourf.

        Parameters
        ----------
        output_path : Path or str, optional
            Path to save figure. If None, figure is not saved.
        show : bool, default False
            Whether to show the plot.
        """
        if self.points is None:
            raise ValueError("Fields data not available. Provide fields_path during initialization.")

        fig, ax = plt.subplots(figsize=(8, 7))
        cf = ax.tricontourf(self.x, self.y, self.p, levels=20, cmap='coolwarm')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Pressure field (Re = {self.Re:.0f})')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='Pressure')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Pressure plot saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()
