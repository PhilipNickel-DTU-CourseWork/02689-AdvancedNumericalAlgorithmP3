"""Plotting utilities for spectral method visualizations.

This module provides utilities for:
- Automatic style application (seaborn + custom mplstyle)
- Formatting labels and parameters for plots
- Common plotting helpers
- LDC results plotting class

Automatically applies seaborn style and custom utils.mplstyle on import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pyvista as pv


# ==============================================================================
# Style Application
# ==============================================================================


def _apply_styles():
    """Apply seaborn style and custom utils.mplstyle."""
    # Apply seaborn style first
    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        # Fallback if seaborn style not available
        pass

    # Then apply custom style on top
    style_path = Path(__file__).parent / "utils.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))


# Apply styles when module is imported
_apply_styles()


# ==============================================================================
# Formatting Utilities
# ==============================================================================


def format_dt_latex(dt: float | str) -> str:
    """Format a timestep value as LaTeX scientific notation.

    Parameters
    ----------
    dt : float or str
        Timestep value to format. If str and equals '?', returns '?'

    Returns
    -------
    str
        LaTeX-formatted string in the form 'mantissa \\times 10^{exponent}'

    Examples
    --------
    >>> format_dt_latex(0.001)
    '1.00 \\times 10^{-3}'

    """
    if dt == "?":
        return "?"

    dt_str = f"{float(dt):.2e}"
    mantissa, exp = dt_str.split("e")
    exp_int = int(exp)
    return rf"{mantissa} \times 10^{{{exp_int}}}"


def format_parameter_range(
    values: list | tuple,
    name: str,
    latex: bool = True,
) -> str:
    """Format a parameter range for display.

    Parameters
    ----------
    values : list or tuple
        Parameter values (should be sorted)
    name : str
        Parameter name (e.g., 'N', 'L', 'dt')
    latex : bool, default True
        Whether to use LaTeX formatting

    Returns
    -------
    str
        Formatted string

    Examples
    --------
    >>> format_parameter_range([10, 20, 30], 'N')
    '$N \\in [10, 30]$'

    """
    if len(values) == 0:
        return f"{name} = ?"

    if len(values) == 1:
        val = values[0]
        if latex:
            return rf"${name} = {val}$"
        return f"{name} = {val}"

    min_val, max_val = min(values), max(values)

    # Format based on type
    if isinstance(min_val, int) and isinstance(max_val, int):
        range_str = f"[{min_val}, {max_val}]"
    else:
        range_str = f"[{min_val:.1f}, {max_val:.1f}]"

    if latex:
        return rf"${name} \in {range_str}$"
    return f"{name} ∈ {range_str}"


def build_parameter_string(
    params: dict[str, Any],
    separator: str = ", ",
    latex: bool = True,
) -> str:
    """Build a parameter string from a dictionary.

    Parameters
    ----------
    params : dict
        Dictionary of parameter names and values
    separator : str, default ', '
        Separator between parameters
    latex : bool, default True
        Whether to use LaTeX formatting (wraps each param in $ $)

    Returns
    -------
    str
        Formatted parameter string

    Examples
    --------
    >>> build_parameter_string({'N': 100, 'dt': 0.001})
    '$N = 100$, $dt = 1.00 \\times 10^{-3}$'

    """
    parts = []
    for name, value in params.items():
        if isinstance(value, (list, tuple)):
            parts.append(format_parameter_range(value, name, latex=latex))
        else:
            # Handle special formatting for dt
            if "dt" in name.lower() or "Delta t" in name:
                value_str = format_dt_latex(value)
                if latex:
                    parts.append(rf"${name} = {value_str}$")
                else:
                    parts.append(f"{name} = {value_str}")
            else:
                if latex:
                    parts.append(rf"${name} = {value}$")
                else:
                    parts.append(f"{name} = {value}")

    return separator.join(parts)


# ==============================================================================
# LDC Results Plotter
# ==============================================================================


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



# ==============================================================================
# Ghia Benchmark Validator
# ==============================================================================


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
