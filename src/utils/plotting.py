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

    Parameters
    ----------
    data_path : Path or str
        Path to parquet file with metadata and residuals.
    fields_path : Path or str, optional
        Path to VTK file with spatial fields. If None, field plots are disabled.
    """

    def __init__(self, data_path: Path | str, fields_path: Optional[Path | str] = None):
        """Initialize plotter and load data."""
        self.data_path = Path(data_path)
        self.fields_path = Path(fields_path) if fields_path else None

        # Load metadata and residuals
        self.data = pd.read_parquet(self.data_path)
        self.Re = self.data.loc['config', 'Re'].iloc[0]
        self.residuals = self.data.loc['residuals', 'residual'].values

        # Load spatial fields if available
        if self.fields_path and self.fields_path.exists():
            mesh = pv.read(self.fields_path)
            self.points = mesh.points
            self.x = self.points[:, 0]
            self.y = self.points[:, 1]
            self.u = mesh['u']
            self.v = mesh['v']
            self.p = mesh['p']
            self.vel_mag = mesh['velocity_magnitude']
        else:
            self.points = None

    def plot_convergence(self, output_path: Optional[Path | str] = None, show: bool = False):
        """Plot convergence history using seaborn.

        Parameters
        ----------
        output_path : Path or str, optional
            Path to save figure. If None, figure is not saved.
        show : bool, default False
            Whether to show the plot.
        """
        # Create DataFrame for seaborn
        df = pd.DataFrame({
            'Iteration': range(len(self.residuals)),
            'Residual': self.residuals
        })

        # Create plot using seaborn relplot
        g = sns.relplot(
            data=df,
            x='Iteration',
            y='Residual',
            kind='line',
            height=5,
            aspect=1.6,
            linewidth=2
        )

        # Set log scale for y-axis
        g.ax.set_yscale('log')
        g.ax.grid(True, alpha=0.3)
        g.ax.set_title(f'Convergence History (Re = {self.Re:.0f})', fontweight='bold')

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
