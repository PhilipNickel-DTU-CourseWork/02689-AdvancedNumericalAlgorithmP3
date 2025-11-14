"""Data structures for solver configuration and results.

This module defines the configuration and result data structures
for lid-driven cavity solvers (both FV and spectral).
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np


@dataclass
class Config:
    """Base configuration with shared physics parameters.

    All solver configs inherit from this to ensure consistent
    problem definition across different numerical methods.

    Parameters
    ----------
    Re : float
        Reynolds number (Re = U*L/nu).
    lid_velocity : float
        Velocity of the moving lid (typically 1.0).
    Lx : float
        Domain width (x-direction).
    Ly : float
        Domain height (y-direction).
    """

    Re: float = 100.0
    lid_velocity: float = 1.0
    Lx: float = 1.0
    Ly: float = 1.0


@dataclass
class FVConfig(Config):
    """Finite volume solver configuration.

    Inherits physics parameters (Re, Lx, Ly, lid_velocity) from Config.

    Parameters
    ----------
    nx : int
        Number of cells in x-direction.
    ny : int
        Number of cells in y-direction.
    convection_scheme : str
        Convection discretization ('upwind', 'TVD', 'central').
    limiter : str
        Limiter for TVD schemes ('MUSCL', 'vanLeer', 'minmod').
    alpha_uv : float
        Under-relaxation factor for momentum equations.
    alpha_p : float
        Under-relaxation factor for pressure correction.
    """

    nx: int = 32
    ny: int = 32
    convection_scheme: str = "TVD"
    limiter: str = "MUSCL"
    alpha_uv: float = 0.7
    alpha_p: float = 0.3


@dataclass
class SpectralConfig(Config):
    """Pseudo-spectral solver configuration.

    Inherits physics parameters (Re, Lx, Ly, lid_velocity) from Config.

    Parameters
    ----------
    Nx : int
        Number of grid points in x-direction.
    Ny : int
        Number of grid points in y-direction.
    differentiation_method : str
        Method for spatial derivatives ('fft', 'chebyshev', 'matrix').
    time_scheme : str
        Time integration scheme ('rk4', 'ab2', 'euler').
    dt : float
        Time step size.
    dealiasing : bool
        Use 3/2 rule for dealiasing (for FFT-based methods).
    multigrid : bool
        Use multigrid acceleration.
    mg_levels : int
        Number of multigrid levels (if multigrid=True).
    """

    Nx: int = 64
    Ny: int = 64
    differentiation_method: str = "fft"  # 'fft', 'chebyshev', 'matrix'
    time_scheme: str = "rk4"
    dt: float = 0.001
    dealiasing: bool = True
    multigrid: bool = False
    mg_levels: int = 3


@dataclass
class Results:
    """Container for all solver results grouped by dimensionality.

    This structure organizes data by dimensionality for clean HDF5 storage
    and pandas integration:
    - fields: spatial data (n_cells dimension)
    - time_series: temporal data (n_iterations dimension)
    - metadata: scalar configuration and convergence info

    Parameters
    ----------
    fields : Dict[str, np.ndarray]
        Spatial solution fields (u, v, p) with shape (n_cells,).
    time_series : Dict[str, List]
        Time-series data (residuals, etc.) as lists (length = n_iterations).
        Lists are more natural since length is unknown during solving.
    metadata : Dict[str, Any]
        Scalar metadata including solver config and convergence info.

    Examples
    --------
    >>> results = Results(
    ...     fields={'u': u_array, 'v': v_array, 'p': p_array},
    ...     time_series={'residual': [0.1, 0.05, 0.01, ...]},
    ...     metadata={'Re': 100.0, 'iterations': 250, 'converged': True}
    ... )
    """

    fields: Dict[str, np.ndarray]
    time_series: Dict[str, List]
    metadata: Dict[str, Any]
