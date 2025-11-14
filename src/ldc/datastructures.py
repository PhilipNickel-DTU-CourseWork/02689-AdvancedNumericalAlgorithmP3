"""Data structures for solver configuration and results.

This module defines the configuration and result data structures
for lid-driven cavity solvers (both FV and spectral).
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np


@dataclass
class Fields:
    """Base spatial solution fields."""
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    x: np.ndarray
    y: np.ndarray
    grid_points: np.ndarray


@dataclass
class FVFields(Fields):
    """FV-specific fields with mass flux."""
    mdot: np.ndarray = None


@dataclass
class TimeSeries:
    """Time series data common to all solvers."""
    residual: List[float]
    u_residual: List[float] = None
    v_residual: List[float] = None
    continuity_residual: List[float] = None


@dataclass
class Metadata:
    """Base solver metadata combining config and convergence info."""
    # Physics parameters
    Re: float
    lid_velocity: float
    Lx: float
    Ly: float
    # Grid parameters
    nx: int
    ny: int
    # Convergence info
    iterations: int
    converged: bool
    final_residual: float


@dataclass
class FVMetadata(Metadata):
    """FV-specific metadata with discretization parameters."""
    convection_scheme: str = None
    limiter: str = None
    alpha_uv: float = None
    alpha_p: float = None


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
    nx : int
        Number of grid points/cells in x-direction.
    ny : int
        Number of grid points/cells in y-direction.
    """

    Re: float = 100.0
    lid_velocity: float = 1.0
    Lx: float = 1.0
    Ly: float = 1.0
    nx: int = 32
    ny: int = 32


@dataclass
class FVConfig(Config):
    """Finite volume solver configuration.

    Inherits physics parameters (Re, Lx, Ly, lid_velocity, nx, ny) from Config.

    Parameters
    ----------
    convection_scheme : str
        Convection discretization ('upwind', 'TVD', 'central').
    limiter : str
        Limiter for TVD schemes ('MUSCL', 'vanLeer', 'minmod').
    alpha_uv : float
        Under-relaxation factor for momentum equations.
    alpha_p : float
        Under-relaxation factor for pressure correction.
    """

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


