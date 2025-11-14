"""Data structures for solver configuration and results.

This module defines the configuration and result data structures
for lid-driven cavity solvers (both FV and spectral).
"""
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np


@dataclass
class SolverConfig:
    """Physical problem configuration for lid-driven cavity.

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
class RuntimeConfig:
    """Runtime configuration for iterative solvers.

    Parameters
    ----------
    tolerance : float
        Convergence tolerance for residual norm.
    max_iter : int
        Maximum number of iterations/pseudo-timesteps.
    N : Optional[int]
        Grid resolution parameter. For FV: cells per direction.
        For spectral: polynomial order or number of modes.
    """

    tolerance: float = 1e-6
    max_iter: int = 1000
    N: Optional[int] = None  # Solver-specific grid/order parameter


@dataclass
class FVConfig:
    """Finite volume solver-specific configuration.

    Parameters
    ----------
    mesh_path : str
        Path to mesh file (.msh format).
    convection_scheme : str
        Convection discretization ('upwind', 'TVD', 'central').
    limiter : str
        Limiter for TVD schemes ('MUSCL', 'vanLeer', 'minmod').
    alpha_uv : float
        Under-relaxation factor for momentum equations.
    alpha_p : float
        Under-relaxation factor for pressure correction.
    """

    mesh_path: str = "data/meshes/structured/fine.msh"
    convection_scheme: str = "TVD"
    limiter: str = "MUSCL"
    alpha_uv: float = 0.7
    alpha_p: float = 0.3


@dataclass
class SolutionFields:
    """Container for spatial solution fields.

    Parameters
    ----------
    u : np.ndarray
        x-component of velocity field.
    v : np.ndarray
        y-component of velocity field.
    p : np.ndarray
        Pressure field.
    """

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray


@dataclass
class ConvergenceResults:
    """Container for convergence metadata.

    Parameters
    ----------
    iterations : int
        Total number of iterations performed.
    converged : bool
        Whether the solver converged within tolerance.
    final_alg_residual : float
        Final residual value.
    wall_time : float
        Total wall-clock time in seconds.
    """

    iterations: int = 0
    converged: bool = False
    final_alg_residual: float = float('inf')
    wall_time: float = 0.0


@dataclass
class Results:
    """Container for all solver results.

    Parameters
    ----------
    convergence : ConvergenceResults
        Convergence metadata.
    fields : SolutionFields
        Spatial solution fields (u, v, p).
    res_his : List[float]
        Residual history over iterations.
    """

    convergence: ConvergenceResults
    fields: SolutionFields
    res_his: List[float] = field(default_factory=list)


