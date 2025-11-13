"""Lid-driven cavity solver framework.

This module provides the abstract base class and data structures
for implementing lid-driven cavity solvers.
"""

from .base_solver import LidDrivenCavitySolver
from .datastructures import (
    SolverConfig,
    RuntimeConfig,
    FVConfig,
    SpectralConfig,
    Results,
)
from .fv_solver import FVSolver

__all__ = [
    "LidDrivenCavitySolver",
    "SolverConfig",
    "RuntimeConfig",
    "FVConfig",
    "SpectralConfig",
    "Results",
    "FVSolver",
]
