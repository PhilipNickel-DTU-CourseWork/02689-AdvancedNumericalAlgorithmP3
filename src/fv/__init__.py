"""Finite volume solver package.

This package contains the collocated finite volume solver implementation
with SIMPLE/PISO algorithms for pressure-velocity coupling.
"""

# Core solver
from .core.simple_algorithm import simple_algorithm

__all__ = [
    "simple_algorithm",
]
