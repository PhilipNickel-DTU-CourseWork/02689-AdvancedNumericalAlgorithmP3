"""Utility modules for plotting and project management."""

from pathlib import Path
from . import plotting
from .ldc_plotter import LDCPlotter
from .ghia_validator import GhiaValidator

__all__ = ["plotting", "get_project_root", "LDCPlotter", "GhiaValidator"]


def get_project_root() -> Path:
    """Get project root directory.

    Returns
    -------
    Path
        Project root directory (contains pyproject.toml).
    """
    # Start from this file and search upward for pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Fallback: assume standard structure
    return Path(__file__).resolve().parent.parent.parent
