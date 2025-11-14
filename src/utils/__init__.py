"""Utility modules for data handling, plotting, and CLI.

Import conveniences:
- from utils import datatools  # For data operations
- from utils import plotting    # For plotting operations
- from utils import cli         # For command-line argument parsing
"""

from pathlib import Path
from . import datatools, plotting, cli

__all__ = ["datatools", "plotting", "cli", "get_project_root"]


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
