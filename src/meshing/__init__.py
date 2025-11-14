"""Simplified mesh generation for structured quad meshes."""

from .mesh_data import MeshData2D
from .structured_inmemory import create_structured_mesh_2d

__all__ = ["MeshData2D", "create_structured_mesh_2d"]
