"""
Unit tests for the mesh module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from algorithms.mesh import Mesh2D


def test_mesh_creation():
    """Test basic mesh creation."""
    mesh = Mesh2D(11, 11, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
    
    assert mesh.nx == 11
    assert mesh.ny == 11
    assert mesh.xmin == 0.0
    assert mesh.xmax == 1.0
    assert mesh.ymin == 0.0
    assert mesh.ymax == 1.0
    assert mesh.dx == 0.1
    assert mesh.dy == 0.1
    
    print("✓ test_mesh_creation passed")


def test_mesh_grids():
    """Test mesh grid generation."""
    mesh = Mesh2D(5, 5)
    
    assert mesh.X.shape == (5, 5)
    assert mesh.Y.shape == (5, 5)
    assert mesh.x.shape == (5,)
    assert mesh.y.shape == (5,)
    
    # Check corner values
    assert mesh.X[0, 0] == 0.0
    assert mesh.X[-1, -1] == 1.0
    assert mesh.Y[0, 0] == 0.0
    assert mesh.Y[-1, -1] == 1.0
    
    print("✓ test_mesh_grids passed")


def test_interior_points():
    """Test interior points identification."""
    mesh = Mesh2D(5, 5)
    i, j = mesh.get_interior_points()
    
    assert len(i) == 3  # Points 1, 2, 3
    assert len(j) == 3
    
    print("✓ test_interior_points passed")


def test_boundary_mask():
    """Test boundary mask generation."""
    mesh = Mesh2D(5, 5)
    mask = mesh.get_boundary_mask()
    
    # Check that all boundary points are marked
    assert np.all(mask[0, :])   # Bottom
    assert np.all(mask[-1, :])  # Top
    assert np.all(mask[:, 0])   # Left
    assert np.all(mask[:, -1])  # Right
    
    # Check that interior points are not marked
    assert not np.any(mask[1:-1, 1:-1])
    
    print("✓ test_boundary_mask passed")


def test_custom_domain():
    """Test mesh creation with custom domain."""
    mesh = Mesh2D(11, 11, xmin=-1.0, xmax=2.0, ymin=-2.0, ymax=3.0)
    
    assert mesh.xmin == -1.0
    assert mesh.xmax == 2.0
    assert mesh.ymin == -2.0
    assert mesh.ymax == 3.0
    
    # Check grid spacing
    assert np.isclose(mesh.dx, 0.3)
    assert np.isclose(mesh.dy, 0.5)
    
    print("✓ test_custom_domain passed")


if __name__ == "__main__":
    print("Running mesh tests...")
    print("-" * 60)
    
    test_mesh_creation()
    test_mesh_grids()
    test_interior_points()
    test_boundary_mask()
    test_custom_domain()
    
    print("-" * 60)
    print("All mesh tests passed!")
