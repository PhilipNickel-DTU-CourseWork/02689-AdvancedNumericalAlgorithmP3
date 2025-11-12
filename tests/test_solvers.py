"""
Unit tests for the Poisson solver.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from algorithms import PoissonSolver, Mesh2D
from algorithms.utils import compute_error


def test_poisson_solver_direct():
    """Test Poisson solver with direct method."""
    # Create mesh
    mesh = Mesh2D(21, 21)
    
    # Define problem: u_xx + u_yy = 2π² sin(πx) sin(πy)
    def source_function(x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def boundary_function(x, y):
        return np.zeros_like(x)
    
    # Exact solution
    def exact_solution(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Solve
    solver = PoissonSolver(mesh)
    u_numerical = solver.solve(source_function, boundary_function, method='direct')
    
    # Compute exact solution
    u_exact = exact_solution(mesh.X, mesh.Y)
    
    # Check error
    errors = compute_error(u_numerical, u_exact)
    
    # Error should be small for this grid size
    assert errors['l2'] < 5e-3, f"L2 error too large: {errors['l2']}"
    assert errors['linf'] < 1e-2, f"Linf error too large: {errors['linf']}"
    
    print(f"✓ test_poisson_solver_direct passed (L2 error: {errors['l2']:.6e})")


def test_poisson_solver_jacobi():
    """Test Poisson solver with Jacobi iteration."""
    # Use smaller mesh for iterative method
    mesh = Mesh2D(11, 11)
    
    def source_function(x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def boundary_function(x, y):
        return np.zeros_like(x)
    
    def exact_solution(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Solve
    solver = PoissonSolver(mesh)
    u_numerical = solver.solve(source_function, boundary_function, method='jacobi')
    
    # Compute exact solution
    u_exact = exact_solution(mesh.X, mesh.Y)
    
    # Check error
    errors = compute_error(u_numerical, u_exact)
    
    assert errors['l2'] < 1e-2, f"L2 error too large: {errors['l2']}"
    
    print(f"✓ test_poisson_solver_jacobi passed (L2 error: {errors['l2']:.6e})")


def test_poisson_solver_gauss_seidel():
    """Test Poisson solver with Gauss-Seidel iteration."""
    mesh = Mesh2D(11, 11)
    
    def source_function(x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def boundary_function(x, y):
        return np.zeros_like(x)
    
    def exact_solution(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Solve
    solver = PoissonSolver(mesh)
    u_numerical = solver.solve(source_function, boundary_function, method='gauss_seidel')
    
    # Compute exact solution
    u_exact = exact_solution(mesh.X, mesh.Y)
    
    # Check error
    errors = compute_error(u_numerical, u_exact)
    
    assert errors['l2'] < 1e-2, f"L2 error too large: {errors['l2']}"
    
    print(f"✓ test_poisson_solver_gauss_seidel passed (L2 error: {errors['l2']:.6e})")


def test_boundary_conditions():
    """Test that boundary conditions are correctly applied."""
    mesh = Mesh2D(11, 11)
    
    def source_function(x, y):
        return np.zeros_like(x)
    
    def boundary_function(x, y):
        return x + y  # Non-zero boundary
    
    solver = PoissonSolver(mesh)
    u = solver.solve(source_function, boundary_function, method='direct')
    
    # Check boundaries
    assert np.allclose(u[0, :], mesh.X[0, :] + mesh.Y[0, :])
    assert np.allclose(u[-1, :], mesh.X[-1, :] + mesh.Y[-1, :])
    assert np.allclose(u[:, 0], mesh.X[:, 0] + mesh.Y[:, 0])
    assert np.allclose(u[:, -1], mesh.X[:, -1] + mesh.Y[:, -1])
    
    print("✓ test_boundary_conditions passed")


def test_convergence_order():
    """Test that the method achieves expected convergence order."""
    def source_function(x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def boundary_function(x, y):
        return np.zeros_like(x)
    
    def exact_solution(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Test with different mesh sizes
    mesh_sizes = [11, 21, 41]
    errors = []
    h_values = []
    
    for n in mesh_sizes:
        mesh = Mesh2D(n, n)
        solver = PoissonSolver(mesh)
        u_num = solver.solve(source_function, boundary_function, method='direct')
        u_exact = exact_solution(mesh.X, mesh.Y)
        
        error_dict = compute_error(u_num, u_exact)
        errors.append(error_dict['l2'])
        h_values.append(mesh.dx)
    
    # Check convergence order (should be approximately 2 for second-order method)
    for i in range(1, len(mesh_sizes)):
        order = np.log(errors[i-1] / errors[i]) / np.log(h_values[i-1] / h_values[i])
        assert order > 1.8, f"Convergence order too low: {order}"
    
    print("✓ test_convergence_order passed")


if __name__ == "__main__":
    print("Running Poisson solver tests...")
    print("-" * 60)
    
    test_poisson_solver_direct()
    test_poisson_solver_jacobi()
    test_poisson_solver_gauss_seidel()
    test_boundary_conditions()
    test_convergence_order()
    
    print("-" * 60)
    print("All Poisson solver tests passed!")
