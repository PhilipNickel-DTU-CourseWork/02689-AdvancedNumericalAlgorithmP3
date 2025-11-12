"""
Example: Solving 2D Poisson equation with known analytical solution.

This example solves:
    -∇²u = 2π² sin(πx) sin(πy)  in Ω = [0,1] × [0,1]
    u = 0                         on ∂Ω

Analytical solution: u(x,y) = sin(πx) sin(πy)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from algorithms import PoissonSolver, Mesh2D
from algorithms.utils import plot_solution, plot_solution_3d, compute_error


def main():
    # Problem parameters
    nx, ny = 51, 51  # Grid points
    
    # Create mesh
    mesh = Mesh2D(nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
    
    # Define source function: f(x,y) = 2π² sin(πx) sin(πy)
    def source_function(x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Define boundary conditions: u = 0 on all boundaries
    def boundary_function(x, y):
        return np.zeros_like(x)
    
    # Analytical solution
    def exact_solution(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Create solver and solve
    print("Solving 2D Poisson equation...")
    print(f"Grid size: {nx} × {ny}")
    
    solver = PoissonSolver(mesh)
    
    # Solve using direct method
    print("\nMethod: Direct sparse solver")
    u_direct = solver.solve(source_function, boundary_function, method='direct')
    
    # Compute exact solution
    u_exact = exact_solution(mesh.X, mesh.Y)
    
    # Compute errors
    errors = compute_error(u_direct, u_exact)
    print(f"L² error:         {errors['l2']:.6e}")
    print(f"L∞ error:         {errors['linf']:.6e}")
    print(f"Relative L² error: {errors['relative_l2']:.6e}")
    
    # Visualization
    print("\nGenerating plots...")
    
    # Plot numerical solution
    fig1, ax1 = plot_solution(mesh, u_direct, title="Numerical Solution")
    plt.savefig('poisson_numerical.png', dpi=150, bbox_inches='tight')
    print("Saved: poisson_numerical.png")
    
    # Plot exact solution
    fig2, ax2 = plot_solution(mesh, u_exact, title="Exact Solution")
    plt.savefig('poisson_exact.png', dpi=150, bbox_inches='tight')
    print("Saved: poisson_exact.png")
    
    # Plot error
    error_field = np.abs(u_direct - u_exact)
    fig3, ax3 = plot_solution(mesh, error_field, title="Absolute Error", cmap='hot')
    plt.savefig('poisson_error.png', dpi=150, bbox_inches='tight')
    print("Saved: poisson_error.png")
    
    # 3D plot of numerical solution
    fig4, ax4 = plot_solution_3d(mesh, u_direct, title="Numerical Solution (3D)")
    plt.savefig('poisson_3d.png', dpi=150, bbox_inches='tight')
    print("Saved: poisson_3d.png")
    
    print("\nComparison of iterative methods:")
    print("-" * 60)
    
    # Compare with iterative methods (on smaller grid for speed)
    nx_iter, ny_iter = 31, 31
    mesh_iter = Mesh2D(nx_iter, ny_iter)
    solver_iter = PoissonSolver(mesh_iter)
    
    # Jacobi
    print("\nJacobi iteration...")
    u_jacobi = solver_iter.solve(source_function, boundary_function, method='jacobi')
    u_exact_iter = exact_solution(mesh_iter.X, mesh_iter.Y)
    errors_jacobi = compute_error(u_jacobi, u_exact_iter)
    print(f"  L² error: {errors_jacobi['l2']:.6e}")
    
    # Gauss-Seidel
    print("\nGauss-Seidel iteration...")
    u_gs = solver_iter.solve(source_function, boundary_function, method='gauss_seidel')
    errors_gs = compute_error(u_gs, u_exact_iter)
    print(f"  L² error: {errors_gs['l2']:.6e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
