"""
Example: Mesh convergence study for the Poisson solver.

Demonstrates how solution accuracy improves with mesh refinement.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from algorithms import PoissonSolver, Mesh2D
from algorithms.utils import convergence_study, plot_convergence, compute_error


def main():
    # Define problem functions
    def source_function(x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def boundary_function(x, y):
        return np.zeros_like(x)
    
    def exact_solution_func(mesh):
        return np.sin(np.pi * mesh.X) * np.sin(np.pi * mesh.Y)
    
    # Solver function for convergence study
    def solver_func(n):
        mesh = Mesh2D(n, n)
        solver = PoissonSolver(mesh)
        u = solver.solve(source_function, boundary_function, method='direct')
        return u, mesh
    
    # Perform convergence study
    print("Performing mesh convergence study...")
    mesh_sizes = [11, 21, 41, 81, 161]
    
    results = convergence_study(mesh_sizes, solver_func, exact_solution_func)
    
    # Display results
    print("\nConvergence Results:")
    print("-" * 60)
    print(f"{'h':>12} {'L² error':>15} {'L∞ error':>15} {'Order':>10}")
    print("-" * 60)
    
    for i, (h, l2_err, linf_err) in enumerate(zip(results['h'], 
                                                    results['l2_error'],
                                                    results['linf_error'])):
        if i > 0:
            # Compute convergence order
            order = np.log(results['l2_error'][i-1] / l2_err) / np.log(results['h'][i-1] / h)
            print(f"{h:12.6f} {l2_err:15.6e} {linf_err:15.6e} {order:10.2f}")
        else:
            print(f"{h:12.6f} {l2_err:15.6e} {linf_err:15.6e} {'--':>10}")
    
    # Plot convergence
    print("\nGenerating convergence plot...")
    fig, ax = plot_convergence(results)
    plt.savefig('convergence_study.png', dpi=150, bbox_inches='tight')
    print("Saved: convergence_study.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
