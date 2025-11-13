#!/usr/bin/env python3
"""Test script for the new FV solver architecture.

This script demonstrates the usage of the LidDrivenCavitySolver framework
with the FVSolver implementation.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ldc import FVSolver, SolverConfig, RuntimeConfig, FVConfig
from validation import GhiaBenchmark


def plot_results(solver: FVSolver, results, Re: float, save_path: str = None):
    """Plot velocity profiles and compare with Ghia benchmark."""
    # Get mesh coordinates
    cell_centers = solver.get_cell_centers()
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]

    # Get velocity fields
    u, v = solver.get_velocity_field()
    p = solver.get_pressure_field()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: u-velocity contour
    ax = axes[0, 0]
    scatter = ax.tricontourf(x, y, u, levels=20, cmap='RdBu_r')
    plt.colorbar(scatter, ax=ax, label='u-velocity')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'u-velocity contour (Re={Re})')
    ax.set_aspect('equal')

    # Plot 2: v-velocity contour
    ax = axes[0, 1]
    scatter = ax.tricontourf(x, y, v, levels=20, cmap='RdBu_r')
    plt.colorbar(scatter, ax=ax, label='v-velocity')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'v-velocity contour (Re={Re})')
    ax.set_aspect('equal')

    # Plot 3: u-velocity along vertical centerline
    ax = axes[1, 0]

    # Extract values along vertical centerline (x ≈ 0.5)
    centerline_x = 0.5
    tol = 0.02  # Tolerance for finding centerline
    mask = np.abs(x - centerline_x) < tol
    y_center = y[mask]
    u_center = u[mask]

    # Sort by y coordinate
    sort_idx = np.argsort(y_center)
    y_center = y_center[sort_idx]
    u_center = u_center[sort_idx]

    ax.plot(u_center, y_center, 'b-o', linewidth=2, markersize=4, label='FV Solver')

    # Load Ghia benchmark data if available
    try:
        ghia_data = GhiaBenchmark.get_data(int(Re))
        ghia_u = ghia_data['u']
        ax.plot(ghia_u['U'], ghia_u['y'], 'r--', linewidth=2, label='Ghia et al.')
    except ValueError:
        print(f"No Ghia benchmark data available for Re={Re}")

    ax.set_xlabel('u-velocity')
    ax.set_ylabel('y')
    ax.set_title('Velocity profile along vertical centerline')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Convergence history
    ax = axes[1, 1]
    ax.semilogy(results.res_his, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title('Convergence History')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    """Run FV solver test case."""
    print("=" * 60)
    print("Testing FV Solver Architecture")
    print("=" * 60)

    # Configure problem
    solver_config = SolverConfig(
        Re=100.0,
        lid_velocity=1.0,
        Lx=1.0,
        Ly=1.0,
    )

    runtime_config = RuntimeConfig(
        tolerance=1e-6,
        max_iter=1000,
    )

    fv_config = FVConfig(
        mesh_type="structured_uniform",
        nx=32,
        ny=32,
        convection_scheme="upwind",
        limiter="MUSCL",
        alpha_uv=0.7,
        alpha_p=0.3,
        use_piso=False,
        piso_corrections=2,
    )

    print(f"\nProblem Configuration:")
    print(f"  Reynolds number: {solver_config.Re}")
    print(f"  Lid velocity: {solver_config.lid_velocity}")
    print(f"  Domain: {solver_config.Lx} x {solver_config.Ly}")
    print(f"\nFV Configuration:")
    print(f"  Mesh resolution: {fv_config.nx} x {fv_config.ny}")
    print(f"  Convection scheme: {fv_config.convection_scheme}")
    print(f"  Under-relaxation (momentum): {fv_config.alpha_uv}")
    print(f"  Under-relaxation (pressure): {fv_config.alpha_p}")
    print(f"  Algorithm: {'PISO' if fv_config.use_piso else 'SIMPLE'}")
    print(f"\nRuntime Configuration:")
    print(f"  Tolerance: {runtime_config.tolerance}")
    print(f"  Max iterations: {runtime_config.max_iter}")
    print()

    # Create solver
    print("Initializing FV solver...")
    solver = FVSolver(solver_config, runtime_config, fv_config)

    # Run solver
    print("\nRunning solver...")
    results = solver.solve()

    # Print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  Converged: {results.converged}")
    print(f"  Iterations: {results.iterations}")
    print(f"  Final residual: {results.final_alg_residual:.6e}")
    print(f"  Wall time: {results.wall_time:.2f} s")
    print()

    # Visualize results
    print("Generating plots...")
    figures_dir = repo_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_path = figures_dir / f"fv_solver_Re{int(solver_config.Re)}.png"

    plot_results(solver, results, solver_config.Re, save_path=str(plot_path))

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
