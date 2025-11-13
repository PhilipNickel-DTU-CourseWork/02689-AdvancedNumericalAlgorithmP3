#!/usr/bin/env python3
"""Plot lid-driven cavity solution from FV solver.

This script loads and visualizes the solution computed by compute_LDC.py,
comparing with Ghia et al. benchmark data when available.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Add src to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from validation import GhiaBenchmark


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot lid-driven cavity solution"
    )
    parser.add_argument(
        "solution_file",
        type=str,
        nargs="?",
        default=None,
        help="Path to solution NPZ file (default: most recent in data/FV-Solver)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot file (default: figures/FV-Solver/<solution_name>.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively",
    )
    return parser.parse_args()


def extract_reynolds_number(solution_path: Path) -> float:
    """Extract Reynolds number from solution filename."""
    # Try to get from config file
    config_file = solution_path.parent / solution_path.name.replace("_solution.npz", "_config.parquet")
    if config_file.exists():
        df = pd.read_parquet(config_file)
        return float(df["Re"].iloc[0])

    # Fallback: parse from filename
    name = solution_path.stem
    if "Re" in name:
        try:
            re_str = name.split("Re")[1].split("_")[0]
            return float(re_str)
        except (IndexError, ValueError):
            pass

    return 100.0  # Default


def plot_solution(solution_path: Path, output_path: Path = None, show: bool = False):
    """Load and plot the solution."""
    print(f"Loading solution from {solution_path}")

    # Load solution data
    data = np.load(solution_path)
    u = data["u"]
    v = data["v"]
    p = data["p"]
    x = data["x"]
    y = data["y"]
    res_his = data["residual_history"]

    # Get Reynolds number
    Re = extract_reynolds_number(solution_path)
    print(f"Reynolds number: {Re}")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: u-velocity contour
    ax = axes[0, 0]
    levels = 20
    contour = ax.tricontourf(x, y, u, levels=levels, cmap='RdBu_r')
    fig.colorbar(contour, ax=ax, label='u-velocity')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'u-velocity (Re={int(Re)})')
    ax.set_aspect('equal')

    # Plot 2: v-velocity contour
    ax = axes[0, 1]
    contour = ax.tricontourf(x, y, v, levels=levels, cmap='RdBu_r')
    fig.colorbar(contour, ax=ax, label='v-velocity')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'v-velocity (Re={int(Re)})')
    ax.set_aspect('equal')

    # Plot 3: u-velocity along vertical centerline vs Ghia
    ax = axes[1, 0]

    # Extract values along vertical centerline (x ≈ 0.5)
    centerline_x = 0.5
    tol = 0.02
    mask = np.abs(x - centerline_x) < tol
    y_center = y[mask]
    u_center = u[mask]

    # Sort by y coordinate
    sort_idx = np.argsort(y_center)
    y_center_sorted = y_center[sort_idx]
    u_center_sorted = u_center[sort_idx]

    ax.plot(u_center_sorted, y_center_sorted, 'b-o', linewidth=2,
            markersize=4, label='FV Solver', markevery=max(1, len(y_center_sorted)//20))

    # Load Ghia benchmark data if available
    try:
        ghia_data = GhiaBenchmark.get_data(int(Re))
        ghia_u = ghia_data['u']
        ax.plot(ghia_u['U'], ghia_u['y'], 'r--', linewidth=2, label='Ghia et al. (1982)')
        print(f"Loaded Ghia benchmark data for Re={int(Re)}")
    except ValueError:
        print(f"No Ghia benchmark data available for Re={int(Re)}")

    ax.set_xlabel('u-velocity')
    ax.set_ylabel('y')
    ax.set_title('u-velocity along vertical centerline (x=0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: v-velocity along horizontal centerline vs Ghia
    ax = axes[1, 1]

    # Extract values along horizontal centerline (y ≈ 0.5)
    centerline_y = 0.5
    mask = np.abs(y - centerline_y) < tol
    x_center = x[mask]
    v_center = v[mask]

    # Sort by x coordinate
    sort_idx = np.argsort(x_center)
    x_center_sorted = x_center[sort_idx]
    v_center_sorted = v_center[sort_idx]

    ax.plot(x_center_sorted, v_center_sorted, 'b-o', linewidth=2,
            markersize=4, label='FV Solver', markevery=max(1, len(x_center_sorted)//20))

    # Load Ghia benchmark data if available
    try:
        ghia_data = GhiaBenchmark.get_data(int(Re))
        ghia_v = ghia_data['v']
        ax.plot(ghia_v['x'], ghia_v['V'], 'r--', linewidth=2, label='Ghia et al. (1982)')
    except ValueError:
        pass

    ax.set_xlabel('x')
    ax.set_ylabel('v-velocity')
    ax.set_title('v-velocity along horizontal centerline (y=0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    # Create convergence plot separately
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(res_his, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title(f'Convergence History (Re={int(Re)})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        conv_output = output_path.parent / (output_path.stem + "_convergence.png")
        plt.savefig(conv_output, dpi=150, bbox_inches='tight')
        print(f"Convergence plot saved to {conv_output}")

    if show:
        plt.show()
    else:
        plt.close()


def find_most_recent_solution():
    """Find the most recent solution file in data/FV-Solver."""
    data_dir = repo_root / "data" / "FV-Solver"
    if not data_dir.exists():
        return None

    solution_files = sorted(
        data_dir.glob("*_solution.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return solution_files[0] if solution_files else None


def main():
    """Main plotting routine."""
    args = parse_args()

    # Determine solution file
    if args.solution_file:
        solution_path = Path(args.solution_file)
    else:
        solution_path = find_most_recent_solution()
        if solution_path is None:
            print("Error: No solution files found in data/FV-Solver/")
            print("Run compute_LDC.py first to generate solution data")
            return
        print(f"Using most recent solution: {solution_path.name}")

    if not solution_path.exists():
        print(f"Error: Solution file not found: {solution_path}")
        return

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: save in figures directory
        figures_dir = repo_root / "figures" / "FV-Solver"
        figures_dir.mkdir(parents=True, exist_ok=True)
        output_path = figures_dir / (solution_path.stem.replace("_solution", "") + ".png")

    plot_solution(solution_path, output_path, args.show)

    print("\nPlotting completed successfully!")


if __name__ == "__main__":
    main()
