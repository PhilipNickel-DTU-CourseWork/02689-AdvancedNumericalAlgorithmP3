#!/usr/bin/env python3
"""Visualize all meshes in data/meshes/."""

# %% Imports
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio
from utils import plotting  # Apply proper styling

# %% Setup directories
data_dir = repo_root / "data" / "meshes"
figures_dir = repo_root / "figures" / "meshes"
figures_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Visualizing meshes...")
print("=" * 60)

# %% Plot structured meshes
structured_dir = data_dir / "structured"
mesh_files = sorted(structured_dir.glob("*.msh"))

print(f"\nStructured meshes ({len(mesh_files)}):")
for mesh_file in mesh_files:
    mesh = meshio.read(mesh_file)
    points = mesh.points[:, :2]
    cells = mesh.cells_dict["quad"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot quad edges
    for quad in cells:
        quad_points = points[quad]
        quad_closed = list(quad_points) + [quad_points[0]]
        xs, ys = zip(*quad_closed)
        ax.plot(xs, ys, 'k-', linewidth=0.5)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'structured/{mesh_file.stem}\n{len(cells)} quadrilateral cells, {len(points)} nodes')
    ax.grid(True, alpha=0.3)

    output_file = figures_dir / f"structured_{mesh_file.stem}.pdf"
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  ✓ {output_file.name}")

# %% Plot unstructured meshes
unstructured_dir = data_dir / "unstructured"
mesh_files = sorted(unstructured_dir.glob("*.msh"))

print(f"\nUnstructured meshes ({len(mesh_files)}):")
for mesh_file in mesh_files:
    mesh = meshio.read(mesh_file)
    points = mesh.points[:, :2]
    cells = mesh.cells_dict["triangle"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot triangulation
    triang = tri.Triangulation(points[:, 0], points[:, 1], cells)
    ax.triplot(triang, 'k-', linewidth=0.5)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'unstructured/{mesh_file.stem}\n{len(cells)} triangular cells, {len(points)} nodes')
    ax.grid(True, alpha=0.3)

    output_file = figures_dir / f"unstructured_{mesh_file.stem}.pdf"
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  ✓ {output_file.name}")

# %% Done
print("\n" + "=" * 60)
print(f"Figures saved to: {figures_dir}")
print("=" * 60)
