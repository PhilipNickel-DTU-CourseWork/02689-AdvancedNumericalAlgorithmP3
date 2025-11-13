#!/usr/bin/env python3
"""Compute lid-driven cavity solution using FV solver."""

# %% Imports
import numpy as np
import pyvista as pv
from pathlib import Path
from ldc import FVSolver

# %% Setup paths
data_dir = Path(__file__).parent.parent.parent / "data" / "FV-Solver"
data_dir.mkdir(parents=True, exist_ok=True)

# %% Solve
solver = FVSolver(Re=100.0)
results = solver.solve(tolerance=1e-5, max_iter=300)

print(f"Converged: {results.converged}")
print(f"Iterations: {results.iterations}")
print(f"Final residual: {results.final_alg_residual:.6e}")

# %% Save results as VTK
cell_centers = solver.get_cell_centers()

# Create PyVista point cloud from cell centers
points = np.column_stack([cell_centers[:, 0], cell_centers[:, 1], np.zeros(len(cell_centers))])
cloud = pv.PolyData(points)

# Add solution fields as point data
cloud['u'] = results.u
cloud['v'] = results.v
cloud['p'] = results.p
cloud['velocity_magnitude'] = np.sqrt(results.u**2 + results.v**2)

# Add metadata
cloud.field_data['Re'] = np.array([solver.solver_config.Re])
cloud.field_data['converged'] = np.array([results.converged])
cloud.field_data['iterations'] = np.array([results.iterations])
cloud.field_data['final_residual'] = np.array([results.final_alg_residual])

# Save as VTK file
output_file = data_dir / "LDC_Re100_solution.vtp"
cloud.save(output_file)

print(f"\nSolution saved to: {output_file}")
