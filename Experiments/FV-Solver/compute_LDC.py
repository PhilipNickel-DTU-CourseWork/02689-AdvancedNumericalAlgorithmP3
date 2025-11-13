#!/usr/bin/env python3
"""Compute lid-driven cavity solution using FV solver."""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from utils import plotting
from ldc import FVSolver

# %% Setup and solve
solver = FVSolver(Re=100.0)  # Uses default fine mesh and TVD scheme

results = solver.solve(tolerance=1e-5, max_iter=300)

print(f"Converged: {results.converged}")
print(f"Iterations: {results.iterations}")
print(f"Final residual: {results.final_alg_residual:.6e}")

# %% Visualize
cell_centers = solver.get_cell_centers()
x, y = cell_centers[:, 0], cell_centers[:, 1]
vel_mag = np.sqrt(results.u**2 + results.v**2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].tricontourf(x, y, results.u, levels=20, cmap='RdBu_r')
axes[0].set_title('U velocity')
axes[0].set_aspect('equal')

axes[1].tricontourf(x, y, results.v, levels=20, cmap='RdBu_r')
axes[1].set_title('V velocity')
axes[1].set_aspect('equal')

axes[2].tricontourf(x, y, vel_mag, levels=20, cmap='viridis')
axes[2].set_title('Velocity magnitude')
axes[2].set_aspect('equal')

plt.suptitle(f'Lid-Driven Cavity: Re = {solver.solver_config.Re}', fontweight='bold')
plt.tight_layout()
plt.show()
