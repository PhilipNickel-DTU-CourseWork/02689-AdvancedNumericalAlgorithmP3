#!/usr/bin/env python3
"""Plot lid-driven cavity solution."""

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
from utils import plotting

# %% Setup paths
data_dir = Path(__file__).parent.parent.parent / "data" / "FV-Solver"
fig_dir = Path(__file__).parent.parent.parent / "figures" / "FV-Solver"
fig_dir.mkdir(parents=True, exist_ok=True)

# %% Load results
mesh = pv.read(data_dir / "LDC_Re100_solution.vtp")
points = mesh.points
x, y = points[:, 0], points[:, 1]
u = mesh['u']
v = mesh['v']
p = mesh['p']
vel_mag = mesh['velocity_magnitude']
Re = float(mesh.field_data['Re'][0])

# %% Plot velocity field
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].tricontourf(x, y, u, levels=20, cmap='RdBu_r')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('U velocity')
axes[0].set_aspect('equal')

axes[1].tricontourf(x, y, v, levels=20, cmap='RdBu_r')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('V velocity')
axes[1].set_aspect('equal')

axes[2].tricontourf(x, y, vel_mag, levels=20, cmap='viridis')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('Velocity magnitude')
axes[2].set_aspect('equal')

plt.suptitle(f'Lid-Driven Cavity: Re = {Re:.0f}', fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / "LDC_Re100_velocity.pdf", bbox_inches='tight', dpi=300)
print(f"Velocity plot saved to: {fig_dir / 'LDC_Re100_velocity.pdf'}")

# %% Plot pressure field
fig, ax = plt.subplots(figsize=(8, 7))
cf = ax.tricontourf(x, y, p, levels=20, cmap='coolwarm')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Pressure field (Re = {Re:.0f})')
ax.set_aspect('equal')
plt.colorbar(cf, ax=ax, label='Pressure')
plt.tight_layout()
plt.savefig(fig_dir / "LDC_Re100_pressure.pdf", bbox_inches='tight', dpi=300)
print(f"Pressure plot saved to: {fig_dir / 'LDC_Re100_pressure.pdf'}")

plt.close('all')
