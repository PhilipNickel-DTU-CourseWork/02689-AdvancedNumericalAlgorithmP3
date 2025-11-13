#!/usr/bin/env python3
"""Compute lid-driven cavity solution using FV solver."""

# %% Imports
import numpy as np
import pandas as pd
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

# %% Save spatial fields as VTK
cell_centers = solver.get_cell_centers()

points = np.column_stack([cell_centers[:, 0], cell_centers[:, 1], np.zeros(len(cell_centers))])
cloud = pv.PolyData(points)

cloud['u'] = results.u
cloud['v'] = results.v
cloud['p'] = results.p
cloud['velocity_magnitude'] = np.sqrt(results.u**2 + results.v**2)

fields_file = data_dir / "LDC_Re100_fields.vtp"
cloud.save(fields_file)
print(f"\nFields saved to: {fields_file}")

# %% Save metadata and time-series as Parquet
# Create MultiIndex DataFrame with config, results, and residuals
data_frames = []

# Config
config_df = pd.DataFrame({
    'Re': [solver.solver_config.Re],
    'lid_velocity': [solver.solver_config.lid_velocity],
    'Lx': [solver.solver_config.Lx],
    'Ly': [solver.solver_config.Ly],
    'convection_scheme': [solver.fv_config.convection_scheme],
    'limiter': [solver.fv_config.limiter],
    'alpha_uv': [solver.fv_config.alpha_uv],
    'alpha_p': [solver.fv_config.alpha_p],
})
config_df['data_type'] = 'config'
config_df['index'] = 0

# Results
results_df = pd.DataFrame({
    'converged': [results.converged],
    'iterations': [results.iterations],
    'final_residual': [results.final_alg_residual],
    'wall_time': [results.wall_time],
})
results_df['data_type'] = 'results'
results_df['index'] = 0

# Residuals (time-series)
residuals_df = pd.DataFrame({
    'iteration': range(len(results.res_his)),
    'residual': results.res_his,
})
residuals_df['data_type'] = 'residuals'
residuals_df = residuals_df.rename(columns={'iteration': 'index'})

# Combine and set MultiIndex
all_data = pd.concat([config_df, results_df, residuals_df], ignore_index=True)
all_data = all_data.set_index(['data_type', 'index'])

data_file = data_dir / "LDC_Re100_data.parquet"
all_data.to_parquet(data_file)
print(f"Data saved to: {data_file}")
