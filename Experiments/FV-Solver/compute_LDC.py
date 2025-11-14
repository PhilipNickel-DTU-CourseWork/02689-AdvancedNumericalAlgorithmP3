#!/usr/bin/env python3
"""Compute lid-driven cavity solution using FV solver."""

# %% Imports
from ldc import FVSolver
from utils import get_project_root

# %% Setup paths
project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
data_dir.mkdir(parents=True, exist_ok=True)

# %% Solve
solver = FVSolver(Re=100.0)
results = solver.solve(tolerance=1e-5, max_iter=300)

print(f"Converged: {results.convergence.converged}")
print(f"Iterations: {results.convergence.iterations}")
print(f"Final residual: {results.convergence.final_alg_residual:.6e}")

# %% Save results
fields_file = data_dir / "LDC_Re100_fields.vtp"
data_file = data_dir / "LDC_Re100_data.parquet"

solver.save_fields(fields_file, results)
solver.save_data(data_file, results)

print(f"\nFields saved to: {fields_file}")
print(f"Data saved to: {data_file}")
