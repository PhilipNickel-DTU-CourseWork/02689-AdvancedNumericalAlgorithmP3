#!/usr/bin/env python3
"""Compute lid-driven cavity solution using FV solver."""

# %% Imports
from ldc import FVSolver, FVConfig
from utils import get_project_root

# %% Setup paths
project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
data_dir.mkdir(parents=True, exist_ok=True)

# %% Configure and solve
config = FVConfig(Re=100.0)  # Single config with physics + numerics

solver = FVSolver(config)
results = solver.solve(tolerance=1e-5, max_iter=300)


print(f"Converged: {results.metadata['converged']}")
print(f"Iterations: {results.metadata['iterations']}")
print(f"Final residual: {results.metadata['final_residual']:.6e}")

# %% Save results
output_file = data_dir / "LDC_Re100.h5"
solver.save(output_file, results)

print(f"\nResults saved to: {output_file}")
