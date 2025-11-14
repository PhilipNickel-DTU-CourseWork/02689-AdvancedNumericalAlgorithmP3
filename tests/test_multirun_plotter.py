#!/usr/bin/env python3
"""Test multi-run plotter functionality."""

from pathlib import Path
from utils import get_project_root, LDCPlotter

# Setup paths
project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
fig_dir = project_root / "figures" / "test"
fig_dir.mkdir(parents=True, exist_ok=True)

# Test multi-run with the same data (simulating mesh convergence study)
# In reality, you'd have different mesh resolutions
runs = [
    {
        'label': 'Run 1',
        'data_path': data_dir / "LDC_Re100_data.parquet",
    },
    {
        'label': 'Run 2',
        'data_path': data_dir / "LDC_Re100_data.parquet",
    },
]

plotter = LDCPlotter(runs)

# Test convergence comparison
plotter.plot_convergence(output_path=fig_dir / "test_multirun_convergence.pdf")

print("Multi-run test completed successfully!")
print(f"Output: {fig_dir / 'test_multirun_convergence.pdf'}")
