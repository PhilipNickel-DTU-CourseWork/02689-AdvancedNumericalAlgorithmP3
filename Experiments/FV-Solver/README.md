# Finite Volume Solver Experiments

This directory contains scripts for running and visualizing lid-driven cavity simulations using the finite volume solver.

## Scripts

### `compute_LDC.py`
Runs the FV solver for lid-driven cavity problem and saves results.

**Usage:**
```bash
# Basic run with defaults (Re=100, 64x64 grid)
python compute_LDC.py

# Custom Reynolds number and resolution
python compute_LDC.py --Re 1000 --nx 128 --ny 128

# Use PISO algorithm instead of SIMPLE
python compute_LDC.py --piso

# Change convection scheme
python compute_LDC.py --scheme TVD

# All options
python compute_LDC.py --Re 400 --nx 64 --ny 64 --max-iter 2000 --tolerance 1e-7 --scheme upwind --output my_run
```

**Options:**
- `--Re`: Reynolds number (default: 100)
- `--nx`, `--ny`: Grid resolution (default: 64x64)
- `--max-iter`: Maximum iterations (default: 1000)
- `--tolerance`: Convergence tolerance (default: 1e-6)
- `--scheme`: Convection scheme - `upwind`, `TVD`, or `central` (default: upwind)
- `--piso`: Use PISO algorithm instead of SIMPLE
- `--output`: Custom output filename prefix

**Outputs:**
Results are saved to `data/FV-Solver/`:
- `<name>_config.parquet`: Configuration parameters
- `<name>_results.parquet`: Convergence and timing results
- `<name>_solution.npz`: Solution fields (u, v, p, x, y, residual history)

### `plot_LDC.py`
Visualizes FV solver results and compares with Ghia benchmark data.

**Usage:**
```bash
# Plot most recent solution
python plot_LDC.py

# Plot specific solution file
python plot_LDC.py data/FV-Solver/LDC_Re100_nx64_ny64_upwind_solution.npz

# Show plot interactively
python plot_LDC.py --show

# Custom output location
python plot_LDC.py --output my_plot.png
```

**Outputs:**
Plots are saved to `figures/FV-Solver/`:
- Main plot: velocity contours and centerline profiles vs Ghia benchmark
- Convergence plot: residual history

### `test_fv_solver.py`
Complete test script demonstrating the solver framework (runs and plots in one go).

**Usage:**
```bash
python test_fv_solver.py
```

## Examples

### Run a quick test
```bash
# Small grid for quick testing
python compute_LDC.py --Re 100 --nx 32 --ny 32
python plot_LDC.py
```

### Benchmark against Ghia et al.
```bash
# Run for Re=100, 400, 1000 (Ghia benchmark cases)
python compute_LDC.py --Re 100 --nx 128 --ny 128
python compute_LDC.py --Re 400 --nx 128 --ny 128
python compute_LDC.py --Re 1000 --nx 128 --ny 128

# Plot results (automatically compares with Ghia data)
python plot_LDC.py
```

### Compare SIMPLE vs PISO
```bash
python compute_LDC.py --Re 100 --output Re100_SIMPLE
python compute_LDC.py --Re 100 --piso --output Re100_PISO

python plot_LDC.py data/FV-Solver/Re100_SIMPLE_solution.npz --output figures/Re100_SIMPLE.png
python plot_LDC.py data/FV-Solver/Re100_PISO_solution.npz --output figures/Re100_PISO.png
```

## Architecture

The scripts use the clean solver framework defined in `src/ldc/`:
- `LidDrivenCavitySolver`: Abstract base class
- `FVSolver`: Finite volume implementation (SIMPLE/PISO)
- `SolverConfig`, `RuntimeConfig`, `FVConfig`: Configuration dataclasses
- `Results`: Solution data container

The FV solver wraps the existing collocated FV implementation in `src/fv/` with automatic mesh generation and boundary condition setup.
