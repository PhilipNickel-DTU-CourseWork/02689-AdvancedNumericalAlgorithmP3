# LDC Plotting Design

## Architecture

### 1. `LDCPlotter` - Field Visualization (single or multiple runs)
Handles plotting of simulation fields and convergence. Uses seaborn's `hue` parameter for multi-run comparisons.

### 2. `GhiaValidator` - Benchmark Validation (separate, focused class)
Dedicated class for Ghia et al. (1982) benchmark validation.

## Usage

### Single Run
```python
from utils import LDCPlotter, GhiaValidator

# Field plots
plotter = LDCPlotter({
    'data_path': "data/FV-Solver/LDC_Re100_data.parquet",
    'fields_path': "data/FV-Solver/LDC_Re100_fields.vtp"
})

plotter.plot_convergence(output_path="figures/convergence.pdf")
plotter.plot_velocity_fields(output_path="figures/velocity.pdf")
plotter.plot_pressure(output_path="figures/pressure.pdf")

# Ghia validation (separate)
validator = GhiaValidator(Re=100.0, fields_path="data/FV-Solver/LDC_Re100_fields.vtp")
validator.plot_validation(output_path="figures/ghia_validation.pdf")
```

**Strengths:**
- Clean separation: field plots vs. validation
- Simple single-run interface
- Automatic Re detection

## Multi-Run Comparison

### Use Cases
1. **Mesh convergence study**: Compare different mesh resolutions (nx=16, 32, 64, 128)
2. **Reynolds number sweep**: Compare different Re values
3. **Solver comparison**: Compare FV vs Spectral solvers
4. **Parameter study**: Compare different solver settings (alpha_uv, schemes, etc.)

### Example: Mesh Convergence Study

```python
from utils import LDCPlotter, GhiaValidator

# Define runs to compare
runs = [
    {
        'label': '32x32',
        'data_path': 'data/FV-Solver/LDC_Re100_nx32_data.parquet',
        'fields_path': 'data/FV-Solver/LDC_Re100_nx32_fields.vtp'
    },
    {
        'label': '64x64',
        'data_path': 'data/FV-Solver/LDC_Re100_nx64_data.parquet',
        'fields_path': 'data/FV-Solver/LDC_Re100_nx64_fields.vtp'
    },
    {
        'label': '128x128',
        'data_path': 'data/FV-Solver/LDC_Re100_nx128_data.parquet',
        'fields_path': 'data/FV-Solver/LDC_Re100_nx128_fields.vtp'
    }
]

# Create plotter with multiple runs
plotter = LDCPlotter(runs)

# Convergence comparison (automatic hue by run label)
plotter.plot_convergence(output_path="figures/convergence_comparison.pdf")

# Note: field plots (velocity, pressure) are only for single runs
# For multi-run field comparison, create separate plotters
```

**Key Features:**
- Same `LDCPlotter` class handles both single and multiple runs
- Uses seaborn's `hue` parameter for automatic styling
- Convergence plots automatically show legend when multiple runs
- Clean, consistent API

### Implementation Status

✅ **Implemented:**
- `LDCPlotter`: Single/multi-run convergence comparison using `hue`
- Automatic legend and styling for multiple runs
- Backward compatible with single-run interface

📋 **Future Extensions:**
- Multi-run centerline profile comparisons
- Grid convergence analysis (error vs mesh size)
- Multi-run Ghia validation overlay

## Naming Files for Multi-Run Studies

### Recommended naming convention
```
LDC_Re{Re}_nx{nx}_ny{ny}_{solver}_data.parquet
LDC_Re{Re}_nx{nx}_ny{ny}_{solver}_fields.vtp
```

Examples:
- `LDC_Re100_nx32_ny32_FV_data.parquet`
- `LDC_Re100_nx64_ny64_FV_fields.vtp`
- `LDC_Re1000_nx128_ny128_Spectral_data.parquet`
