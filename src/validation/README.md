# Validation Module

This module contains benchmark data for validating numerical solvers.

## Ghia Benchmark Data

The `GhiaBenchmark` class provides reference data from Ghia et al. (1982) for lid-driven cavity flow validation.

### Available Reynolds Numbers

- 100
- 400
- 1000
- 3200
- 5000
- 7500
- 10000

### Usage

```python
from validation import GhiaBenchmark

# Get benchmark data for Re=1000
ghia_data = GhiaBenchmark.get_data(1000)

# The data contains velocity profiles along centerlines:
# ghia_data['x'] - x-coordinates along horizontal centerline
# ghia_data['y'] - y-coordinates along vertical centerline
# ghia_data['u'] - u-velocity along vertical centerline (at x=0.5)
# ghia_data['v'] - v-velocity along horizontal centerline (at y=0.5)

# Check available Reynolds numbers
available_re = GhiaBenchmark.available_reynolds_numbers()
print(f"Available Re: {available_re}")

# Find closest available Reynolds number
closest_re = GhiaBenchmark.get_closest_reynolds(1234)
print(f"Closest Re to 1234: {closest_re}")
```

### Reference

Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of computational physics*, 48(3), 387-411.
