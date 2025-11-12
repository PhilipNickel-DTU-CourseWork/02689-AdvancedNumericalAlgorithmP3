# Quick Start Guide

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PhilipNickel-DTU-CourseWork/02689-AdvancedNumericalAlgorithmP3.git
cd 02689-AdvancedNumericalAlgorithmP3
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Examples

### Example 1: Basic Poisson Solver

```bash
cd examples
python poisson_example.py
```

This will:
- Solve the 2D Poisson equation with analytical solution
- Generate plots: `poisson_numerical.png`, `poisson_exact.png`, `poisson_error.png`, `poisson_3d.png`
- Compare different solution methods (direct, Jacobi, Gauss-Seidel)

Expected output:
```
Solving 2D Poisson equation...
Grid size: 51 × 51

Method: Direct sparse solver
L² error:         1.612999e-04
L∞ error:         3.290518e-04
Relative L² error: 3.290518e-04
```

### Example 2: Convergence Study

```bash
cd examples
python convergence_study.py
```

This will:
- Test the solver with progressively finer meshes
- Verify second-order convergence
- Generate convergence plot: `convergence_study.png`

Expected output:
```
Convergence Results:
------------------------------------------------------------
           h        L² error        L∞ error      Order
------------------------------------------------------------
    0.100000    3.757008e-03    8.265417e-03         --
    0.050000    9.803366e-04    2.058707e-03       1.94
    0.025000    2.508295e-04    5.142005e-04       1.97
    0.012500    6.346686e-05    1.285204e-04       1.98
    0.006250    1.596434e-05    3.212824e-05       1.99
```

## Running Tests

### All tests:
```bash
python tests/test_mesh.py
python tests/test_solvers.py
```

### Individual test:
```bash
cd tests
python test_mesh.py
```

Expected output:
```
Running mesh tests...
------------------------------------------------------------
✓ test_mesh_creation passed
✓ test_mesh_grids passed
✓ test_interior_points passed
✓ test_boundary_mask passed
✓ test_custom_domain passed
------------------------------------------------------------
All mesh tests passed!
```

## Creating Your Own Solver

Here's a minimal example:

```python
# my_solver.py
import sys
sys.path.insert(0, '../src')

import numpy as np
from algorithms import PoissonSolver, Mesh2D
from algorithms.utils import plot_solution

# Create mesh (51x51 grid on [0,1]x[0,1])
mesh = Mesh2D(51, 51)

# Define your problem
def my_source(x, y):
    """Source term f(x,y)"""
    return -2.0 * np.ones_like(x)  # Constant source

def my_boundary(x, y):
    """Boundary values g(x,y)"""
    return np.zeros_like(x)  # Zero on boundary

# Solve
solver = PoissonSolver(mesh)
u = solver.solve(my_source, my_boundary, method='direct')

# Visualize
plot_solution(mesh, u, title="My Solution")
```

## Customization

### Changing Grid Size

For a finer grid:
```python
mesh = Mesh2D(101, 101)  # 101x101 grid
```

For a coarser grid (faster computation):
```python
mesh = Mesh2D(21, 21)  # 21x21 grid
```

### Changing Domain

For a different domain, e.g., [-1, 2] × [-1, 1]:
```python
mesh = Mesh2D(51, 51, xmin=-1.0, xmax=2.0, ymin=-1.0, ymax=1.0)
```

### Changing Solution Method

Direct solver (default, most accurate):
```python
u = solver.solve(source, boundary, method='direct')
```

Jacobi iteration (slower, but works for very large problems):
```python
u = solver.solve(source, boundary, method='jacobi')
```

Gauss-Seidel iteration (faster than Jacobi):
```python
u = solver.solve(source, boundary, method='gauss_seidel')
```

## Tips

1. **Start with small grids** (21×21 or 31×31) for testing
2. **Use direct solver** for production runs (most accurate)
3. **Check convergence** with progressively finer grids
4. **Visualize results** to verify correctness

## Troubleshooting

### Import errors
Make sure you're running from the correct directory and that `src/` is in your Python path:
```python
import sys
sys.path.insert(0, '../src')  # Adjust path as needed
```

### Slow convergence with iterative methods
- Try using the direct solver instead
- Increase the grid size gradually
- Check the tolerance in `config.ini`

### Memory errors
- Use a coarser grid
- Use iterative methods instead of direct solver

## Next Steps

1. Explore the test files for more examples
2. Read `docs/project_overview.md` for mathematical background
3. Modify examples to solve your own problems
4. Contribute to Issue #2 (Finite Volume) or Issue #3 (Multigrid)
