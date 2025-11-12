# Project 3 Documentation

## Overview

This project implements numerical methods for solving partial differential equations (PDEs) as part of the DTU course 02689 - Advanced Numerical Algorithms.

## Mathematical Background

### Poisson Equation

The 2D Poisson equation is:

```
-∇²u = f  in Ω
u = g     on ∂Ω
```

where:
- `u(x,y)` is the unknown solution
- `f(x,y)` is the source term
- `g(x,y)` are the Dirichlet boundary conditions
- `∇²` is the Laplacian operator: `∇² = ∂²/∂x² + ∂²/∂y²`

### Finite Difference Discretization

The Laplacian is discretized using a 5-point stencil:

```
∇²u[i,j] ≈ (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4u[i,j]) / h²
```

This leads to a system of linear equations:

```
Au = b
```

where:
- `A` is a sparse matrix (band diagonal structure)
- `u` is the vector of unknown values at grid points
- `b` is the right-hand side (source term and boundary conditions)

## Solution Methods

### 1. Direct Solver

Uses sparse matrix factorization (via SciPy's `spsolve`). This is the most accurate and efficient method for moderate-sized problems (< 100,000 unknowns).

**Advantages:**
- Exact solution (up to machine precision)
- Fast for moderate-sized problems
- No convergence issues

**Disadvantages:**
- Memory usage scales as O(n^1.5) for 2D problems
- Can be slow for very large problems

### 2. Jacobi Iteration

Simple iterative method where new values are computed from old values:

```
u[i,j]^(k+1) = (u[i-1,j]^k + u[i+1,j]^k + u[i,j-1]^k + u[i,j+1]^k + h²f[i,j]) / 4
```

**Advantages:**
- Simple to implement
- Easy to parallelize
- Low memory usage

**Disadvantages:**
- Slow convergence (especially for fine grids)
- Many iterations required

### 3. Gauss-Seidel Iteration

Improved iterative method that uses updated values as soon as available:

```
u[i,j]^(k+1) = (u[i-1,j]^(k+1) + u[i+1,j]^k + u[i,j-1]^(k+1) + u[i,j+1]^k + h²f[i,j]) / 4
```

**Advantages:**
- Faster convergence than Jacobi
- Low memory usage

**Disadvantages:**
- Still slow for fine grids
- Sequential nature limits parallelization

## Convergence Analysis

For smooth solutions, the finite difference method achieves second-order accuracy:

```
||u_exact - u_numerical|| = O(h²)
```

This is verified in the convergence study example, which shows convergence order ≈ 2.

## API Documentation

### Mesh2D Class

```python
mesh = Mesh2D(nx, ny, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
```

**Parameters:**
- `nx, ny`: Number of grid points in x and y directions
- `xmin, xmax, ymin, ymax`: Domain boundaries

**Attributes:**
- `X, Y`: Meshgrid arrays of coordinates
- `dx, dy`: Grid spacing
- `x, y`: 1D coordinate arrays

### PoissonSolver Class

```python
solver = PoissonSolver(mesh)
solution = solver.solve(source_function, boundary_function, method='direct')
```

**Parameters:**
- `mesh`: Mesh2D object
- `source_function`: Function `f(x, y)` defining source term
- `boundary_function`: Function `g(x, y)` defining boundary values
- `method`: Solution method ('direct', 'jacobi', 'gauss_seidel')

**Returns:**
- 2D array of solution values at grid points

## Examples

### Basic Usage

```python
from algorithms import PoissonSolver, Mesh2D
import numpy as np

# Create mesh
mesh = Mesh2D(51, 51)

# Define problem
def source_function(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def boundary_function(x, y):
    return np.zeros_like(x)

# Solve
solver = PoissonSolver(mesh)
u = solver.solve(source_function, boundary_function)
```

### Visualization

```python
from algorithms.utils import plot_solution, plot_solution_3d

# 2D contour plot
fig, ax = plot_solution(mesh, u, title="Solution")

# 3D surface plot
fig, ax = plot_solution_3d(mesh, u, title="Solution 3D")
```

### Error Analysis

```python
from algorithms.utils import compute_error

u_exact = np.sin(np.pi * mesh.X) * np.sin(np.pi * mesh.Y)
errors = compute_error(u, u_exact)

print(f"L² error: {errors['l2']:.6e}")
print(f"L∞ error: {errors['linf']:.6e}")
```

## Future Enhancements

### Finite Volume Method

Will implement a conservative discretization scheme suitable for:
- Conservation laws
- Discontinuous solutions
- Complex geometries

### Multigrid Method

Will implement multigrid acceleration to improve convergence:
- V-cycle and W-cycle schemes
- Restriction and prolongation operators
- Multiple levels of grids

Expected performance: O(n) complexity instead of O(n²) for iterative methods.

## References

1. LeVeque, R. J. (2007). Finite Difference Methods for Ordinary and Partial Differential Equations.
2. Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000). A Multigrid Tutorial.
3. LeVeque, R. J. (2002). Finite Volume Methods for Hyperbolic Problems.
