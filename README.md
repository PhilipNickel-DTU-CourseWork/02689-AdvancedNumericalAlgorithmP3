# Advanced Numerical Algorithms - Project 3

This repository contains implementations of numerical methods for solving partial differential equations (PDEs) as part of the DTU course 02689 - Advanced Numerical Algorithms.

## Project Overview

This project focuses on developing efficient numerical solvers for PDEs, including:

- **Finite Difference Methods**: Basic discretization techniques for PDEs
- **Finite Volume Methods**: Conservative discretization schemes
- **Multigrid Methods**: Acceleration techniques for iterative solvers

## Project Structure

```
02689-AdvancedNumericalAlgorithmP3/
├── src/
│   └── algorithms/
│       ├── __init__.py       # Package initialization
│       ├── mesh.py           # Mesh generation and handling
│       ├── solvers.py        # PDE solvers (FD, FV)
│       └── utils.py          # Visualization and analysis tools
├── examples/
│   ├── poisson_example.py    # Basic Poisson equation example
│   └── convergence_study.py  # Mesh convergence analysis
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

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

## Usage

### Basic Example

Solve the 2D Poisson equation with a known analytical solution:

```bash
cd examples
python poisson_example.py
```

This will generate several plots showing the numerical solution, exact solution, error distribution, and a 3D visualization.

### Convergence Study

Perform a mesh refinement study to verify convergence order:

```bash
cd examples
python convergence_study.py
```

### Using the Library

```python
from algorithms import PoissonSolver, Mesh2D
import numpy as np

# Create mesh
mesh = Mesh2D(nx=51, ny=51, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

# Define problem
def source_function(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def boundary_function(x, y):
    return np.zeros_like(x)

# Solve
solver = PoissonSolver(mesh)
solution = solver.solve(source_function, boundary_function, method='direct')
```

## Methods

### Finite Difference Method

The Poisson solver implements the standard 5-point stencil for the Laplacian operator:

```
∇²u ≈ (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4u[i,j]) / h²
```

### Solution Methods

- **Direct solver**: Uses sparse matrix factorization (recommended for moderate-sized problems)
- **Jacobi iteration**: Simple iterative method (good for learning, slow convergence)
- **Gauss-Seidel iteration**: Improved iterative method (faster than Jacobi)

## Future Work

See open issues for planned enhancements:

- Issue #2: Finite Volume Solver implementation
- Issue #3: Multigrid acceleration

## Requirements

- Python 3.7+
- NumPy ≥ 1.21.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.4.0

## License

See LICENSE file for details.

## Course Information

- **Course**: 02689 Advanced Numerical Algorithms
- **Institution**: DTU (Technical University of Denmark)
- **Project**: 3