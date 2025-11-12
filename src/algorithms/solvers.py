"""
Solvers for partial differential equations.
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from .mesh import Mesh2D


class PoissonSolver:
    """
    Solver for 2D Poisson equation using finite difference method.
    
    Solves: -∇²u = f in Ω
            u = g on ∂Ω (Dirichlet boundary conditions)
    
    Parameters
    ----------
    mesh : Mesh2D
        2D mesh object
    """
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.solution = None
    
    def solve(self, source_function, boundary_function, method='direct'):
        """
        Solve the Poisson equation.
        
        Parameters
        ----------
        source_function : callable
            Function f(x, y) defining the source term
        boundary_function : callable
            Function g(x, y) defining the boundary values
        method : str, optional
            Solution method: 'direct', 'jacobi', 'gauss_seidel' (default: 'direct')
        
        Returns
        -------
        ndarray
            Solution array u(x, y)
        """
        if method == 'direct':
            return self._solve_direct(source_function, boundary_function)
        elif method == 'jacobi':
            return self._solve_jacobi(source_function, boundary_function)
        elif method == 'gauss_seidel':
            return self._solve_gauss_seidel(source_function, boundary_function)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _solve_direct(self, source_function, boundary_function):
        """
        Solve using direct sparse matrix solver.
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        N = nx * ny
        
        # Build sparse matrix for finite difference discretization
        A = lil_matrix((N, N))
        b = np.zeros(N)
        
        dx2 = self.mesh.dx ** 2
        dy2 = self.mesh.dy ** 2
        
        for i in range(ny):
            for j in range(nx):
                idx = i * nx + j
                x = self.mesh.x[j]
                y = self.mesh.y[i]
                
                # Check if on boundary
                if i == 0 or i == ny - 1 or j == 0 or j == nx - 1:
                    # Dirichlet boundary condition
                    A[idx, idx] = 1.0
                    b[idx] = boundary_function(x, y)
                else:
                    # Interior point: 5-point stencil for -∇²u = f
                    A[idx, idx] = 2.0 / dx2 + 2.0 / dy2
                    A[idx, idx - 1] = -1.0 / dx2  # Left
                    A[idx, idx + 1] = -1.0 / dx2  # Right
                    A[idx, idx - nx] = -1.0 / dy2  # Bottom
                    A[idx, idx + nx] = -1.0 / dy2  # Top
                    b[idx] = source_function(x, y)
        
        # Solve sparse linear system
        u_flat = spsolve(A.tocsr(), b)
        self.solution = u_flat.reshape((ny, nx))
        
        return self.solution
    
    def _solve_jacobi(self, source_function, boundary_function, 
                      max_iter=10000, tol=1e-6):
        """
        Solve using Jacobi iteration.
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        u = np.zeros((ny, nx))
        u_new = np.zeros((ny, nx))
        
        # Set boundary conditions
        u[0, :] = boundary_function(self.mesh.X[0, :], self.mesh.Y[0, :])
        u[-1, :] = boundary_function(self.mesh.X[-1, :], self.mesh.Y[-1, :])
        u[:, 0] = boundary_function(self.mesh.X[:, 0], self.mesh.Y[:, 0])
        u[:, -1] = boundary_function(self.mesh.X[:, -1], self.mesh.Y[:, -1])
        
        # Evaluate source term
        f = source_function(self.mesh.X, self.mesh.Y)
        
        dx2 = self.mesh.dx ** 2
        dy2 = self.mesh.dy ** 2
        denom = 2.0 / dx2 + 2.0 / dy2
        
        # Iterative solution
        for iteration in range(max_iter):
            u_new[:, :] = u[:, :]
            
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    u_new[i, j] = (
                        (u[i, j-1] + u[i, j+1]) / dx2 +
                        (u[i-1, j] + u[i+1, j]) / dy2 +
                        f[i, j]
                    ) / denom
            
            # Check convergence
            error = np.max(np.abs(u_new - u))
            u[:, :] = u_new[:, :]
            
            if error < tol:
                break
        
        self.solution = u
        return self.solution
    
    def _solve_gauss_seidel(self, source_function, boundary_function,
                           max_iter=10000, tol=1e-6):
        """
        Solve using Gauss-Seidel iteration.
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        u = np.zeros((ny, nx))
        
        # Set boundary conditions
        u[0, :] = boundary_function(self.mesh.X[0, :], self.mesh.Y[0, :])
        u[-1, :] = boundary_function(self.mesh.X[-1, :], self.mesh.Y[-1, :])
        u[:, 0] = boundary_function(self.mesh.X[:, 0], self.mesh.Y[:, 0])
        u[:, -1] = boundary_function(self.mesh.X[:, -1], self.mesh.Y[:, -1])
        
        # Evaluate source term
        f = source_function(self.mesh.X, self.mesh.Y)
        
        dx2 = self.mesh.dx ** 2
        dy2 = self.mesh.dy ** 2
        denom = 2.0 / dx2 + 2.0 / dy2
        
        # Iterative solution
        for iteration in range(max_iter):
            u_old = u.copy()
            
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    u[i, j] = (
                        (u[i, j-1] + u[i, j+1]) / dx2 +
                        (u[i-1, j] + u[i+1, j]) / dy2 +
                        f[i, j]
                    ) / denom
            
            # Check convergence
            error = np.max(np.abs(u - u_old))
            
            if error < tol:
                break
        
        self.solution = u
        return self.solution
