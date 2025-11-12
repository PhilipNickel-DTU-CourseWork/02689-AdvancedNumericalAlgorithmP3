"""
Utility functions for visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_solution(mesh, u, title="Solution", figsize=(10, 8), cmap='viridis'):
    """
    Plot the 2D solution as a contour plot.
    
    Parameters
    ----------
    mesh : Mesh2D
        Mesh object
    u : ndarray
        Solution array
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap name
    
    Returns
    -------
    Figure, Axes
        Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    contour = ax.contourf(mesh.X, mesh.Y, u, levels=20, cmap=cmap)
    ax.contour(mesh.X, mesh.Y, u, levels=10, colors='k', linewidths=0.5, alpha=0.3)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('u')
    
    plt.tight_layout()
    return fig, ax


def plot_solution_3d(mesh, u, title="Solution", figsize=(12, 9), cmap='viridis'):
    """
    Plot the 2D solution as a 3D surface plot.
    
    Parameters
    ----------
    mesh : Mesh2D
        Mesh object
    u : ndarray
        Solution array
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap name
    
    Returns
    -------
    Figure, Axes
        Matplotlib figure and axes objects
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(mesh.X, mesh.Y, u, cmap=cmap, 
                          linewidth=0, antialiased=True, alpha=0.9)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title(title)
    
    fig.colorbar(surf, ax=ax, shrink=0.5)
    
    plt.tight_layout()
    return fig, ax


def compute_error(u_numerical, u_exact):
    """
    Compute various error norms.
    
    Parameters
    ----------
    u_numerical : ndarray
        Numerical solution
    u_exact : ndarray
        Exact solution
    
    Returns
    -------
    dict
        Dictionary with error norms: 'l2', 'linf', 'relative_l2'
    """
    error = u_numerical - u_exact
    
    l2_error = np.sqrt(np.mean(error**2))
    linf_error = np.max(np.abs(error))
    relative_l2 = l2_error / np.sqrt(np.mean(u_exact**2)) if np.any(u_exact != 0) else l2_error
    
    return {
        'l2': l2_error,
        'linf': linf_error,
        'relative_l2': relative_l2
    }


def convergence_study(mesh_sizes, solver_func, exact_solution):
    """
    Perform a mesh convergence study.
    
    Parameters
    ----------
    mesh_sizes : list
        List of mesh sizes (nx values, assuming nx=ny)
    solver_func : callable
        Function that takes mesh size and returns numerical solution
    exact_solution : callable
        Function that takes mesh and returns exact solution
    
    Returns
    -------
    dict
        Dictionary with mesh sizes and errors
    """
    errors_l2 = []
    errors_linf = []
    h_values = []
    
    for n in mesh_sizes:
        u_num, mesh = solver_func(n)
        u_exact = exact_solution(mesh)
        
        error_dict = compute_error(u_num, u_exact)
        errors_l2.append(error_dict['l2'])
        errors_linf.append(error_dict['linf'])
        h_values.append(mesh.dx)
    
    return {
        'h': np.array(h_values),
        'l2_error': np.array(errors_l2),
        'linf_error': np.array(errors_linf)
    }


def plot_convergence(convergence_data, figsize=(10, 6)):
    """
    Plot convergence results.
    
    Parameters
    ----------
    convergence_data : dict
        Dictionary from convergence_study
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    Figure, Axes
        Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    h = convergence_data['h']
    l2_error = convergence_data['l2_error']
    linf_error = convergence_data['linf_error']
    
    ax.loglog(h, l2_error, 'o-', label='L² error', linewidth=2, markersize=8)
    ax.loglog(h, linf_error, 's-', label='L∞ error', linewidth=2, markersize=8)
    
    # Add reference lines for convergence order
    ax.loglog(h, h**2, '--', label='O(h²)', color='gray', alpha=0.5)
    
    ax.set_xlabel('Grid spacing h', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Convergence Study', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax
