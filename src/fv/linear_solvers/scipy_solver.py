"""Scipy-based linear solver fallback for when PETSc is not available."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def scipy_solver(
    A_csr: csr_matrix,
    b_np: np.ndarray,
    ksp=None,
    tolerance=1e-6,
    max_iterations=1000,
    solver_type="direct",
    type=None,  # Alias for solver_type for compatibility
    preconditioner="hypre",
    remove_nullspace=False,
):
    """
    Solve A x = b using SciPy sparse direct solver (spsolve).

    This is a simplified version that uses only the direct solver for reliability.
    Works well for small to medium-sized problems (< 100k unknowns).

    Parameters
    ----------
    A_csr : csr_matrix
        Sparse matrix in CSR format.
    b_np : np.ndarray
        Right-hand side vector.
    ksp : None (ignored for compatibility)
        Included for API compatibility with petsc_solver.
    tolerance : float, optional
        Ignored (direct solver doesn't iterate).
    max_iterations : int, optional
        Ignored (direct solver doesn't iterate).
    solver_type : str, optional
        Ignored (always uses direct solver).
    type : str, optional
        Ignored (compatibility parameter).
    preconditioner : str, optional
        Ignored (compatibility parameter).
    remove_nullspace : bool, optional
        Whether to handle the nullspace (default: False).
        If True, constrains the solution to have zero at first DOF.

    Returns
    -------
    x_np : np.ndarray
        Solution vector x.
    residual_norm : float
        Final 2-norm of the residual: ||b - A x||_2.
    ksp : None
        Returns None for compatibility with PETSc interface.
    """
    n = A_csr.shape[0]

    # Handle nullspace if requested (pressure correction equation)
    if remove_nullspace:
        # Pin first value to zero to remove nullspace
        A_modified = A_csr.tolil()  # Convert to LIL for efficient row modification
        A_modified[0, :] = 0.0
        A_modified[0, 0] = 1.0
        A_work = A_modified.tocsr()

        b_modified = b_np.copy()
        b_modified[0] = 0.0
        b_work = b_modified
    else:
        A_work = A_csr
        b_work = b_np

    # Solve using direct solver (LU factorization)
    try:
        x_np = spsolve(A_work, b_work)
    except Exception as e:
        raise RuntimeError(f"Direct solver failed: {e}")

    # Ensure x_np is a 1D array (spsolve sometimes returns matrices)
    if hasattr(x_np, 'A1'):
        x_np = x_np.A1

    # Compute residual norm
    residual = b_work - A_work @ x_np
    residual_norm = np.linalg.norm(residual)

    return x_np, residual_norm, None


# Alias for compatibility
petsc_solver = scipy_solver
