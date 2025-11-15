import os
import numpy as np
import numpy.linalg as la
from scipy.sparse import coo_matrix
from fv.assembly.convection_diffusion_matrix import assemble_diffusion_convection_matrix
from fv.discretization.gradient.leastSquares import compute_cell_gradients
from fv.linear_solvers.scipy_solver import scipy_solver
from fv.assembly.rhie_chow import mdot_calculation, rhie_chow_velocity
from fv.assembly.pressure_correction_eq_assembly import assemble_pressure_correction_matrix, pressure_correction_loop_term
from fv.assembly.divergence import compute_divergence_from_face_fluxes
from fv.core.corrections import velocity_correction
import matplotlib.pyplot as plt
from fv.core.helpers import bold_Dv_calculation, interpolate_to_face, compute_residual, relax_momentum_equation, apply_mean_zero_constraint, set_pressure_boundaries, compute_l2_norm, get_unique_cells_from_faces
import time
from numba import njit

# Set up persistent Numba cache for faster subsequent runs
os.environ['NUMBA_CACHE_DIR'] = os.path.expanduser('~/.numba_cache')
os.makedirs(os.environ['NUMBA_CACHE_DIR'], exist_ok=True)




@njit(cache=True)
def enforce_boundary_conditions(mesh, u_field):
    boundary_faces = mesh.boundary_faces
    n_boundary = boundary_faces.shape[0]
    for i in range(n_boundary):
        f = boundary_faces[i]
        owner_cell = mesh.owner_cells[f]
        u_field[owner_cell, 0] = mesh.boundary_values[f, 0]
        u_field[owner_cell, 1] = mesh.boundary_values[f, 1]
    return u_field

def is_diagonally_dominant(A):
    # Convert sparse matrix to dense array if needed
    if hasattr(A, 'toarray'):
        A = A.toarray()
    else:
        A = np.asarray(A)
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    abs_A = np.abs(A)
    diagonal = np.diag(abs_A)
    off_diagonal_sum = np.sum(abs_A, axis=1) - diagonal
    dominance = np.all(diagonal >= off_diagonal_sum)
    return dominance

def simple_algorithm(mesh, config, rho, mu, max_iter, tol):
    # Convert tolerance from string to float if needed
    tol = float(tol)

    # Linear solver settings
    linear_solver_settings = {
        'momentum': {'type': 'bcgs', 'preconditioner': 'hypre', 'tolerance': 1e-6, 'max_iterations': 1000},
        'pressure': {'type': 'bcgs', 'preconditioner': 'hypre', 'tolerance': 1e-6, 'max_iterations': 1000}
    }

    time_start = time.time()

    # cells and faces
    n_cells = mesh.cell_volumes.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    n_faces = n_internal + n_boundary

    # Mass fluxes
    mdot = np.ascontiguousarray(np.zeros(n_faces))
    mdot_star = np.ascontiguousarray(np.zeros(n_faces))

    # Velocity fields
    U = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_prev_iter = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_star = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_old_time = np.ascontiguousarray(np.zeros((n_cells, 2)))
    U_old_faces = np.ascontiguousarray(np.zeros((n_faces, 2)))

    # Pressure field
    p = np.ascontiguousarray(np.zeros(n_cells))

    # Residuals (build as lists instead of pre-allocated arrays)
    u_l2norm = []
    v_l2norm = []
    continuity_l2norm = []
    max_u_l2norm = 0.0
    max_v_l2norm = 0.0
    max_mass_imbalance = 0.0
    
    # Initialize residual fields
    u_residual_field = np.zeros(n_cells)
    v_residual_field = np.zeros(n_cells)
    continuity_field = np.zeros(n_cells)
    
    internal_cells = get_unique_cells_from_faces(mesh, mesh.internal_faces)

    final_iter_count = 0
    is_converged = False

    for i in range(max_iter):
        final_iter_count = i + 1

        #=============================================================================
        # PRECOMPUTE QUANTITIES
        #=============================================================================
        grad_p = compute_cell_gradients(mesh, p, weighted=True, weight_exponent=0.5, use_limiter=False)
        grad_p_bar = interpolate_to_face(mesh, grad_p)
        U_old_bar = interpolate_to_face(mesh, U)
        grad_u = compute_cell_gradients(mesh, U[:,0], weighted=True, weight_exponent=0.5, use_limiter=True)
        grad_v = compute_cell_gradients(mesh, U[:,1], weighted=True, weight_exponent=0.5, use_limiter=True)

        #=============================================================================
        # ASSEMBLE and solve U-MOMENTUM EQUATIONS
        #=============================================================================
        row, col, data, b_u = assemble_diffusion_convection_matrix(
            mesh, mdot, grad_u, U_prev_iter, rho, mu, 0, phi=U[:,0],
            scheme=config.convection_scheme, limiter=config.limiter, pressure_field=p,
            grad_pressure_field=grad_p
        )
        A_u = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        A_u_diag = A_u.diagonal()
        rhs_u = b_u - grad_p[:, 0] * mesh.cell_volumes
        rhs_u_unrelaxed = rhs_u.copy()

        # Relax
        relaxed_A_u_diag, rhs_u = relax_momentum_equation(rhs_u, A_u_diag, U_prev_iter[:,0], config.alpha_uv)
        A_u.setdiag(relaxed_A_u_diag)

        # solve
        U_star[:,0], _, _= scipy_solver(A_u, rhs_u, **linear_solver_settings['momentum'])
        A_u.setdiag(A_u_diag) # Restore original diagonal

        # Store residual field for u-momentum
        u_residual_field = A_u @ U_star[:,0] - rhs_u_unrelaxed

        # compute normalized residual
        u_res, max_u_l2norm = compute_residual(A_u.data, A_u.indices, A_u.indptr, U_star[:,0], rhs_u_unrelaxed, max_u_l2norm, internal_cells)
        u_l2norm.append(u_res)

        #=============================================================================
        # ASSEMBLE and solve V-MOMENTUM EQUATIONS
        #=============================================================================
        row, col, data, b_v = assemble_diffusion_convection_matrix(
            mesh, mdot, grad_v, U_prev_iter, rho, mu, 1, phi=U[:,1],
            scheme=config.convection_scheme, limiter=config.limiter, pressure_field=p,
            grad_pressure_field=grad_p
        )
        A_v = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        A_v_diag = A_v.diagonal()
        rhs_v = b_v - grad_p[:, 1] * mesh.cell_volumes
        rhs_v_unrelaxed = rhs_v.copy()

        # Relax
        relaxed_A_v_diag, rhs_v = relax_momentum_equation(rhs_v, A_v_diag, U_prev_iter[:,1], config.alpha_uv)
        A_v.setdiag(relaxed_A_v_diag)

        # solve
        U_star[:,1], _, _= scipy_solver(A_v, rhs_v, **linear_solver_settings['momentum'])
        A_v.setdiag(A_v_diag) # Restore original diagonal

        # Store residual field for v-momentum
        v_residual_field = A_v @ U_star[:,1] - rhs_v_unrelaxed

        # compute normalized residual
        v_res, max_v_l2norm = compute_residual(A_v.data, A_v.indices, A_v.indptr, U_star[:,1], rhs_v_unrelaxed, max_v_l2norm, internal_cells)
        v_l2norm.append(v_res)

        #=============================================================================
        # RHIE-CHOW, MASS FLUX, and PRESSURE CORRECTION
        #=============================================================================
        bold_D = bold_Dv_calculation(mesh, A_u_diag, A_v_diag)
        bold_D_bar = interpolate_to_face(mesh, bold_D)
        U_star_bar = interpolate_to_face(mesh, U_star)
        U_star_rc = rhie_chow_velocity(mesh, U_star, U_star_bar, U_old_bar, U_old_faces, grad_p_bar, grad_p, p, config.alpha_uv, bold_D_bar)

        mdot_star = mdot_calculation(mesh, rho, U_star_rc)

        row, col, data = assemble_pressure_correction_matrix(mesh, rho)
        A_p = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
        rhs_p = -compute_divergence_from_face_fluxes(mesh, mdot_star)

        # Store continuity residual field
        continuity_field = rhs_p.copy()

        cont_res, max_mass_imbalance = compute_residual(A_p.data, A_p.indices, A_p.indptr, np.zeros_like(p), rhs_p, max_mass_imbalance, internal_cells)
        continuity_l2norm.append(cont_res)

        p_prime, _, ksp_1 = scipy_solver(A_p, rhs_p, remove_nullspace=True, **linear_solver_settings['pressure'])

        #=============================================================================
        # CORRECTIONS (SIMPLE)
        #=============================================================================
        grad_p_prime = compute_cell_gradients(mesh, p_prime, weighted=True, weight_exponent=0.5, use_limiter=False)
        U_prime = velocity_correction(mesh, grad_p_prime, bold_D)
        U_corrected = U_star + U_prime

        U_prime_face = interpolate_to_face(mesh, U_prime)
        U_faces_corrected = U_star_rc + U_prime_face

        mdot_prime = mdot_calculation(mesh, rho, U_prime_face, correction=True)
        mdot_corrected = mdot_star + mdot_prime

        p_corrected = p + config.alpha_p * p_prime

        # Update fields for next iteration
        p = p_corrected
        U = U_corrected
        U_prev_iter = U_corrected.copy()
        mdot = mdot_corrected
        U_old_faces = U_faces_corrected

        is_converged = u_res < tol and v_res < tol
        if is_converged:
            print(f"Converged at iteration {i}")
            break

    time_end = time.time()
    print(f"Solver finished in {time_end - time_start:.2f} seconds.")

    # Compute combined residual
    combined_residual = [max(u, v, c) for u, v, c in zip(u_l2norm, v_l2norm, continuity_l2norm)]

    # Return FV-specific dataclass instances
    from ldc.datastructures import FVFields, TimeSeries, FVMetadata

    fields = FVFields(
        u=U[:, 0],
        v=U[:, 1],
        p=p,
        x=mesh.cell_centers[:, 0],
        y=mesh.cell_centers[:, 1],
        grid_points=mesh.cell_centers,
        mdot=mdot,
    )

    time_series = TimeSeries(
        residual=combined_residual,
        u_residual=u_l2norm,
        v_residual=v_l2norm,
        continuity_residual=continuity_l2norm,
    )

    metadata = FVMetadata(
        # Physics parameters from config
        Re=config.Re,
        lid_velocity=config.lid_velocity,
        Lx=config.Lx,
        Ly=config.Ly,
        # Grid parameters from config
        nx=config.nx,
        ny=config.ny,
        # Convergence info
        iterations=final_iter_count,
        converged=is_converged,
        final_residual=combined_residual[-1] if combined_residual else float('inf'),
        # FV-specific parameters from config
        convection_scheme=config.convection_scheme,
        limiter=config.limiter,
        alpha_uv=config.alpha_uv,
        alpha_p=config.alpha_p,
    )

    return fields, time_series, metadata