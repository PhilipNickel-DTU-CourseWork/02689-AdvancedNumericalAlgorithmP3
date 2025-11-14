"""Create structured mesh directly in memory as MeshData2D."""

import numpy as np
from .mesh_data import MeshData2D


def create_structured_mesh_2d(nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0,
                                lid_velocity: float = 1.0) -> MeshData2D:
    """Create a uniform structured quad mesh directly as MeshData2D.

    Parameters
    ----------
    nx, ny : int
        Number of cells in x and y directions.
    Lx, Ly : float
        Domain size.
    lid_velocity : float
        Velocity of the top lid (for boundary conditions).

    Returns
    -------
    MeshData2D
        Mesh object ready for FV solver.
    """
    dx = Lx / nx
    dy = Ly / ny

    # --- Cells ---
    n_cells = nx * ny
    cell_volumes = np.full(n_cells, dx * dy, dtype=np.float64)

    # Cell centers: row-major ordering (i + j*nx)
    cell_centers = np.zeros((n_cells, 2), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            idx = i + j * nx
            cell_centers[idx] = [(i + 0.5) * dx, (j + 0.5) * dy]

    # --- Vertices ---
    n_verts = (nx + 1) * (ny + 1)
    vertices = np.zeros((n_verts, 2), dtype=np.float64)
    for j in range(ny + 1):
        for i in range(nx + 1):
            idx = i + j * (nx + 1)
            vertices[idx] = [i * dx, j * dy]

    # --- Faces ---
    # Internal x-faces (vertical): (nx-1) * ny
    # Internal y-faces (horizontal): nx * (ny-1)
    # Boundary: bottom nx, top nx, left ny, right ny
    n_internal_x = (nx - 1) * ny
    n_internal_y = nx * (ny - 1)
    n_internal = n_internal_x + n_internal_y
    n_boundary = 2 * nx + 2 * ny
    n_faces = n_internal + n_boundary

    owner_cells = np.zeros(n_faces, dtype=np.int64)
    neighbor_cells = np.full(n_faces, -1, dtype=np.int64)
    face_centers = np.zeros((n_faces, 2), dtype=np.float64)
    face_areas = np.zeros(n_faces, dtype=np.float64)
    face_vertices = np.zeros((n_faces, 2), dtype=np.int64)
    vector_S_f = np.zeros((n_faces, 2), dtype=np.float64)
    vector_d_CE = np.zeros((n_faces, 2), dtype=np.float64)

    face_idx = 0

    # Internal vertical faces (x-direction)
    for j in range(ny):
        for i in range(nx - 1):
            owner = i + j * nx
            neighbor = (i + 1) + j * nx
            owner_cells[face_idx] = owner
            neighbor_cells[face_idx] = neighbor
            face_centers[face_idx] = [(i + 1) * dx, (j + 0.5) * dy]
            face_areas[face_idx] = dy
            # Vertices: bottom and top of vertical edge
            v1 = (i + 1) + j * (nx + 1)
            v2 = (i + 1) + (j + 1) * (nx + 1)
            face_vertices[face_idx] = [v1, v2]
            # Surface vector points in +x direction
            vector_S_f[face_idx] = [dy, 0.0]
            # Distance vector from owner to neighbor
            vector_d_CE[face_idx] = [dx, 0.0]
            face_idx += 1

    # Internal horizontal faces (y-direction)
    for j in range(ny - 1):
        for i in range(nx):
            owner = i + j * nx
            neighbor = i + (j + 1) * nx
            owner_cells[face_idx] = owner
            neighbor_cells[face_idx] = neighbor
            face_centers[face_idx] = [(i + 0.5) * dx, (j + 1) * dy]
            face_areas[face_idx] = dx
            # Vertices: left and right of horizontal edge
            v1 = i + (j + 1) * (nx + 1)
            v2 = (i + 1) + (j + 1) * (nx + 1)
            face_vertices[face_idx] = [v1, v2]
            # Surface vector points in +y direction
            vector_S_f[face_idx] = [0.0, dx]
            # Distance vector from owner to neighbor
            vector_d_CE[face_idx] = [0.0, dy]
            face_idx += 1

    internal_faces = np.arange(n_internal, dtype=np.int64)
    boundary_start = face_idx

    # Boundary faces
    boundary_patches = np.full(n_faces, -1, dtype=np.int64)
    boundary_types = np.full((n_faces, 2), -1, dtype=np.int64)  # [vel_type, p_type]
    boundary_values = np.zeros((n_faces, 3), dtype=np.float64)  # [u, v, p]
    d_Cb = np.zeros(n_faces, dtype=np.float64)

    # Bottom boundary (y=0)
    for i in range(nx):
        owner = i
        owner_cells[face_idx] = owner
        face_centers[face_idx] = [(i + 0.5) * dx, 0.0]
        face_areas[face_idx] = dx
        v1 = i
        v2 = i + 1
        face_vertices[face_idx] = [v1, v2]
        vector_S_f[face_idx] = [0.0, -dx]  # Outward normal (downward)
        vector_d_CE[face_idx] = [0.0, -dy / 2]
        boundary_patches[face_idx] = 0  # bottom patch
        boundary_types[face_idx] = [0, 3]  # wall velocity, zeroGradient pressure
        boundary_values[face_idx] = [0.0, 0.0, 0.0]
        d_Cb[face_idx] = dy / 2
        face_idx += 1

    # Top boundary (y=Ly) - moving lid
    for i in range(nx):
        owner = i + (ny - 1) * nx
        owner_cells[face_idx] = owner
        face_centers[face_idx] = [(i + 0.5) * dx, Ly]
        face_areas[face_idx] = dx
        v1 = i + ny * (nx + 1)
        v2 = (i + 1) + ny * (nx + 1)
        face_vertices[face_idx] = [v1, v2]
        vector_S_f[face_idx] = [0.0, dx]  # Outward normal (upward)
        vector_d_CE[face_idx] = [0.0, dy / 2]
        boundary_patches[face_idx] = 1  # top patch
        boundary_types[face_idx] = [0, 3]  # wall velocity, zeroGradient pressure
        boundary_values[face_idx] = [lid_velocity, 0.0, 0.0]
        d_Cb[face_idx] = dy / 2
        face_idx += 1

    # Left boundary (x=0)
    for j in range(ny):
        owner = j * nx
        owner_cells[face_idx] = owner
        face_centers[face_idx] = [0.0, (j + 0.5) * dy]
        face_areas[face_idx] = dy
        v1 = j * (nx + 1)
        v2 = (j + 1) * (nx + 1)
        face_vertices[face_idx] = [v1, v2]
        vector_S_f[face_idx] = [-dy, 0.0]  # Outward normal (leftward)
        vector_d_CE[face_idx] = [-dx / 2, 0.0]
        boundary_patches[face_idx] = 2  # left patch
        boundary_types[face_idx] = [0, 3]  # wall velocity, zeroGradient pressure
        boundary_values[face_idx] = [0.0, 0.0, 0.0]
        d_Cb[face_idx] = dx / 2
        face_idx += 1

    # Right boundary (x=Lx)
    for j in range(ny):
        owner = (nx - 1) + j * nx
        owner_cells[face_idx] = owner
        face_centers[face_idx] = [Lx, (j + 0.5) * dy]
        face_areas[face_idx] = dy
        v1 = nx + j * (nx + 1)
        v2 = nx + (j + 1) * (nx + 1)
        face_vertices[face_idx] = [v1, v2]
        vector_S_f[face_idx] = [dy, 0.0]  # Outward normal (rightward)
        vector_d_CE[face_idx] = [dx / 2, 0.0]
        boundary_patches[face_idx] = 3  # right patch
        boundary_types[face_idx] = [0, 3]  # wall velocity, zeroGradient pressure
        boundary_values[face_idx] = [0.0, 0.0, 0.0]
        d_Cb[face_idx] = dx / 2
        face_idx += 1

    boundary_faces = np.arange(boundary_start, n_faces, dtype=np.int64)

    # --- Compute unit vectors and other geometric quantities ---
    unit_vector_n = np.zeros((n_faces, 2), dtype=np.float64)
    unit_vector_e = np.zeros((n_faces, 2), dtype=np.float64)

    for f in range(n_faces):
        # Unit normal (from surface vector)
        S_mag = np.linalg.norm(vector_S_f[f])
        if S_mag > 1e-12:
            unit_vector_n[f] = vector_S_f[f] / S_mag

        # Unit vector e (along d_CE)
        d_mag = np.linalg.norm(vector_d_CE[f])
        if d_mag > 1e-12:
            unit_vector_e[f] = vector_d_CE[f] / d_mag

    # For structured orthogonal mesh, E_f = S_f and T_f = 0
    vector_E_f = vector_S_f.copy()
    vector_T_f = np.zeros((n_faces, 2), dtype=np.float64)
    vector_skewness = np.zeros((n_faces, 2), dtype=np.float64)

    # Interpolation factors (0.5 for uniform grid)
    face_interp_factors = np.full(n_faces, 0.5, dtype=np.float64)

    # Reconstruction weights
    rc_interp_weights = np.zeros(n_faces, dtype=np.float64)
    for f in range(n_internal):
        g_f = face_interp_factors[f]
        delta_PN = np.linalg.norm(vector_d_CE[f])
        if delta_PN > 1e-12 and g_f > 1e-12 and (1 - g_f) > 1e-12:
            rc_interp_weights[f] = 1.0 / (g_f * (1 - g_f) * delta_PN)

    # Cell-to-face connectivity (padded)
    max_faces_per_cell = 4  # Quads
    cell_faces = np.full((n_cells, max_faces_per_cell), -1, dtype=np.int64)

    # Count faces per cell
    face_counts = np.zeros(n_cells, dtype=np.int32)

    for f in range(n_faces):
        owner = owner_cells[f]
        cell_faces[owner, face_counts[owner]] = f
        face_counts[owner] += 1

        neighbor = neighbor_cells[f]
        if neighbor >= 0:
            cell_faces[neighbor, face_counts[neighbor]] = f
            face_counts[neighbor] += 1

    # Binary masks
    face_boundary_mask = np.zeros(n_faces, dtype=np.int64)
    face_boundary_mask[boundary_faces] = 1

    face_flux_mask = np.ones(n_faces, dtype=np.int64)  # All active

    # Create MeshData2D
    mesh = MeshData2D(
        cell_volumes=cell_volumes,
        cell_centers=cell_centers,
        face_areas=face_areas,
        face_centers=face_centers,
        owner_cells=owner_cells,
        neighbor_cells=neighbor_cells,
        cell_faces=cell_faces,
        face_vertices=face_vertices,
        vertices=vertices,
        vector_S_f=vector_S_f,
        vector_d_CE=vector_d_CE,
        unit_vector_n=unit_vector_n,
        unit_vector_e=unit_vector_e,
        vector_E_f=vector_E_f,
        vector_T_f=vector_T_f,
        vector_skewness=vector_skewness,
        face_interp_factors=face_interp_factors,
        rc_interp_weights=rc_interp_weights,
        internal_faces=internal_faces,
        boundary_faces=boundary_faces,
        boundary_patches=boundary_patches,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
        d_Cb=d_Cb,
        face_boundary_mask=face_boundary_mask,
        face_flux_mask=face_flux_mask,
    )

    return mesh
