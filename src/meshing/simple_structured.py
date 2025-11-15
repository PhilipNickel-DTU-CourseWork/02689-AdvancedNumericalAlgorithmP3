"""Simplified structured quad mesh builder for FV solver.

This is a minimal implementation specifically for structured quad meshes,
using pure numpy arrays without external mesh generation libraries.
"""

import numpy as np
from numba import njit
from .mesh_data import MeshData2D


@njit
def _build_face_connectivity(cells):
    """Build face connectivity from quad cells.

    Returns owner/neighbor arrays where internal faces have both,
    boundary faces have neighbor=-1.
    """
    n_cells = cells.shape[0]

    # Estimate face count (4 faces per cell, but internal shared)
    face_dict_size = n_cells * 4
    face_keys = np.zeros((face_dict_size, 2), dtype=np.int64)
    face_owners = np.zeros(face_dict_size, dtype=np.int64)
    face_neighbors = np.full(face_dict_size, -1, dtype=np.int64)
    n_faces = 0

    # Build faces from cells
    for cell_id in range(n_cells):
        cell_nodes = cells[cell_id]

        # Four edges of quad (in order)
        edges = [
            (cell_nodes[0], cell_nodes[1]),
            (cell_nodes[1], cell_nodes[2]),
            (cell_nodes[2], cell_nodes[3]),
            (cell_nodes[3], cell_nodes[0])
        ]

        for n0, n1 in edges:
            # Canonical edge ordering (smaller index first)
            if n0 > n1:
                n0, n1 = n1, n0

            # Check if face already exists
            found = False
            for f in range(n_faces):
                if face_keys[f, 0] == n0 and face_keys[f, 1] == n1:
                    # Face exists, this cell is neighbor
                    face_neighbors[f] = cell_id
                    found = True
                    break

            if not found:
                # New face, this cell is owner
                face_keys[n_faces, 0] = n0
                face_keys[n_faces, 1] = n1
                face_owners[n_faces] = cell_id
                n_faces += 1

    return face_keys[:n_faces], face_owners[:n_faces], face_neighbors[:n_faces]


@njit
def _compute_face_geometry(points, face_vertices, owner_cells, neighbor_cells, cell_centers):
    """Compute face centers, areas, and normal vectors."""
    n_faces = face_vertices.shape[0]

    face_centers = np.zeros((n_faces, 2), dtype=np.float64)
    face_areas = np.zeros(n_faces, dtype=np.float64)
    vector_S_f = np.zeros((n_faces, 2), dtype=np.float64)

    for f in range(n_faces):
        v0_idx = face_vertices[f, 0]
        v1_idx = face_vertices[f, 1]

        v0 = points[v0_idx]
        v1 = points[v1_idx]

        # Face center
        face_centers[f] = 0.5 * (v0 + v1)

        # Edge vector and length
        edge = v1 - v0
        length = np.sqrt(edge[0]**2 + edge[1]**2)
        face_areas[f] = length

        # Normal vector (rotate edge by 90 degrees)
        # Convention: normal points from owner to neighbor
        normal = np.array([edge[1], -edge[0]])

        # Check orientation: normal should point outward from owner
        owner = owner_cells[f]
        neighbor = neighbor_cells[f]

        if neighbor >= 0:
            # Internal face: normal points from owner to neighbor
            d = cell_centers[neighbor] - cell_centers[owner]
            if normal[0] * d[0] + normal[1] * d[1] < 0:
                normal = -normal
        else:
            # Boundary face: normal points outward from owner
            d = face_centers[f] - cell_centers[owner]
            if normal[0] * d[0] + normal[1] * d[1] < 0:
                normal = -normal

        vector_S_f[f] = normal

    return face_centers, face_areas, vector_S_f


@njit
def _compute_geometric_factors(n_faces, owner_cells, neighbor_cells,
                                 cell_centers, face_centers, vector_S_f, face_areas):
    """Compute geometric factors for FV discretization."""
    vector_d_CE = np.zeros((n_faces, 2), dtype=np.float64)
    unit_vector_n = np.zeros((n_faces, 2), dtype=np.float64)
    unit_vector_e = np.zeros((n_faces, 2), dtype=np.float64)
    face_interp_factors = np.zeros(n_faces, dtype=np.float64)
    d_Cb = np.zeros(n_faces, dtype=np.float64)

    for f in range(n_faces):
        owner = owner_cells[f]
        neighbor = neighbor_cells[f]

        # Unit normal
        if face_areas[f] > 1e-12:
            unit_vector_n[f] = vector_S_f[f] / face_areas[f]

        if neighbor >= 0:
            # Internal face
            vector_d_CE[f] = cell_centers[neighbor] - cell_centers[owner]
            d_mag = np.sqrt(vector_d_CE[f, 0]**2 + vector_d_CE[f, 1]**2)

            if d_mag > 1e-12:
                unit_vector_e[f] = vector_d_CE[f] / d_mag

                # Distance from owner to face
                d_Pf = face_centers[f] - cell_centers[owner]
                delta_Pf = np.sqrt(d_Pf[0]**2 + d_Pf[1]**2)

                # Interpolation factor
                face_interp_factors[f] = delta_Pf / d_mag
        else:
            # Boundary face
            d_boundary = face_centers[f] - cell_centers[owner]
            d_Cb[f] = np.sqrt(d_boundary[0]**2 + d_boundary[1]**2)
            vector_d_CE[f] = d_boundary

            if d_Cb[f] > 1e-12:
                unit_vector_e[f] = d_boundary / d_Cb[f]

    return vector_d_CE, unit_vector_n, unit_vector_e, face_interp_factors, d_Cb


def create_structured_mesh_2d(nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0,
                                lid_velocity: float = 1.0) -> MeshData2D:
    """Create structured Cartesian quad mesh using pure numpy.

    This implementation:
    - Uses numpy meshgrid to generate a uniform Cartesian grid
    - Builds FV connectivity for structured quad cells
    - Hard-codes lid-driven cavity boundary conditions
    - Assumes orthogonal grid (no non-orthogonality corrections needed)

    Parameters
    ----------
    nx, ny : int
        Number of cells in x and y directions
    Lx, Ly : float
        Domain size in x and y directions
    lid_velocity : float
        Velocity of the top lid

    Returns
    -------
    MeshData2D
        Mesh data structure ready for FV solver
    """
    # 1. Generate uniform Cartesian grid vertices
    # Create (nx+1) x (ny+1) vertices
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Flatten to get vertex coordinates
    points = np.column_stack([X.ravel(), Y.ravel()])
    n_vertices = len(points)

    # 2. Build quad cells
    # Each cell is defined by 4 vertices in counter-clockwise order:
    # v3---v2
    # |    |
    # v0---v1
    cells = []
    for i in range(nx):
        for j in range(ny):
            # Vertex indices for cell (i,j)
            v0 = i * (ny + 1) + j
            v1 = (i + 1) * (ny + 1) + j
            v2 = (i + 1) * (ny + 1) + (j + 1)
            v3 = i * (ny + 1) + (j + 1)
            cells.append([v0, v1, v2, v3])

    cells = np.array(cells, dtype=np.int64)
    n_cells = len(cells)

    # 3. Compute cell geometry
    # For uniform Cartesian grid: all cells have same area
    dx = Lx / nx
    dy = Ly / ny
    cell_area = dx * dy
    cell_volumes = np.full(n_cells, cell_area)

    # Cell centers are at the geometric center of each quad
    cell_centers = np.mean(points[cells], axis=1)

    # 4. Build face connectivity
    face_vertices, owner_cells, neighbor_cells = _build_face_connectivity(cells)
    n_faces = len(face_vertices)

    # 5. Face geometry
    face_centers, face_areas, vector_S_f = _compute_face_geometry(
        points, face_vertices, owner_cells, neighbor_cells, cell_centers
    )

    # 6. Geometric factors
    vector_d_CE, unit_vector_n, unit_vector_e, face_interp_factors, d_Cb = \
        _compute_geometric_factors(n_faces, owner_cells, neighbor_cells,
                                     cell_centers, face_centers, vector_S_f, face_areas)

    # 7. Boundary conditions (lid-driven cavity specific)
    internal_faces = np.where(neighbor_cells >= 0)[0].astype(np.int64)
    boundary_faces_list = []
    boundary_types = np.full((n_faces, 2), -1, dtype=np.int64)
    boundary_values = np.zeros((n_faces, 3), dtype=np.float64)

    # Identify boundary faces by checking face centers against domain boundaries
    eps = 1e-10
    for f in range(n_faces):
        if neighbor_cells[f] >= 0:
            continue  # Internal face

        boundary_faces_list.append(f)
        fc = face_centers[f]

        # Determine which boundary this face belongs to
        if abs(fc[1] - Ly) < eps:
            # Top boundary (moving lid)
            boundary_types[f] = [0, 3]  # wall velocity, zeroGradient pressure
            boundary_values[f] = [lid_velocity, 0.0, 0.0]
        elif abs(fc[1] - 0.0) < eps:
            # Bottom boundary (stationary wall)
            boundary_types[f] = [0, 3]
            boundary_values[f] = [0.0, 0.0, 0.0]
        elif abs(fc[0] - 0.0) < eps:
            # Left boundary (stationary wall)
            boundary_types[f] = [0, 3]
            boundary_values[f] = [0.0, 0.0, 0.0]
        elif abs(fc[0] - Lx) < eps:
            # Right boundary (stationary wall)
            boundary_types[f] = [0, 3]
            boundary_values[f] = [0.0, 0.0, 0.0]

    boundary_faces = np.array(boundary_faces_list, dtype=np.int64)

    # 8. Simplified structured mesh: skip non-orthogonality terms
    vector_E_f = vector_S_f.copy()  # E_f = S_f for orthogonal mesh
    vector_T_f = np.zeros_like(vector_S_f)  # T_f = 0 for orthogonal mesh
    vector_skewness = np.zeros_like(face_centers)  # No skewness

    # Rhie-Chow weights
    rc_interp_weights = np.zeros(n_faces)
    for f in internal_faces:
        g_f = face_interp_factors[f]
        delta_PN = np.linalg.norm(vector_d_CE[f])
        if delta_PN > 1e-12 and g_f > 1e-12 and (1 - g_f) > 1e-12:
            rc_interp_weights[f] = 1.0 / (g_f * (1 - g_f) * delta_PN)

    # Cell-face connectivity
    max_faces = 4  # Quads have 4 faces
    cell_faces = np.full((n_cells, max_faces), -1, dtype=np.int64)
    face_count = np.zeros(n_cells, dtype=np.int32)

    for f in range(n_faces):
        owner = owner_cells[f]
        cell_faces[owner, face_count[owner]] = f
        face_count[owner] += 1

        neighbor = neighbor_cells[f]
        if neighbor >= 0:
            cell_faces[neighbor, face_count[neighbor]] = f
            face_count[neighbor] += 1

    # Masks
    face_boundary_mask = np.zeros(n_faces, dtype=np.int64)
    face_boundary_mask[boundary_faces] = 1
    face_flux_mask = np.ones(n_faces, dtype=np.int64)

    # Build MeshData2D
    return MeshData2D(
        cell_volumes=np.ascontiguousarray(cell_volumes),
        cell_centers=np.ascontiguousarray(cell_centers),
        face_areas=np.ascontiguousarray(face_areas),
        face_centers=np.ascontiguousarray(face_centers),
        owner_cells=np.ascontiguousarray(owner_cells),
        neighbor_cells=np.ascontiguousarray(neighbor_cells),
        cell_faces=np.ascontiguousarray(cell_faces),
        vector_S_f=np.ascontiguousarray(vector_S_f),
        vector_d_CE=np.ascontiguousarray(vector_d_CE),
        unit_vector_n=np.ascontiguousarray(unit_vector_n),
        unit_vector_e=np.ascontiguousarray(unit_vector_e),
        vector_E_f=np.ascontiguousarray(vector_E_f),
        vector_T_f=np.ascontiguousarray(vector_T_f),
        vector_skewness=np.ascontiguousarray(vector_skewness),
        face_interp_factors=np.ascontiguousarray(face_interp_factors),
        rc_interp_weights=np.ascontiguousarray(rc_interp_weights),
        internal_faces=np.ascontiguousarray(internal_faces),
        boundary_faces=np.ascontiguousarray(boundary_faces),
        boundary_types=np.ascontiguousarray(boundary_types),
        boundary_values=np.ascontiguousarray(boundary_values),
        d_Cb=np.ascontiguousarray(d_Cb),
        face_boundary_mask=np.ascontiguousarray(face_boundary_mask),
        face_flux_mask=np.ascontiguousarray(face_flux_mask),
    )
