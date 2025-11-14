"""Create structured mesh using gmsh (in-memory) and convert to MeshData2D."""

import numpy as np
import gmsh
from .mesh_data import MeshData2D
from .mesh_loader import _build_meshdata2d


def create_structured_mesh_2d(nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0,
                                lid_velocity: float = 1.0) -> MeshData2D:
    """Create a uniform structured quad mesh using gmsh (no file I/O).

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
    # Initialize gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()

    gmsh.clear()
    gmsh.model.add("structured_cavity")

    # 1. Create geometry
    lc = Lx / max(nx, ny)

    def add_point(x, y):
        return gmsh.model.geo.addPoint(x, y, 0.0, lc)

    # Create corner points
    p = [
        add_point(0, 0),      # bottom-left
        add_point(Lx, 0),     # bottom-right
        add_point(Lx, Ly),    # top-right
        add_point(0, Ly)      # top-left
    ]

    # Create boundary lines
    lines = [
        gmsh.model.geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)
    ]  # [bottom, right, top, left]

    # Create surface
    cloop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([cloop])

    # 2. Set transfinite mesh (structured quad mesh)
    gmsh.model.geo.mesh.setTransfiniteCurve(lines[0], nx + 1)  # bottom
    gmsh.model.geo.mesh.setTransfiniteCurve(lines[2], nx + 1)  # top
    gmsh.model.geo.mesh.setTransfiniteCurve(lines[1], ny + 1)  # right
    gmsh.model.geo.mesh.setTransfiniteCurve(lines[3], ny + 1)  # left
    gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    gmsh.model.geo.mesh.setRecombine(2, surface)

    gmsh.model.geo.synchronize()

    # 3. Physical tagging for boundaries
    boundary_names = ["bottom", "right", "top", "left"]
    for i, name in enumerate(boundary_names):
        tag = gmsh.model.addPhysicalGroup(1, [lines[i]])
        gmsh.model.setPhysicalName(1, tag, name)

    # Fluid domain
    fluid_tag = gmsh.model.addPhysicalGroup(2, [surface], 10)
    gmsh.model.setPhysicalName(2, fluid_tag, "fluid_domain")

    # 4. Generate mesh
    gmsh.model.mesh.generate(2)

    # 5. Extract mesh data from gmsh (in-memory)
    # Get nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    points = node_coords.reshape(-1, 3)[:, :2]  # Extract x, y coordinates

    # Build node tag to index map (gmsh tags are 1-indexed)
    node_tag_to_idx = {int(tag): idx for idx, tag in enumerate(node_tags)}

    # Get quad elements (cell type 3 = quad)
    elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2)

    # Find quad elements
    quad_idx = None
    for i, elem_type in enumerate(elem_types):
        if elem_type == 3:  # Quad
            quad_idx = i
            break

    if quad_idx is None:
        gmsh.finalize()
        raise ValueError("No quad elements found in mesh")

    # Convert element node tags to 0-indexed array
    elem_node_tags = elem_node_tags_list[quad_idx]
    cells_raw = elem_node_tags.reshape(-1, 4)

    # Convert gmsh node tags to 0-indexed
    cells = np.array([[node_tag_to_idx[int(tag)] for tag in cell] for cell in cells_raw], dtype=np.int64)

    # 6. Extract boundary edges
    # Get physical groups (dimension 1 = lines/edges)
    physical_groups = gmsh.model.getPhysicalGroups(1)

    physical_id_to_name = {}
    boundary_lines_list = []
    boundary_tags_list = []

    for dim, tag in physical_groups:
        name = gmsh.model.getPhysicalName(dim, tag)
        physical_id_to_name[tag] = name

        # Get entities in this physical group
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)

        # Get elements (edges) for each entity
        for entity in entities:
            elem_types_1d, elem_tags_1d, elem_node_tags_1d = gmsh.model.mesh.getElements(dim, entity)

            if len(elem_types_1d) > 0:
                # Line elements (type 1)
                line_node_tags = elem_node_tags_1d[0].reshape(-1, 2)

                # Convert to 0-indexed
                for line in line_node_tags:
                    edge = [node_tag_to_idx[int(line[0])], node_tag_to_idx[int(line[1])]]
                    boundary_lines_list.append(edge)
                    boundary_tags_list.append(tag)

    boundary_lines = np.array(boundary_lines_list, dtype=np.int64)
    boundary_tags = np.array(boundary_tags_list, dtype=np.int64)

    # Finalize gmsh
    gmsh.finalize()

    # 7. Setup boundary conditions for lid-driven cavity
    boundary_conditions = {
        "bottom": {
            "velocity": {"bc": "wall", "value": [0.0, 0.0]},
            "pressure": {"bc": "neumann", "value": 0.0}
        },
        "right": {
            "velocity": {"bc": "wall", "value": [0.0, 0.0]},
            "pressure": {"bc": "neumann", "value": 0.0}
        },
        "top": {
            "velocity": {"bc": "wall", "value": [lid_velocity, 0.0]},
            "pressure": {"bc": "neumann", "value": 0.0}
        },
        "left": {
            "velocity": {"bc": "wall", "value": [0.0, 0.0]},
            "pressure": {"bc": "neumann", "value": 0.0}
        }
    }

    # 8. Use existing mesh_loader infrastructure to build MeshData2D
    return _build_meshdata2d(
        points=points,
        cells=cells,
        boundary_lines=boundary_lines,
        boundary_tags=boundary_tags,
        physical_id_to_name=physical_id_to_name,
        boundary_conditions=boundary_conditions
    )
