#!/usr/bin/env python3
"""Generate meshes for lid-driven cavity problem."""

# %% Imports
import sys
from pathlib import Path
import gmsh
import yaml

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from meshing import generate_structured

# %% Load configuration
config_file = Path(__file__).parent / "experiments.yaml"
with open(config_file) as f:
    config = yaml.safe_load(f)

resolutions = config["lidDrivenCavity"]["uniform"]["resolutions"]

# %% Setup directories
data_dir = repo_root / "data" / "meshes"
(data_dir / "structured").mkdir(parents=True, exist_ok=True)
(data_dir / "unstructured").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Generating meshes...")
print("=" * 60)

# %% Generate structured meshes
for name, params in resolutions.items():
    output = data_dir / "structured" / f"{name}.msh"
    generate_structured(L=params["L"], nx=params["nx"], ny=params["ny"], output_filename=str(output))
    print(f"✓ Structured {name}: {params['nx']}x{params['ny']} cells")

# %% Generate unstructured meshes
gmsh.initialize()
for name, params in resolutions.items():
    gmsh.clear()
    gmsh.model.add(f"unstructured_{name}")

    # Geometry
    L = params["L"]
    h = L / params["nx"]
    p1, p2, p3, p4 = [gmsh.model.geo.addPoint(x, y, 0, h) for x, y in [(0,0), (L,0), (L,L), (0,L)]]
    l1, l2, l3, l4 = [gmsh.model.geo.addLine(p1, p2), gmsh.model.geo.addLine(p2, p3),
                      gmsh.model.geo.addLine(p3, p4), gmsh.model.geo.addLine(p4, p1)]
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    # Physical groups
    for i, line, tag in [(1, l1, "bottom"), (2, l2, "right"), (3, l3, "top"), (4, l4, "left")]:
        gmsh.model.addPhysicalGroup(1, [line], name=tag)
    gmsh.model.addPhysicalGroup(2, [surf], 10, name="fluid_domain")

    # Mesh and save
    gmsh.model.mesh.generate(2)
    output = data_dir / "unstructured" / f"{name}.msh"
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveGroupsOfElements", 1)
    gmsh.write(str(output))

    n_elem = len(gmsh.model.mesh.getElements(2)[1][0])
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    print(f"✓ Unstructured {name}: {n_elem} elements, {n_nodes} nodes")

gmsh.finalize()

# %% Done
print("=" * 60)
print(f"Meshes saved to: {data_dir}")
print("=" * 60)
