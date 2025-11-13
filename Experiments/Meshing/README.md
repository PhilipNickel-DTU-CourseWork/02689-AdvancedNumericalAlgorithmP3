# Mesh Generation and Visualization

This directory contains scripts for generating and visualizing meshes for the lid-driven cavity problem.

## Scripts

### `generate_meshes.py`
Generates structured and unstructured meshes at different resolutions.

**Usage:**
```bash
# Generate all meshes (structured and unstructured, all resolutions)
python generate_meshes.py

# Generate only structured meshes
python generate_meshes.py --type structured

# Generate only medium resolution
python generate_meshes.py --resolution medium

# Custom domain size
python generate_meshes.py --domain-size 2.0

# Specific combination
python generate_meshes.py --type unstructured --resolution fine
```

**Options:**
- `--type`: `structured`, `unstructured`, or `both` (default: both)
- `--resolution`: `coarse`, `medium`, `fine`, or `all` (default: all)
- `--domain-size`: Domain size L (default: 1.0)

**Resolutions:**
- **Coarse**: 16x16 cells (structured), h=L/16 (unstructured)
- **Medium**: 32x32 cells (structured), h=L/32 (unstructured)
- **Fine**: 64x64 cells (structured), h=L/64 (unstructured)

**Outputs:**
Meshes are saved to `data/meshing/`:
- `structured_coarse.msh`, `structured_medium.msh`, `structured_fine.msh`
- `unstructured_coarse.msh`, `unstructured_medium.msh`, `unstructured_fine.msh`

### `visualize_meshes.py`
Visualizes mesh files with boundary highlighting and quality metrics.

**Usage:**
```bash
# Visualize most recent mesh
python visualize_meshes.py

# Visualize specific mesh
python visualize_meshes.py data/meshing/structured_medium.msh

# Visualize all meshes
python visualize_meshes.py --all

# Show plot interactively
python visualize_meshes.py --show

# Custom output location
python visualize_meshes.py data/meshing/unstructured_fine.msh --output my_mesh.png
```

**Options:**
- `mesh_file`: Path to mesh file (optional, defaults to most recent)
- `--output`: Custom output plot path
- `--show`: Display plot interactively
- `--all`: Visualize all meshes in data/meshing

**Outputs:**
Plots are saved to `figures/meshing/` showing:
- **Left panel**: Mesh with highlighted boundaries (bottom=blue, right=green, top=red, left=orange)
- **Right panel**: Cell area distribution with colormap

## Examples

### Generate and visualize all meshes
```bash
# Generate all meshes
python generate_meshes.py

# Visualize all generated meshes
python visualize_meshes.py --all
```

### Compare structured vs unstructured
```bash
# Generate both types at medium resolution
python generate_meshes.py --resolution medium

# Visualize structured mesh
python visualize_meshes.py data/meshing/structured_medium.msh

# Visualize unstructured mesh
python visualize_meshes.py data/meshing/unstructured_medium.msh
```

### Generate custom meshes
```bash
# Fine structured mesh only
python generate_meshes.py --type structured --resolution fine

# Coarse unstructured mesh only
python generate_meshes.py --type unstructured --resolution coarse
```

## Mesh Characteristics

### Structured Meshes
- Quadrilateral elements
- Uniform spacing
- Regular connectivity
- Good for simple geometries
- Compatible with FV solver

### Unstructured Meshes
- Triangular elements
- Adaptive sizing (currently uniform)
- Flexible connectivity
- Good for complex geometries
- Compatible with FV solver

## File Format

All meshes are saved in Gmsh `.msh` format (version 2.2) with:
- Physical boundary tags: `bottom`, `right`, `top`, `left`
- Fluid domain tag: `fluid_domain`
- Compatible with meshio and the FV solver
