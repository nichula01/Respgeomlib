# RespGeomLib

**A reproducible parametric engine for generating analysis-ready human airway lumen geometry.**

RespGeomLib converts compact **YAML airway specifications** into smooth, editable, and simulation-ready 3D lumen meshes with explicit inlet/outlet ports.

It combines:

* analytic pipe/tapered segments,
* smooth implicit Y2/Y3 junction blending,
* local marching-cubes extraction around bifurcations,
* controlled stenosis/dilation variants,
* morphometry-guided airway growth,
* STL/PLY/VTP export for downstream analysis and CFD.

This repository accompanies the paper:

> **RespGeomLib: A Reproducible Parametric Engine for Generating Analysis-Ready Human Airway Lumen Geometry**
> Accepted at **IEEE MERCon 2026**

---

## Highlights

| Capability                    | Description                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------ |
| **YAML-driven assembly**      | Define airway trees using editable human-readable specifications.                          |
| **Smooth implicit junctions** | Generate seamless Y2/Y3 junctions without Boolean stitching or internal walls.             |
| **Hybrid meshing**            | Keep straight/tapered segments analytic and extract implicit surfaces only near junctions. |
| **Synthetic variants**        | Create controlled stenosis and dilation while preserving topology and ports.               |
| **Morphometry-guided growth** | Generate Weibel/ICRP-style airway subtrees with optional Murray-type constraints.          |
| **CFD-ready export**          | Export meshes with clean open ports for boundary-condition assignment.                     |
| **Airway Studio**             | Optional PyQt/PyVista GUI for generation, visualization, and Fluent export.                |

---

## Repository layout

```text
RespGeomLib/
├── respgeomlib/               # core Python package
│   ├── frames.py              # coordinate frames and minimal-twist placement
│   ├── primitives.py          # analytic straight/tapered cylinder meshes
│   ├── curved_primitives.py   # curved pipe along a centerline
│   ├── segments.py            # pipe / Y2 / Y3 segment geometry and ports
│   ├── junctions.py           # tube-based Y2 baseline
│   ├── implicit_y.py          # smooth-min implicit Y2 junction
│   ├── implicit_y3.py         # smooth-min implicit Y3 junction
│   ├── tree_builder.py        # YAML loading and port-based assembly
│   ├── morphometry_rules.py   # Weibel/ICRP/Murray growth rules
│   ├── morphometry_summary.py # generation-wise diameter/length summary
│   └── merge_trees.py         # combine base and generated subtree specs
├── app/
│   └── airway_studio.py       # optional desktop GUI
├── examples/                  # runnable visualization and usage examples
├── scripts/
│   └── build_mesh.py          # CLI: YAML specification → mesh
├── trees/                     # example YAML airway trees
├── results/                   # generated outputs and figures
├── docs/                      # paper and supplementary material
├── website/                   # project website
├── pyproject.toml             # package metadata
├── requirements.txt           # core dependencies
└── requirements_studio.txt    # GUI dependencies
```

---

## Installation

Requires **Python ≥ 3.9**.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .
```

Alternative installation without editable mode:

```bash
pip install -r requirements.txt
```

To run the optional desktop GUI:

```bash
pip install -e ".[studio]"
# or
pip install -r requirements_studio.txt
```

> The GUI requires a Qt stack such as `PyQt6` and `pyvistaqt`.
> The core engine and headless CLI mainly require `numpy`, `pyyaml`, `pyvista`, and `vtk`.

---

## Quickstart

### 1. Build a mesh from a YAML file

```bash
python scripts/build_mesh.py trees/stenosis_test.yaml
```

Export a mesh:

```bash
python scripts/build_mesh.py trees/example_tree.yaml -o results/example.stl
```

### 2. Use RespGeomLib from Python

```python
import numpy as np
import respgeomlib as rgl

specs = rgl.load_specs_from_yaml("trees/example_tree.yaml")

built = rgl.build_tree(
    specs,
    root_origin=np.zeros(3),
    root_z_world=np.array([0.0, 0.0, -1.0]),
)

points, faces = rgl.merge_built_segments(built)

print(len(built), "segments")
print(len(points), "vertices")
```

### 3. Launch Airway Studio

```bash
python app/airway_studio.py
```

---

## YAML tree format

RespGeomLib trees are defined as a list of connected segments.

Each segment specifies:

* a unique `id`,
* a primitive `kind`,
* geometric `params`,
* parent attachment information.

```yaml
- id: root
  kind: pipe
  params:
    length: 6.0
    d_in: 2.0
    d_out: 1.0
  parent_id: null
  parent_port_index: null

- id: Y2_main
  kind: y2
  params:
    length_trunk: 6.0
    length_child1: 4.0
    length_child2: 4.0
    d_trunk: 2.0
    d_child1: 1.0
    d_child2: 1.5
    theta1_deg: 45.0
    phi1_deg: 0.0
    theta2_deg: 45.0
    phi2_deg: 120.0
  parent_id: root
  parent_port_index: 1
```

Synthetic variants use:

```text
pipe_stenosis
pipe_dilation
```

with parameters such as:

```text
center, width, r_min_factor
```

See:

```text
trees/stenosis_test.yaml
```

---

## Example tree specifications

| File                                              | Description                                        |
| ------------------------------------------------- | -------------------------------------------------- |
| `example_tree.yaml`                               | Minimal pipe + Y2 demonstration tree               |
| `human_central_v1.yaml`                           | Hand-specified central airway model                |
| `human_central_v2_weibel.yaml`                    | Central airway model with Weibel-style morphometry |
| `right_lung_subtree_auto.yaml`                    | Automatically generated right-lung subtree         |
| `human_central_v2_weibel_with_right_subtree.yaml` | Central airway model merged with generated subtree |
| `stenosis_test.yaml`                              | Stenotic and dilated synthetic variants            |

---

## Reproducing paper workflows

Run all commands from the repository root.

### Morphometry-guided growth

```bash
python -m respgeomlib.morphometry_rules
python -m respgeomlib.merge_trees
python -m respgeomlib.morphometry_summary
```

### Junction and tree visualizations

Install the package first:

```bash
pip install -e .
```

Then run:

```bash
cd examples

python example_y_implicit_vis.py
python example_y3_implicit_vis.py
python example_y_vis.py
python tree_example.py
python example_human_airway_colored.py
```

> Visualization scripts open PyVista/Qt windows and require a display.
> The CLI tools and morphometry modules can run headlessly.

---

## Airway Studio

The optional desktop GUI supports:

* YAML-based tree loading,
* geometry preview,
* controlled variant generation,
* mesh export,
* validity checks,
* CFD/ANSYS Fluent export package generation.

Launch:

```bash
python app/airway_studio.py
```

---

## Project website

The paper website is included in:

```text
website/
```

It contains:

* project landing page,
* paper figures,
* interactive STL preview,
* demo placeholder,
* citation section.

---

## Citation

If you use RespGeomLib, please cite:

```bibtex
@inproceedings{wasalathilaka2026respgeomlib,
  title     = {RespGeomLib: A Reproducible Parametric Engine for Generating
               Analysis-Ready Human Airway Lumen Geometry},
  author    = {Wasalathilaka, Nichula and Ekanayake, Parakrama and Godaliyadda, Roshan},
  booktitle = {IEEE MERCon},
  year      = {2026}
}
```

Citation details will be updated after the official IEEE proceedings release.

---

## License

See [`LICENSE`](LICENSE).
