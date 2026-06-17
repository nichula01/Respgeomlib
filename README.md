# RespGeomLib

**A reproducible parametric engine for generating analysis-ready human airway lumen geometry.**

RespGeomLib turns compact, human-readable **YAML specifications** into clean,
**simulation-ready triangulated lumen surfaces** with explicit inlet/outlet
ports. It combines analytic pipe/taper segments with **implicit smooth-min
junction blending**, so bifurcations (Y2) and trifurcations (Y3) are seamless
and free of stitched seams or internal walls — while avoiding the cost of
voxelizing the whole airway tree.

This repository accompanies the paper *"RespGeomLib: A Reproducible Parametric
Engine for Generating Analysis-Ready Human Airway Lumen Geometry."*

---

## Key features

- **YAML-driven tree assembly** — each segment declares a primitive (`pipe`,
  `y2`, `y3`) and attaches to a parent port; the builder resolves a
  deterministic build order.
- **Smooth implicit junctions** — bifurcations/trifurcations modeled as the
  zero level-set of a blended tube field and extracted locally with marching
  cubes (no Boolean stitching).
- **Hybrid meshing** — straight/tapered segments stay analytic; implicit
  extraction runs only on tight local grids around junctions.
- **Controlled synthetic variants** — localized radius modulation produces
  reproducible **stenosis** and **dilation** while preserving topology and ports.
- **Morphometry-guided growth** — Weibel/ICRP-style generation trends with
  optional Murray-type diameter constraints.
- **Analysis / CFD-ready export** — open, planar boundary ports for boundary
  conditions; STL/PLY/VTP output.
- **Airway Studio** — an optional desktop GUI for generating, visualizing, and
  exporting airway geometry (incl. an ANSYS Fluent export package).

---

## Repository layout

```
RespGeomLib/
├── respgeomlib/             # the importable Python package (the engine)
│   ├── frames.py            # coordinate frames, angle→direction, minimal-twist child frame
│   ├── primitives.py        # analytic (tapered) cylinder meshes
│   ├── curved_primitives.py # curved pipe along an arbitrary centerline
│   ├── segments.py          # pipe / Y2 / Y3 segment-local geometry + ports
│   ├── junctions.py         # tube-based Y2 junction (baseline)
│   ├── implicit_y.py        # implicit smooth-min Y2 junction (local extraction)
│   ├── implicit_y3.py       # implicit smooth-min Y3 junction (local extraction)
│   ├── tree_builder.py      # YAML loader + port-based tree assembly
│   ├── morphometry_rules.py # Weibel/ICRP/Murray growth rules, subtree generation
│   ├── morphometry_summary.py # per-generation diameter/length summary
│   └── merge_trees.py       # combine base + subtree YAML specs
├── app/
│   └── airway_studio.py     # desktop GUI (PyQt + embedded PyVista viewer)
├── examples/                # small, runnable visualization / usage scripts
├── scripts/
│   └── build_mesh.py        # headless CLI: YAML spec → mesh (+ optional export)
├── trees/                   # example YAML tree specifications
├── results/                 # output figures / generated meshes
├── docs/                    # paper and supplementary documentation
├── pyproject.toml           # package metadata + dependencies
├── requirements.txt         # core engine dependencies
└── requirements_studio.txt  # extra GUI dependencies for Airway Studio
```

---

## Installation

Requires **Python ≥ 3.9**.

```bash
# 1. (recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install the engine as an editable package
pip install -e .

# --- or, without installing the package ---
pip install -r requirements.txt
```

To also run the desktop **Airway Studio** GUI:

```bash
pip install -e ".[studio]"
# or:  pip install -r requirements_studio.txt
```

> The GUI needs a Qt stack (`PyQt6`, `pyvistaqt`) and a display. The core engine
> and the headless CLI/examples only need `numpy`, `pyyaml`, `pyvista`, `vtk`.

---

## Quickstart

### 1. Build a mesh from a YAML spec (headless)

```bash
# print mesh statistics
python scripts/build_mesh.py trees/stenosis_test.yaml

# build and export a watertight-ish surface for downstream tools
python scripts/build_mesh.py trees/example_tree.yaml -o results/example.stl
```

### 2. Use the engine from Python

```python
import numpy as np
import respgeomlib as rgl

specs  = rgl.load_specs_from_yaml("trees/example_tree.yaml")
built  = rgl.build_tree(
    specs,
    root_origin=np.zeros(3),
    root_z_world=np.array([0.0, 0.0, -1.0]),   # trachea pointing down
)
points, faces = rgl.merge_built_segments(built)
print(len(built), "segments,", len(points), "vertices")
```

### 3. Launch the desktop app

```bash
python app/airway_studio.py
```

---

## The YAML tree format

Each tree is a YAML **list** of segments. A segment names a primitive `kind`,
its geometric `params`, and how it attaches to its parent:

```yaml
- id: root                # unique segment id
  kind: pipe              # pipe | pipe_stenosis | pipe_dilation | y2 | y3
  params:
    length: 6.0
    d_in: 2.0
    d_out: 1.0
  parent_id: null         # null for the root
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
    theta1_deg: 45.0      # elevation from +z
    phi1_deg: 0.0         # azimuth in the xy-plane from +x
    theta2_deg: 45.0
    phi2_deg: 120.0
  parent_id: root
  parent_port_index: 1    # which parent port this child attaches to
```

**Synthetic variants** use `pipe_stenosis` / `pipe_dilation`, which modulate the
local radius with a smooth window — `center` (position along the segment in
`[0,1]`), `width`, and `r_min_factor` (severity). See
[`trees/stenosis_test.yaml`](trees/stenosis_test.yaml).

Provided example specs in [`trees/`](trees/):

| File | Description |
|------|-------------|
| `example_tree.yaml` | Minimal pipe + Y2 demonstration tree |
| `human_central_v1.yaml` | Hand-specified central airways |
| `human_central_v2_weibel.yaml` | Central airways with Weibel-style morphometry |
| `right_lung_subtree_auto.yaml` | Auto-generated right-lung subtree |
| `human_central_v2_weibel_with_right_subtree.yaml` | Central airways + merged subtree |
| `stenosis_test.yaml` | Stenotic + dilated variant segments |

---

## Reproducing the paper's workflows

All commands are run from the repository root.

**Morphometry-guided growth & summary**

```bash
# (re)generate an auto right-lung subtree from morphometry rules
python -m respgeomlib.morphometry_rules

# merge base central airways with the subtree
python -m respgeomlib.merge_trees

# print per-generation diameter/length statistics
python -m respgeomlib.morphometry_summary
```

**Junctions and variants (visual)**

> The example scripts `import respgeomlib`, so install the package first
> (`pip install -e .`). Alternatively run them from the repo root with
> `PYTHONPATH=.` set.

```bash
cd examples
python example_y_implicit_vis.py      # smooth implicit Y2 junction
python example_y3_implicit_vis.py     # smooth implicit Y3 (trifurcation)
python example_y_vis.py               # tube-based Y2 baseline
python tree_example.py                # assemble a small tree
python example_human_airway_colored.py  # full central-airway tree, colored by generation
```

> The example and GUI scripts open an interactive PyVista/Qt window and need a
> display. The CLI in `scripts/build_mesh.py` and the `morphometry_*` module
> commands are headless.

The **junction-quality** and **local-vs-global scaling** benchmarks, the
**CFD-ready export** (wall STL + inlet/outlet caps + Fluent journal), and mesh
validity checks are available interactively through **Airway Studio**
(`app/airway_studio.py`).

---

## Citation

If you use RespGeomLib in your work, please cite:

```bibtex
@misc{respgeomlib,
  title  = {RespGeomLib: A Reproducible Parametric Engine for Generating
            Analysis-Ready Human Airway Lumen Geometry},
  author = {Wasalathilaka, Nichula and Ekanayake, Parakrama and Godaliyadda, Roshan},
  note   = {University of Peradeniya},
  url    = {https://github.com/nichula01/RespGeomLib}
}
```

## License

Released under the [MIT License](LICENSE).
