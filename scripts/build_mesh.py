#!/usr/bin/env python3
"""
Build an airway lumen mesh from a YAML tree specification (no GUI required).

This is the headless, reproducible entry point to the RespGeomLib engine: it
loads a YAML spec, assembles the tree, merges the per-segment meshes, and
optionally writes the result to disk (e.g. STL/PLY/VTP) for downstream
analysis or CFD.

Examples
--------
    python scripts/build_mesh.py trees/stenosis_test.yaml
    python scripts/build_mesh.py trees/example_tree.yaml -o results/example.stl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running directly (``python scripts/build_mesh.py``) without an editable
# install by putting the repository root (which holds ``respgeomlib``) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from respgeomlib import load_specs_from_yaml, build_tree, merge_built_segments


def _faces_to_pyvista(faces: np.ndarray) -> np.ndarray:
    """Convert (M, 3) triangle indices to PyVista's flat face array."""
    return np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).ravel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="Path to a YAML tree specification")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output mesh path (.stl/.ply/.vtp). If omitted, only stats are printed.",
    )
    parser.add_argument(
        "--root-direction",
        type=float,
        nargs=3,
        default=(0.0, 0.0, -1.0),
        metavar=("X", "Y", "Z"),
        help="World-space direction of the root segment's +z axis (default: 0 0 -1).",
    )
    args = parser.parse_args()

    specs = load_specs_from_yaml(str(args.spec))
    if not specs:
        raise SystemExit(f"No segments loaded from {args.spec}")

    built = build_tree(
        specs,
        root_origin=np.zeros(3),
        root_z_world=np.asarray(args.root_direction, dtype=float),
    )
    points, faces = merge_built_segments(built)

    print(f"Spec      : {args.spec}")
    print(f"Segments  : {len(built)}")
    print(f"Vertices  : {len(points)}")
    print(f"Triangles : {len(faces)}")

    if args.output is not None:
        import pyvista as pv

        mesh = pv.PolyData(points, _faces_to_pyvista(faces))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mesh.save(str(args.output))
        print(f"Saved     : {args.output}")


if __name__ == "__main__":
    main()
