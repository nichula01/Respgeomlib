from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import yaml

from frames import Frame, unit
from segments import (
    Port,
    SegmentGeom,
    build_pipe_segment,
    build_pipe_stenosis_segment,
    build_pipe_dilation_segment,
    build_y2_segment,
    build_y3_segment,
)


@dataclass
class SegmentSpec:
    """
    High-level specification of one airway segment in the tree.

    Attributes
    ----------
    id : str
        Unique identifier for this segment.
    kind : str
        One of {"pipe", "pipe_stenosis", "pipe_dilation", "y2", "y3"}.
    params : dict
        Keyword arguments to pass to the corresponding build_*_segment function.
    parent_id : Optional[str]
        ID of the parent segment, or None if this is the root.
    parent_port_index : Optional[int]
        Index of the parent segment's port where this segment attaches.
        Ignored for the root.
    meta : Optional[dict]
        Optional metadata for downstream use (e.g., labels, region, generation).
    """

    id: str
    kind: str
    params: Dict[str, Any]
    parent_id: Optional[str]
    parent_port_index: Optional[int]
    meta: Optional[Dict[str, Any]] = None


# Expected YAML format:
# A list of segment entries, e.g.:
#
# - id: root
#   kind: pipe
#   params:
#     length: 6.0
#     d_in: 2.0
#     d_out: 2.0
#   parent_id: null
#   parent_port_index: null
#
# - id: Y2_main
#   kind: y2
#   params:
#     length_trunk: 6.0
#     length_child1: 4.0
#     length_child2: 4.0
#     d_trunk: 2.0
#     d_child1: 2.0
#     d_child2: 2.0
#     theta1_deg: 45.0
#     phi1_deg: 0.0
#     theta2_deg: 45.0
#     phi2_deg: 120.0
#   parent_id: root
#   parent_port_index: 1
#
# etc.

@dataclass
class BuiltSegment:
    """
    A segment that has been instantiated in world coordinates.

    Attributes
    ----------
    spec : SegmentSpec
        Original specification.
    geom_local : SegmentGeom
        Geometry and ports in segment-local coordinates.
    frame_world : Frame
        World frame of the segment-local coordinates.
    points_world : (N, 3) float
        Surface vertices in world coordinates.
    faces : (M, 3) int
        Triangle indices into points_world (local to this segment).
    ports_world : list of Port
        Ports with positions and directions expressed in world coordinates.
    """

    spec: SegmentSpec
    geom_local: SegmentGeom
    frame_world: Frame
    points_world: np.ndarray
    faces: np.ndarray
    ports_world: List[Port]


def frame_from_origin_and_z(origin: np.ndarray, z_world: np.ndarray) -> Frame:
    """
    Build a Frame whose origin is `origin` and whose local +z axis
    aligns with `z_world`, with minimal twist using a fixed global
    x-axis as reference.
    """
    origin = np.asarray(origin, dtype=float).reshape(3)
    z = unit(z_world)

    # Use global x as a reference; if it's too parallel to z, use global y.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, z)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])

    x_proj = ref - np.dot(ref, z) * z
    x = unit(x_proj)
    y = unit(np.cross(z, x))

    R = np.column_stack((x, y, z))
    return Frame(origin=origin, R=R)


def segment_to_world(seg: SegmentGeom, frame: Frame) -> Tuple[np.ndarray, np.ndarray, List[Port]]:
    """
    Transform a segment's points and ports from segment-local coordinates
    into world coordinates using the given Frame.

    Returns
    -------
    points_world : (N, 3)
        Transformed vertices.
    faces : (M, 3)
        Same faces array (indices local to this segment).
    ports_world : list of Port
        Ports with position and direction expressed in world coordinates.
    """
    pts_world = frame.to_world(seg.points)

    ports_world: List[Port] = []
    for port in seg.ports:
        pos_w = frame.to_world(port.position).reshape(3)
        dir_w = np.asarray(frame.R @ port.direction, dtype=float).reshape(3)
        ports_world.append(Port(position=pos_w, direction=dir_w))

    return pts_world, seg.faces.copy(), ports_world


def build_segment_geom(spec: SegmentSpec) -> SegmentGeom:
    """
    Dispatch to the appropriate segment builder based on spec.kind.
    """
    kind = spec.kind.lower()
    if kind == "pipe":
        return build_pipe_segment(**spec.params)
    elif kind == "pipe_stenosis":
        return build_pipe_stenosis_segment(**spec.params)
    elif kind == "pipe_dilation":
        return build_pipe_dilation_segment(**spec.params)
    elif kind == "y2":
        return build_y2_segment(**spec.params)
    elif kind == "y3":
        return build_y3_segment(**spec.params)
    else:
        raise ValueError(f"Unknown segment kind: {spec.kind!r}")


def build_tree(
    specs: List[SegmentSpec],
    root_origin: np.ndarray,
    root_z_world: np.ndarray,
) -> Dict[str, BuiltSegment]:
    """
    Instantiate a tree of airway segments in world coordinates.

    Parameters
    ----------
    specs : list of SegmentSpec
        High-level description of the airway tree.
    root_origin : (3,) float
        World-space origin of the root segment's parent port.
    root_z_world : (3,) float
        World-space direction of the root segment's +z axis (into the root).

    Returns
    -------
    built : dict
        Mapping from segment id to BuiltSegment.
    """
    spec_by_id: Dict[str, SegmentSpec] = {s.id: s for s in specs}
    root_specs = [s for s in specs if s.parent_id is None]
    if len(root_specs) != 1:
        raise ValueError(f"Expected exactly one root segment, found {len(root_specs)}")
    root_spec = root_specs[0]

    built: Dict[str, BuiltSegment] = {}

    # Build until all specs are instantiated.
    # Simple iterative approach: each pass tries to build any segment whose parent is ready.
    while len(built) < len(specs):
        progress = False
        for spec in specs:
            if spec.id in built:
                continue

            if spec.parent_id is None:
                # Root segment.
                frame = frame_from_origin_and_z(root_origin, root_z_world)
                geom = build_segment_geom(spec)
                pts_w, faces, ports_w = segment_to_world(geom, frame)
                built[spec.id] = BuiltSegment(
                    spec=spec,
                    geom_local=geom,
                    frame_world=frame,
                    points_world=pts_w,
                    faces=faces,
                    ports_world=ports_w,
                )
                progress = True
            else:
                # Non-root: only build if parent is available.
                parent_id = spec.parent_id
                if parent_id not in built:
                    continue
                parent = built[parent_id]
                if spec.parent_port_index is None:
                    raise ValueError(f"Segment {spec.id!r} has parent_id but no parent_port_index")
                if not (0 <= spec.parent_port_index < len(parent.ports_world)):
                    raise IndexError(
                        f"parent_port_index {spec.parent_port_index} out of range for parent {parent_id!r}"
                    )

                parent_port = parent.ports_world[spec.parent_port_index]
                conn_origin = parent_port.position
                conn_dir = -parent_port.direction

                frame = frame_from_origin_and_z(conn_origin, conn_dir)
                geom = build_segment_geom(spec)
                pts_w, faces, ports_w = segment_to_world(geom, frame)

                built[spec.id] = BuiltSegment(
                    spec=spec,
                    geom_local=geom,
                    frame_world=frame,
                    points_world=pts_w,
                    faces=faces,
                    ports_world=ports_w,
                )
                progress = True

        if not progress:
            remaining = [s.id for s in specs if s.id not in built]
            raise RuntimeError(f"Could not resolve parents for segments: {remaining}")

    return built


def merge_built_segments(built: Dict[str, BuiltSegment]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge all built segments into a single vertex/face array.

    Parameters
    ----------
    built : dict
        Mapping from segment id to BuiltSegment.

    Returns
    -------
    points_merged : (N, 3) float
    faces_merged : (M, 3) int
    """
    all_points = []
    all_faces = []
    offset = 0

    for seg_id in sorted(built.keys()):
        seg = built[seg_id]
        pts = seg.points_world
        faces = seg.faces

        all_points.append(pts)
        all_faces.append(faces + offset)
        offset += pts.shape[0]

    points_merged = np.vstack(all_points)
    faces_merged = np.vstack(all_faces)
    return points_merged, faces_merged


def load_specs_from_yaml(path: str) -> List[SegmentSpec]:
    """
    Load a list of SegmentSpec objects from a YAML file.

    The YAML file should contain a top-level list of mappings with keys:
      - id: str
      - kind: str ("pipe", "y2", "y3", etc.)
      - params: mapping of parameter names to values
      - parent_id: str or null
      - parent_port_index: int or null
      - meta: optional mapping for metadata (labels, regions, generation, etc.)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"YAML file {path!r} must contain a list at top level")

    specs: List[SegmentSpec] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each segment entry in YAML must be a mapping/dict")

        seg_id = item.get("id")
        kind = item.get("kind")
        params = item.get("params", {}) or {}
        parent_id = item.get("parent_id", None)
        parent_port_index = item.get("parent_port_index", None)
        meta = item.get("meta", None)

        if seg_id is None or kind is None:
            raise ValueError(f"Segment entry missing 'id' or 'kind': {item!r}")
        if not isinstance(params, dict):
            raise ValueError(f"'params' for segment {seg_id!r} must be a mapping")
        if meta is not None and not isinstance(meta, dict):
            raise ValueError(f"'meta' for segment {seg_id!r} must be a mapping if provided")

        specs.append(
            SegmentSpec(
                id=str(seg_id),
                kind=str(kind),
                params=dict(params),
                parent_id=parent_id,
                parent_port_index=parent_port_index,
                meta=meta,
            )
        )

    return specs


def make_example_specs() -> List[SegmentSpec]:
    """
    Create a small example airway tree specification:

        root pipe
          -> Y2 bifurcation
             -> left child: pipe
             -> right child: Y3 trifurcation
                -> three terminal pipes
    """
    specs: List[SegmentSpec] = []

    specs.append(
        SegmentSpec(
            id="root",
            kind="pipe",
            params=dict(length=6.0, d_in=2.0, d_out=2.0),
            parent_id=None,
            parent_port_index=None,
        )
    )

    specs.append(
        SegmentSpec(
            id="Y2_main",
            kind="y2",
            params=dict(
                length_trunk=6.0,
                length_child1=4.0,
                length_child2=4.0,
                d_trunk=2.0,
                d_child1=2.0,
                d_child2=2.0,
                theta1_deg=45.0,
                phi1_deg=0.0,
                theta2_deg=45.0,
                phi2_deg=120.0,
            ),
            parent_id="root",
            parent_port_index=1,
        )
    )

    specs.append(
        SegmentSpec(
            id="pipe_left",
            kind="pipe",
            params=dict(length=4.0, d_in=2.0, d_out=2.0),
            parent_id="Y2_main",
            parent_port_index=2,
        )
    )

    specs.append(
        SegmentSpec(
            id="Y3_distal",
            kind="y3",
            params=dict(
                length_trunk=4.0,
                length_child1=3.0,
                length_child2=3.0,
                length_child3=3.0,
                d_trunk=2.0,
                d_child1=2.0,
                d_child2=2.0,
                d_child3=2.0,
                theta1_deg=45.0,
                phi1_deg=0.0,
                theta2_deg=45.0,
                phi2_deg=120.0,
                theta3_deg=45.0,
                phi3_deg=240.0,
            ),
            parent_id="Y2_main",
            parent_port_index=1,
        )
    )

    for i in range(3):
        specs.append(
            SegmentSpec(
                id=f"term_pipe_{i+1}",
                kind="pipe",
                params=dict(length=3.0, d_in=2.0, d_out=2.0),
                parent_id="Y3_distal",
                parent_port_index=i + 1,
            )
        )

    return specs


def main():
    # Path to a YAML file describing the airway tree.
    # You can change this to point to any other tree specification.
    yaml_path = "trees/stenosis_test.yaml"

    specs = load_specs_from_yaml(yaml_path)
    if not specs:
        raise RuntimeError(f"No segment specs loaded from {yaml_path!r}")

    root_origin = np.array([0.0, 0.0, 0.0])
    root_z_world = np.array([0.0, 0.0, -1.0])

    built = build_tree(specs, root_origin=root_origin, root_z_world=root_z_world)
    points_merged, faces_merged = merge_built_segments(built)

    print("Built segments:", list(sorted(built.keys())))
    print("Merged points:", points_merged.shape[0])
    print("Merged faces:", faces_merged.shape[0])

    faces_flat = np.hstack(
        [np.full((faces_merged.shape[0], 1), 3, dtype=int), faces_merged]
    ).ravel()
    mesh = pv.PolyData(points_merged, faces_flat)

    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_text(
        f"Airway tree from YAML: {yaml_path}",
        font_size=12,
    )
    plotter.add_mesh(
        mesh,
        color="lightblue",
        smooth_shading=True,
        show_edges=True,
    )
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error while building or visualizing airway tree:", e)
