from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import json
import numpy as np

from ct_respgeomlib.graph.airway_graph import AirwayGraph, AirwayEdge


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def frame_from_normal(z_axis: np.ndarray) -> np.ndarray:
    """
    Construct a stable local coordinate frame whose z-axis is the port normal.
    Returns 3x3 rotation matrix [x y z].
    """
    z = normalize(z_axis)

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, z)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])

    x = ref - np.dot(ref, z) * z
    x = normalize(x)
    y = normalize(np.cross(z, x))

    return np.column_stack([x, y, z])


def sample_edge_at_distance(edge: AirwayEdge, s: float) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Sample point, radius, and tangent along an edge at arc-length distance s
    from the edge start.
    """
    pts = np.asarray(edge.centerline, dtype=float)
    radii = np.asarray(edge.radii, dtype=float)

    if pts.shape[0] < 2:
        return pts[0], float(radii[0]), np.array([0.0, 0.0, 1.0])

    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(np.sum(seg_len))

    if total <= 1e-12:
        return pts[0], float(radii[0]), np.array([0.0, 0.0, 1.0])

    s = float(np.clip(s, 0.0, total))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    idx = int(np.searchsorted(cum, s, side="right") - 1)
    idx = min(idx, len(seg_len) - 1)

    local_len = seg_len[idx]
    if local_len <= 1e-12:
        t = 0.0
        tangent = np.array([0.0, 0.0, 1.0])
    else:
        t = (s - cum[idx]) / local_len
        tangent = seg[idx] / local_len

    point = (1.0 - t) * pts[idx] + t * pts[idx + 1]
    radius = (1.0 - t) * radii[idx] + t * radii[idx + 1]

    return point, float(radius), normalize(tangent)


@dataclass
class SharedPort:
    """
    A shared connection interface used by fitted airway blocks.

    xyz: port center
    normal: local +z direction of the port frame
    radius: local airway radius at this cut/interface
    frame: 3x3 local coordinate frame [x y z]
    """
    id: str
    xyz: np.ndarray
    normal: np.ndarray
    radius: float
    kind: str
    label: Optional[str] = None
    owner_node: Optional[str] = None
    owner_edge: Optional[str] = None
    frame: np.ndarray = field(default_factory=lambda: np.eye(3))
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "xyz": np.asarray(self.xyz).round(6).tolist(),
            "normal": np.asarray(self.normal).round(6).tolist(),
            "radius": float(self.radius),
            "diameter": float(2.0 * self.radius),
            "kind": self.kind,
            "label": self.label,
            "owner_node": self.owner_node,
            "owner_edge": self.owner_edge,
            "frame": np.asarray(self.frame).round(6).tolist(),
            "meta": self.meta,
        }


@dataclass
class SharedPortGraph:
    ports: Dict[str, SharedPort] = field(default_factory=dict)

    def add_port(self, port: SharedPort) -> None:
        if port.id in self.ports:
            raise ValueError(f"Duplicate shared port id: {port.id}")

        port.xyz = np.asarray(port.xyz, dtype=float)
        port.normal = normalize(port.normal)
        port.frame = frame_from_normal(port.normal)
        port.radius = float(port.radius)

        self.ports[port.id] = port

    def to_dict(self) -> Dict:
        return {
            "num_ports": len(self.ports),
            "ports": {pid: p.to_dict() for pid, p in self.ports.items()},
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def safe_cut_distance(edge: AirwayEdge, preferred: float) -> float:
    """
    Keep cut point inside the branch.
    For very short branches, use at most 40% of branch length.
    """
    return float(min(preferred, 0.4 * edge.length))


def build_shared_port_graph(
    graph: AirwayGraph,
    cut_radius_factor: float = 2.5,
    min_cut_distance: float = 1e-3,
) -> SharedPortGraph:
    """
    Build shared cut ports from an airway graph.

    For a bifurcation/trifurcation node:
    - one parent cut port is placed upstream on the parent branch
    - one child cut port is placed downstream on each child branch

    These ports become exact interfaces for later fitted blocks.
    """
    pg = SharedPortGraph()

    # Root inlet ports
    for root in graph.root_nodes():
        children = graph.children_of(root.id)
        for e in children:
            p, r, tangent = sample_edge_at_distance(e, 0.0)
            pg.add_port(
                SharedPort(
                    id=f"port_inlet_{root.id}_{e.id}",
                    xyz=p,
                    normal=-tangent,
                    radius=r,
                    kind="inlet",
                    label=f"inlet:{root.label or root.id}",
                    owner_node=root.id,
                    owner_edge=e.id,
                )
            )

    # Junction cut ports
    for node in graph.branch_nodes():
        preferred = max(cut_radius_factor * float(node.radius), min_cut_distance)

        parent_edge = graph.parent_edge_of(node.id)
        if parent_edge is not None:
            d = safe_cut_distance(parent_edge, preferred)
            p, r, tangent = sample_edge_at_distance(parent_edge, parent_edge.length - d)

            pg.add_port(
                SharedPort(
                    id=f"port_junction_{node.id}_parent_{parent_edge.id}",
                    xyz=p,
                    normal=-tangent,
                    radius=r,
                    kind="junction_parent_cut",
                    label=f"junction-parent:{node.label or node.id}",
                    owner_node=node.id,
                    owner_edge=parent_edge.id,
                    meta={"cut_distance_from_junction": d},
                )
            )

        for child_edge in graph.children_of(node.id):
            d = safe_cut_distance(child_edge, preferred)
            p, r, tangent = sample_edge_at_distance(child_edge, d)

            pg.add_port(
                SharedPort(
                    id=f"port_junction_{node.id}_child_{child_edge.id}",
                    xyz=p,
                    normal=tangent,
                    radius=r,
                    kind="junction_child_cut",
                    label=f"junction-child:{node.label or node.id}",
                    owner_node=node.id,
                    owner_edge=child_edge.id,
                    meta={"cut_distance_from_junction": d},
                )
            )

    # Outlet ports
    for outlet in graph.outlet_nodes():
        parent_edge = graph.parent_edge_of(outlet.id)
        if parent_edge is None:
            continue

        p, r, tangent = sample_edge_at_distance(parent_edge, parent_edge.length)

        pg.add_port(
            SharedPort(
                id=f"port_outlet_{outlet.id}_{parent_edge.id}",
                xyz=p,
                normal=tangent,
                radius=r,
                kind="outlet",
                label=f"outlet:{outlet.label or outlet.id}",
                owner_node=outlet.id,
                owner_edge=parent_edge.id,
            )
        )

    return pg
