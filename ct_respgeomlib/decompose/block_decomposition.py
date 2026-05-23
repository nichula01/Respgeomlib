from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

from ct_respgeomlib.graph.airway_graph import AirwayGraph, AirwayEdge
from ct_respgeomlib.graph.shared_ports import SharedPortGraph, SharedPort


@dataclass
class FittedBlock:
    """
    A semantic CT-RespGeomLib block.

    This is not yet the final mesh.
    It is the fitted parametric representation that will later be exported
    to RespGeomLib YAML or directly regenerated as clean geometry.
    """
    id: str
    block_type: str
    input_ports: List[str]
    output_ports: List[str]
    source_edges: List[str] = field(default_factory=list)
    source_node: Optional[str] = None
    label: Optional[str] = None
    params: Dict = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "block_type": self.block_type,
            "input_ports": self.input_ports,
            "output_ports": self.output_ports,
            "source_edges": self.source_edges,
            "source_node": self.source_node,
            "label": self.label,
            "params": self.params,
            "meta": self.meta,
        }


@dataclass
class BlockDecomposition:
    blocks: Dict[str, FittedBlock] = field(default_factory=dict)

    def add_block(self, block: FittedBlock) -> None:
        if block.id in self.blocks:
            raise ValueError(f"Duplicate block id: {block.id}")
        self.blocks[block.id] = block

    def to_dict(self) -> Dict:
        return {
            "num_blocks": len(self.blocks),
            "blocks": {bid: b.to_dict() for bid, b in self.blocks.items()},
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def point_arc_length_on_edge(edge: AirwayEdge, point: np.ndarray) -> float:
    """
    Approximate arc-length coordinate of a point on/near an edge centerline.

    Finds nearest projection onto the polyline and returns distance from edge start.
    """
    pts = np.asarray(edge.centerline, dtype=float)
    point = np.asarray(point, dtype=float)

    if pts.shape[0] < 2:
        return 0.0

    segs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(segs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])

    best_s = 0.0
    best_d2 = float("inf")

    for i, (a, v, L) in enumerate(zip(pts[:-1], segs, seg_lens)):
        if L <= 1e-12:
            continue
        t = float(np.dot(point - a, v) / (L * L))
        t = float(np.clip(t, 0.0, 1.0))
        q = a + t * v
        d2 = float(np.sum((point - q) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(cum[i] + t * L)

    return best_s


def edge_is_curved(
    edge: AirwayEdge,
    length_ratio_threshold: float = 1.03,
    max_deviation_fraction_threshold: float = 0.05,
) -> Tuple[bool, Dict]:
    """
    Decide whether an edge should be represented as a straight pipe or curved pipe.

    A branch is considered curved if:
    - polyline length is much larger than end-to-end chord length, or
    - centerline deviates significantly from the straight chord.
    """
    pts = np.asarray(edge.centerline, dtype=float)

    if pts.shape[0] < 3:
        return False, {
            "length": edge.length,
            "chord": edge.length,
            "length_ratio": 1.0,
            "max_deviation": 0.0,
            "max_deviation_fraction": 0.0,
        }

    p0 = pts[0]
    p1 = pts[-1]
    chord_vec = p1 - p0
    chord = float(np.linalg.norm(chord_vec))

    if chord <= 1e-12:
        return True, {
            "length": edge.length,
            "chord": chord,
            "length_ratio": float("inf"),
            "max_deviation": 0.0,
            "max_deviation_fraction": float("inf"),
        }

    u = chord_vec / chord

    deviations = []
    for p in pts:
        proj = p0 + np.dot(p - p0, u) * u
        deviations.append(float(np.linalg.norm(p - proj)))

    max_dev = float(max(deviations))
    length_ratio = float(edge.length / chord)
    max_dev_fraction = float(max_dev / max(edge.length, 1e-12))

    is_curved = (
        length_ratio > length_ratio_threshold
        or max_dev_fraction > max_deviation_fraction_threshold
    )

    return is_curved, {
        "length": float(edge.length),
        "chord": chord,
        "length_ratio": length_ratio,
        "max_deviation": max_dev,
        "max_deviation_fraction": max_dev_fraction,
    }


def ports_on_edge(port_graph: SharedPortGraph, edge_id: str) -> List[SharedPort]:
    return [p for p in port_graph.ports.values() if p.owner_edge == edge_id]


def build_pipe_blocks_for_edge(
    graph: AirwayGraph,
    port_graph: SharedPortGraph,
    edge: AirwayEdge,
) -> List[FittedBlock]:
    """
    Build pipe/curved-pipe blocks between consecutive ports on one CT edge.
    """
    ports = ports_on_edge(port_graph, edge.id)

    if len(ports) < 2:
        return []

    ports_sorted = sorted(
        ports,
        key=lambda p: point_arc_length_on_edge(edge, p.xyz),
    )

    blocks = []
    curved, curvature_meta = edge_is_curved(edge)

    for i in range(len(ports_sorted) - 1):
        p0 = ports_sorted[i]
        p1 = ports_sorted[i + 1]

        s0 = point_arc_length_on_edge(edge, p0.xyz)
        s1 = point_arc_length_on_edge(edge, p1.xyz)

        if s1 <= s0:
            continue

        block_type = "curved_pipe" if curved else "straight_pipe"

        block = FittedBlock(
            id=f"block_{block_type}_{edge.id}_{i}",
            block_type=block_type,
            input_ports=[p0.id],
            output_ports=[p1.id],
            source_edges=[edge.id],
            source_node=None,
            label=edge.label,
            params={
                "length": float(s1 - s0),
                "d_in": float(2.0 * p0.radius),
                "d_out": float(2.0 * p1.radius),
                "radius_in": float(p0.radius),
                "radius_out": float(p1.radius),
                "arc_length_start": float(s0),
                "arc_length_end": float(s1),
            },
            meta={
                "edge_total_length": float(edge.length),
                "curvature": curvature_meta,
            },
        )
        blocks.append(block)

    return blocks


def build_junction_blocks(
    graph: AirwayGraph,
    port_graph: SharedPortGraph,
) -> List[FittedBlock]:
    """
    Build Y2/Y3/general-junction blocks from branching nodes.
    """
    blocks = []

    for node in graph.branch_nodes():
        children = graph.children_of(node.id)
        parent_edge = graph.parent_edge_of(node.id)

        if parent_edge is None:
            continue

        parent_ports = [
            p for p in port_graph.ports.values()
            if p.owner_node == node.id
            and p.owner_edge == parent_edge.id
            and p.kind == "junction_parent_cut"
        ]

        child_ports = []
        for e in children:
            matches = [
                p for p in port_graph.ports.values()
                if p.owner_node == node.id
                and p.owner_edge == e.id
                and p.kind == "junction_child_cut"
            ]
            child_ports.extend(matches)

        if len(parent_ports) != 1:
            raise ValueError(f"Expected one parent cut port for node {node.id}, found {len(parent_ports)}")

        if len(child_ports) == 2:
            block_type = "Y2"
        elif len(child_ports) == 3:
            block_type = "Y3"
        else:
            block_type = "general_junction"

        parent_port = parent_ports[0]

        child_edge_ids = [e.id for e in children]
        all_source_edges = [parent_edge.id] + child_edge_ids

        params = {
            "num_children": len(child_ports),
            "parent_diameter": float(2.0 * parent_port.radius),
            "child_diameters": [float(2.0 * p.radius) for p in child_ports],
            "junction_center_xyz": np.asarray(node.xyz, dtype=float).round(6).tolist(),
            "junction_radius": float(node.radius),
        }

        block = FittedBlock(
            id=f"block_{block_type}_{node.id}",
            block_type=block_type,
            input_ports=[parent_port.id],
            output_ports=[p.id for p in child_ports],
            source_edges=all_source_edges,
            source_node=node.id,
            label=node.label,
            params=params,
            meta={
                "node_generation": node.generation,
                "node_label": node.label,
            },
        )

        blocks.append(block)

    return blocks


def build_block_decomposition(
    graph: AirwayGraph,
    port_graph: SharedPortGraph,
) -> BlockDecomposition:
    """
    Convert an anatomical airway graph and shared-port graph into fitted blocks.
    """
    dec = BlockDecomposition()

    for edge in graph.edges.values():
        for block in build_pipe_blocks_for_edge(graph, port_graph, edge):
            dec.add_block(block)

    for block in build_junction_blocks(graph, port_graph):
        dec.add_block(block)

    return dec
