from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class AirwayNode:
    """
    A node in a CT-derived airway centerline graph.

    This can represent:
    - inlet/root point
    - branch point
    - outlet/end point
    - intermediate sampled centerline point
    """
    id: str
    xyz: np.ndarray
    radius: float
    generation: Optional[int] = None
    label: Optional[str] = None
    meta: Dict = field(default_factory=dict)


@dataclass
class AirwayEdge:
    """
    A branch between two graph nodes.

    centerline: ordered Nx3 points from parent to child.
    radii: ordered N radius values corresponding to centerline points.
    """
    id: str
    parent: str
    child: str
    centerline: np.ndarray
    radii: np.ndarray
    generation: Optional[int] = None
    label: Optional[str] = None
    meta: Dict = field(default_factory=dict)

    @property
    def length(self) -> float:
        if self.centerline.shape[0] < 2:
            return 0.0
        diffs = np.diff(self.centerline, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    @property
    def d_in(self) -> float:
        return float(2.0 * self.radii[0])

    @property
    def d_out(self) -> float:
        return float(2.0 * self.radii[-1])


@dataclass
class AirwayGraph:
    """
    CT-derived anatomical airway graph.

    This is not yet a RespGeomLib block model.
    It stores the measured CT-derived topology, centerlines, and radii.
    """
    nodes: Dict[str, AirwayNode] = field(default_factory=dict)
    edges: Dict[str, AirwayEdge] = field(default_factory=dict)

    def add_node(self, node: AirwayNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        node.xyz = np.asarray(node.xyz, dtype=float)
        self.nodes[node.id] = node

    def add_edge(self, edge: AirwayEdge) -> None:
        if edge.id in self.edges:
            raise ValueError(f"Duplicate edge id: {edge.id}")
        if edge.parent not in self.nodes:
            raise ValueError(f"Missing parent node: {edge.parent}")
        if edge.child not in self.nodes:
            raise ValueError(f"Missing child node: {edge.child}")
        edge.centerline = np.asarray(edge.centerline, dtype=float)
        edge.radii = np.asarray(edge.radii, dtype=float)
        if edge.centerline.ndim != 2 or edge.centerline.shape[1] != 3:
            raise ValueError("centerline must have shape (N, 3)")
        if edge.radii.shape[0] != edge.centerline.shape[0]:
            raise ValueError("radii length must match centerline length")
        self.edges[edge.id] = edge

    def children_of(self, node_id: str) -> List[AirwayEdge]:
        return [e for e in self.edges.values() if e.parent == node_id]

    def parent_edge_of(self, node_id: str) -> Optional[AirwayEdge]:
        parents = [e for e in self.edges.values() if e.child == node_id]
        if len(parents) == 0:
            return None
        if len(parents) > 1:
            raise ValueError(f"Node {node_id} has multiple parent edges")
        return parents[0]

    def branch_nodes(self) -> List[AirwayNode]:
        return [n for n in self.nodes.values() if len(self.children_of(n.id)) >= 2]

    def outlet_nodes(self) -> List[AirwayNode]:
        return [n for n in self.nodes.values() if len(self.children_of(n.id)) == 0]

    def root_nodes(self) -> List[AirwayNode]:
        child_ids = {e.child for e in self.edges.values()}
        return [n for n in self.nodes.values() if n.id not in child_ids]
