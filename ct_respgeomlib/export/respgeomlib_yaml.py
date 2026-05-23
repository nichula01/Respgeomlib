from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import math
import numpy as np

from ct_respgeomlib.graph.shared_ports import SharedPortGraph
from ct_respgeomlib.decompose.block_decomposition import BlockDecomposition, FittedBlock


def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def stable_frame_from_z(z_axis: np.ndarray) -> np.ndarray:
    z = unit(z_axis)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, z))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    x = ref - np.dot(ref, z) * z
    x = unit(x)
    y = unit(np.cross(z, x))
    return np.column_stack([x, y, z])


def direction_to_theta_phi_deg(direction_world: np.ndarray, local_R: np.ndarray) -> Tuple[float, float]:
    """
    Convert a world-space child direction into local RespGeomLib theta/phi.
    local_R has columns [x, y, z].
    """
    d_world = unit(direction_world)
    d_local = local_R.T @ d_world
    d_local = unit(d_local)

    z = float(np.clip(d_local[2], -1.0, 1.0))
    theta = math.degrees(math.acos(z))
    phi = math.degrees(math.atan2(float(d_local[1]), float(d_local[0])))

    return float(theta), float(phi)


def sanitize_id(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def block_output_port_index(block: FittedBlock, output_port_id: str) -> int:
    """
    Existing RespGeomLib segment convention:
    - pipe output port index = 1
    - Y2 child ports = 1, 2
    - Y3 child ports = 1, 2, 3
    """
    if block.block_type in {"straight_pipe", "curved_pipe"}:
        return 1

    if block.block_type in {"Y2", "Y3", "general_junction"}:
        if output_port_id not in block.output_ports:
            raise ValueError(f"Output port {output_port_id} not in block {block.id}")
        return block.output_ports.index(output_port_id) + 1

    raise ValueError(f"Unknown block type for output-port mapping: {block.block_type}")


def yaml_scalar(x):
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.8g}"
    return str(x)


def dump_simple_yaml_list(items: List[Dict], path: str) -> None:
    """
    Minimal YAML writer for the simple RespGeomLib list-of-segments format.
    Avoids requiring PyYAML.
    """
    lines = []

    for item in items:
        lines.append(f"- id: {item['id']}")
        lines.append(f"  kind: {item['kind']}")
        lines.append("  params:")
        for k, v in item["params"].items():
            lines.append(f"    {k}: {yaml_scalar(v)}")
        lines.append(f"  parent_id: {yaml_scalar(item.get('parent_id'))}")
        lines.append(f"  parent_port_index: {yaml_scalar(item.get('parent_port_index'))}")

        meta = item.get("meta", None)
        if meta:
            lines.append("  meta:")
            for k, v in meta.items():
                if isinstance(v, list):
                    lines.append(f"    {k}: {v}")
                else:
                    lines.append(f"    {k}: {yaml_scalar(v)}")

        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def find_root_block(dec: BlockDecomposition, pg: SharedPortGraph) -> FittedBlock:
    """
    Root block = block whose input port is an inlet port.
    """
    root_blocks = []
    for block in dec.blocks.values():
        if not block.input_ports:
            continue
        p = pg.ports[block.input_ports[0]]
        if p.kind == "inlet":
            root_blocks.append(block)

    if len(root_blocks) != 1:
        raise ValueError(f"Expected exactly one root block, found {len(root_blocks)}")

    return root_blocks[0]


def build_parent_maps(dec: BlockDecomposition) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
    - input_port_to_block: port id -> block id that consumes it
    - output_port_to_block: port id -> block id that produces it
    """
    input_port_to_block = {}
    output_port_to_block = {}

    for bid, block in dec.blocks.items():
        for p in block.input_ports:
            if p in input_port_to_block:
                raise ValueError(f"Port {p} is consumed by multiple blocks")
            input_port_to_block[p] = bid

        for p in block.output_ports:
            if p in output_port_to_block:
                raise ValueError(f"Port {p} is produced by multiple blocks")
            output_port_to_block[p] = bid

    return input_port_to_block, output_port_to_block


def segment_params_from_block(block: FittedBlock, pg: SharedPortGraph) -> Tuple[str, Dict]:
    """
    Convert one fitted CT-RespGeomLib block into existing RespGeomLib YAML segment params.
    """
    bt = block.block_type

    if bt in {"straight_pipe", "curved_pipe"}:
        # For now, curved pipe is exported as pipe.
        # Later we will add true spline-pipe support into RespGeomLib.
        kind = "pipe"
        params = {
            "length": float(block.params["length"]),
            "d_in": float(block.params["d_in"]),
            "d_out": float(block.params["d_out"]),
        }
        return kind, params

    if bt in {"Y2", "Y3"}:
        parent_port = pg.ports[block.input_ports[0]]
        center = np.asarray(block.params["junction_center_xyz"], dtype=float)

        parent_xyz = np.asarray(parent_port.xyz, dtype=float)
        trunk_vec = center - parent_xyz
        length_trunk = float(np.linalg.norm(trunk_vec))
        if length_trunk <= 1e-12:
            raise ValueError(f"Invalid trunk length for junction block {block.id}")

        local_R = stable_frame_from_z(trunk_vec)

        child_lengths = []
        child_diams = []
        child_angles = []

        for child_port_id in block.output_ports:
            cp = pg.ports[child_port_id]
            child_xyz = np.asarray(cp.xyz, dtype=float)
            child_vec = child_xyz - center
            child_len = float(np.linalg.norm(child_vec))
            if child_len <= 1e-12:
                raise ValueError(f"Invalid child length for port {child_port_id}")

            theta, phi = direction_to_theta_phi_deg(child_vec, local_R)

            child_lengths.append(child_len)
            child_diams.append(float(2.0 * cp.radius))
            child_angles.append((theta, phi))

        if bt == "Y2":
            kind = "y2"
            params = {
                "length_trunk": length_trunk,
                "length_child1": child_lengths[0],
                "length_child2": child_lengths[1],
                "d_trunk": float(2.0 * parent_port.radius),
                "d_child1": child_diams[0],
                "d_child2": child_diams[1],
                "theta1_deg": child_angles[0][0],
                "phi1_deg": child_angles[0][1],
                "theta2_deg": child_angles[1][0],
                "phi2_deg": child_angles[1][1],
                "blend_length": 0.0,
            }
            return kind, params

        if bt == "Y3":
            kind = "y3"
            params = {
                "length_trunk": length_trunk,
                "length_child1": child_lengths[0],
                "length_child2": child_lengths[1],
                "length_child3": child_lengths[2],
                "d_trunk": float(2.0 * parent_port.radius),
                "d_child1": child_diams[0],
                "d_child2": child_diams[1],
                "d_child3": child_diams[2],
                "theta1_deg": child_angles[0][0],
                "phi1_deg": child_angles[0][1],
                "theta2_deg": child_angles[1][0],
                "phi2_deg": child_angles[1][1],
                "theta3_deg": child_angles[2][0],
                "phi3_deg": child_angles[2][1],
                "blend_length": 0.0,
            }
            return kind, params

    raise ValueError(f"Cannot export unsupported block type: {bt}")


def export_to_respgeomlib_yaml(
    dec: BlockDecomposition,
    pg: SharedPortGraph,
    out_yaml: str,
) -> List[Dict]:
    """
    Export fitted block decomposition into existing RespGeomLib YAML format.

    This first version exports curved_pipe as standard pipe.
    Full spline/curved-pipe YAML support will be added later.
    """
    root = find_root_block(dec, pg)
    input_port_to_block, output_port_to_block = build_parent_maps(dec)

    # Build child adjacency: parent block -> list[(output_port_id, child_block_id)]
    children = {bid: [] for bid in dec.blocks.keys()}
    parent_info = {}

    for child_bid, child_block in dec.blocks.items():
        if child_bid == root.id:
            parent_info[child_bid] = (None, None)
            continue

        if not child_block.input_ports:
            raise ValueError(f"Non-root block has no input port: {child_bid}")

        input_port = child_block.input_ports[0]

        if input_port not in output_port_to_block:
            raise ValueError(f"Could not find parent block producing input port {input_port}")

        parent_bid = output_port_to_block[input_port]
        parent_block = dec.blocks[parent_bid]
        parent_port_index = block_output_port_index(parent_block, input_port)

        parent_info[child_bid] = (parent_bid, parent_port_index)
        children[parent_bid].append((input_port, child_bid))

    # DFS ordering from root
    ordered = []
    visited = set()

    def visit(bid):
        if bid in visited:
            return
        visited.add(bid)
        ordered.append(bid)

        # keep deterministic ordering
        for _, cbid in sorted(children.get(bid, []), key=lambda x: x[1]):
            visit(cbid)

    visit(root.id)

    if len(ordered) != len(dec.blocks):
        missing = set(dec.blocks.keys()) - set(ordered)
        raise ValueError(f"Some blocks were not reachable from root: {sorted(missing)}")

    items = []
    for bid in ordered:
        block = dec.blocks[bid]
        kind, params = segment_params_from_block(block, pg)

        parent_id, parent_port_index = parent_info[bid]

        seg_id = sanitize_id(block.id)

        item = {
            "id": seg_id,
            "kind": kind,
            "params": params,
            "parent_id": sanitize_id(parent_id) if parent_id is not None else None,
            "parent_port_index": parent_port_index,
            "meta": {
                "ct_block_type": block.block_type,
                "label": block.label,
                "source_node": block.source_node,
                "source_edges": block.source_edges,
            },
        }
        items.append(item)

    dump_simple_yaml_list(items, out_yaml)
    return items
