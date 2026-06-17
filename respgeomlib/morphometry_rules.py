from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import yaml


@dataclass
class ChildBranch:
    diameter: float
    length: float
    theta_deg: float
    phi_deg: float


def length_from_diameter(d: float, L_over_D: float = 3.0) -> float:
    """Return airway length given diameter d using L ≈ L_over_D * d."""
    d_safe = max(float(d), 0.0)
    L_over_D_safe = max(float(L_over_D), 0.0)
    if d_safe <= 0.0 or L_over_D_safe <= 0.0:
        return 0.0
    return d_safe * L_over_D_safe


def murray_child_diameters(
    parent_d: float,
    gamma: float = 3.0,
    asymmetry: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute two child diameters (d1, d2) from a parent diameter using a
    Murray's law type relation:

        parent_d**gamma ≈ d1**gamma + d2**gamma

    If asymmetry = 0.0, children are symmetric: d1 = d2.
    If asymmetry > 0, one branch is larger and the other smaller, e.g.
    controlled by an asymmetry fraction of the symmetric diameter.
    """
    eps = 1e-9
    parent_d_safe = max(float(parent_d), eps)
    gamma_safe = max(float(gamma), eps)

    parent_pow = parent_d_safe ** gamma_safe
    d_sym = (parent_pow / 2.0) ** (1.0 / gamma_safe)

    asym_safe = max(float(asymmetry), 0.0)
    # Prevent the large child from exceeding the parent_d_safe contribution.
    max_asym = max(parent_d_safe / d_sym - 1.0, 0.0)
    asym_used = np.clip(asym_safe, 0.0, max_asym * 0.999)

    d1 = d_sym * (1.0 + asym_used)
    remaining = max(parent_pow - d1 ** gamma_safe, eps)
    d2 = remaining ** (1.0 / gamma_safe)

    return float(d1), float(d2)


def weibel_like_children(
    parent_d: float,
    base_theta_deg: float = 35.0,
    base_phi_deg: float = 0.0,
    asymmetry: float = 0.0,
    L_over_D: float = 3.0,
) -> Tuple[ChildBranch, ChildBranch]:
    """
    Given a parent diameter, return two ChildBranch objects with:
      - diameters from murray_child_diameters,
      - lengths from length_from_diameter,
      - branching angles around the parent axis (theta, phi).

    For simplicity, use:
      child1: (theta=base_theta_deg,   phi=base_phi_deg)
      child2: (theta=base_theta_deg,   phi=base_phi_deg + 60)
    """
    d1, d2 = murray_child_diameters(parent_d, asymmetry=asymmetry)
    l1 = length_from_diameter(d1, L_over_D=L_over_D)
    l2 = length_from_diameter(d2, L_over_D=L_over_D)

    child1 = ChildBranch(diameter=d1, length=l1, theta_deg=base_theta_deg, phi_deg=base_phi_deg)
    child2 = ChildBranch(
        diameter=d2,
        length=l2,
        theta_deg=base_theta_deg,
        phi_deg=base_phi_deg + 60.0,
    )
    return child1, child2


def generate_binary_subtree(
    root_id: str,
    parent_id: str,
    parent_port_index: int,
    root_d: float,
    generations: int,
    side: str = "right",
    base_theta_deg: float = 35.0,
    base_phi_deg: float = 0.0,
    asymmetry: float = 0.1,
    L_over_D: float = 3.0,
) -> List[dict]:
    """
    Generate a symmetric (or mildly asymmetric) binary subtree of airway
    segments using only 'y2' and 'pipe' kinds, starting from a parent
    connection.

    Parameters
    ----------
    root_id : str
        ID of the first Y-segment in this subtree.
    parent_id : str
        ID of the segment this subtree attaches to.
    parent_port_index : int
        Port index of the parent segment where this subtree connects.
    root_d : float
        Diameter of the root trunk (cm).
    generations : int
        Number of bifurcation generations (Y2 levels) in this subtree.
    side : str
        "right" or "left"; can be used to bias phi angles.

    Returns
    -------
    segments : list of dict
        Each dict matches the YAML SegmentSpec structure with keys:
        id, kind, params, parent_id, parent_port_index.
    """
    if generations <= 0:
        return []

    segments: List[dict] = []
    side_norm = (side or "").lower()
    phi_bias = base_phi_deg + (180.0 if side_norm == "left" else 0.0)

    base_stem = root_id[2:] if root_id.startswith("Y_") else root_id

    def make_y_id(path_suffix: str) -> str:
        if path_suffix == "":
            return root_id
        return f"Y_{base_stem}_{path_suffix}"

    def make_pipe_id(path_suffix: str) -> str:
        if path_suffix == "":
            return base_stem
        return f"{base_stem}_{path_suffix}"

    def add_generation(path: str, parent_seg_id: str, parent_port: int, trunk_d: float, gen_left: int) -> None:
        y_id = make_y_id(path)
        child1, child2 = weibel_like_children(
            trunk_d,
            base_theta_deg=base_theta_deg,
            base_phi_deg=phi_bias,
            asymmetry=asymmetry,
            L_over_D=L_over_D,
        )
        trunk_length = length_from_diameter(trunk_d, L_over_D=L_over_D)

        segments.append(
            {
                "id": y_id,
                "kind": "y2",
                "params": {
                    "length_trunk": trunk_length,
                    "length_child1": child1.length,
                    "length_child2": child2.length,
                    "d_trunk": trunk_d,
                    "d_child1": child1.diameter,
                    "d_child2": child2.diameter,
                    "theta1_deg": child1.theta_deg,
                    "phi1_deg": child1.phi_deg,
                    "theta2_deg": child2.theta_deg,
                    "phi2_deg": child2.phi_deg,
                },
                "parent_id": parent_seg_id,
                "parent_port_index": parent_port,
            }
        )

        if gen_left > 1:
            add_generation(path + "a", y_id, 1, child1.diameter, gen_left - 1)
            add_generation(path + "b", y_id, 2, child2.diameter, gen_left - 1)
        else:
            for idx, (child, suffix) in enumerate(((child1, "a"), (child2, "b")), start=1):
                pipe_id = make_pipe_id(path + suffix)
                d_out = max(child.diameter * 0.85, 1e-6)
                segments.append(
                    {
                        "id": pipe_id,
                        "kind": "pipe",
                        "params": {
                            "length": child.length,
                            "d_in": child.diameter,
                            "d_out": d_out,
                        },
                        "parent_id": y_id,
                        "parent_port_index": idx,
                    }
                )

    add_generation(path="", parent_seg_id=parent_id, parent_port=parent_port_index, trunk_d=root_d, gen_left=generations)
    return segments


def main() -> None:
    segments = generate_binary_subtree(
        root_id="Y_right_subtree_G3",
        parent_id="Y_right_lobes",
        parent_port_index=1,
        root_d=1.0,
        generations=3,
        side="right",
    )
    with open("trees/right_lung_subtree_auto.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(segments, f, sort_keys=False)


if __name__ == "__main__":
    main()
