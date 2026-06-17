import numpy as np
import pyvista as pv

from typing import Optional, Dict, Any

from respgeomlib.tree_builder import load_specs_from_yaml, build_tree


def color_for_segment(
    seg_id: str,
    meta: Optional[Dict[str, Any]],
) -> str:
    """
    Choose a colour name for a segment based on its meta-data and id.
    Priority:
      - use meta["region"] and meta["side"] if available
      - fall back to heuristics based on seg_id
    """
    sid = seg_id.lower()

    if meta:
        region = str(meta.get("region", "") or "").lower()
        side = str(meta.get("side", "") or "").lower()

        if region == "trachea":
            return "lightgray"

        if side == "right":
            if "upper" in region:
                return "orange"
            if "middle" in region:
                return "gold"
            if "lower" in region:
                return "red"
        elif side == "left":
            if "upper" in region:
                return "deepskyblue"
            if "lower" in region:
                return "royalblue"
        elif side == "central" and region and region != "trachea":
            return "lightcoral"

    if sid.startswith("y_"):
        return "lightpink"
    if "right_" in sid:
        return "coral"
    if "left_" in sid:
        return "lightskyblue"
    if sid == "trachea":
        return "lightgray"
    return "lightgreen"


def main():
    yaml_path = "trees/human_central_v2_weibel.yaml"

    specs = load_specs_from_yaml(yaml_path)
    root_origin = np.array([0.0, 0.0, 0.0])
    root_z_world = np.array([0.0, 0.0, -1.0])

    built = build_tree(specs, root_origin=root_origin, root_z_world=root_z_world)

    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_text(
        "Human central airways (Weibel-inspired) – coloured by side/lobe",
        font_size=12,
    )

    for seg_id, seg in sorted(built.items()):
        pts = seg.points_world
        faces = seg.faces
        faces_flat = np.hstack([np.full((faces.shape[0], 1), 3, dtype=int), faces]).ravel()
        mesh = pv.PolyData(pts, faces_flat)
        meta = seg.spec.meta if hasattr(seg, "spec") else None
        color = color_for_segment(seg_id, meta)
        plotter.add_mesh(mesh, color=color, smooth_shading=True, show_edges=False)

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


if __name__ == "__main__":
    main()
