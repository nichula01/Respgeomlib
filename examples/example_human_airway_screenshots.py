import numpy as np
import pyvista as pv
from pathlib import Path

from respgeomlib.tree_builder import load_specs_from_yaml, build_tree
from example_human_airway_colored import color_for_segment


def main() -> None:
    yaml_path = "trees/human_central_v2_weibel_with_right_subtree.yaml"
    specs = load_specs_from_yaml(yaml_path)
    root_origin = np.array([0.0, 0.0, 0.0])
    root_z_world = np.array([0.0, 0.0, -1.0])

    built = build_tree(specs, root_origin=root_origin, root_z_world=root_z_world)

    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")

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

    out_dir = Path("screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    screenshots = {
        "human_airway_front.png": "xy",
        "human_airway_side.png": "yz",
        "human_airway_top.png": "xz",
    }

    for fname, cam in screenshots.items():
        plotter.camera_position = cam
        plotter.show(screenshot=str(out_dir / fname), auto_close=False)

    plotter.close()


if __name__ == "__main__":
    main()
