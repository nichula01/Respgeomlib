from pathlib import Path
import csv
import numpy as np
import pyvista as pv

from tree_builder import load_specs_from_yaml, build_tree, merge_built_segments


def main():
    yaml_path = Path("outputs/ct_reconstruction/synthetic_ct_respgeomlib.yaml")
    out_dir = Path("outputs/ct_reconstruction")
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = load_specs_from_yaml(str(yaml_path))
    print("Loaded specs:", len(specs))
    for s in specs:
        print(" ", s.id, s.kind, "parent=", s.parent_id, "port=", s.parent_port_index)

    # Our synthetic CT graph grows along +Z.
    root_origin = np.array([0.0, 0.0, 0.0])
    root_z_world = np.array([0.0, 0.0, 1.0])

    built = build_tree(specs, root_origin=root_origin, root_z_world=root_z_world)
    points, faces = merge_built_segments(built)

    print("Built segments:", sorted(built.keys()))
    print("Merged points:", points.shape)
    print("Merged faces:", faces.shape)

    faces_flat = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=int), faces]
    ).ravel()

    mesh = pv.PolyData(points, faces_flat)
    mesh.clean(inplace=True)

    out_vtp = out_dir / "synthetic_ct_respgeomlib_mesh.vtp"
    out_stl = out_dir / "synthetic_ct_respgeomlib_mesh.stl"
    out_ports = out_dir / "synthetic_ct_respgeomlib_ports.csv"

    mesh.save(str(out_vtp))
    mesh.save(str(out_stl))

    with open(out_ports, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment_id",
            "port_index",
            "x", "y", "z",
            "nx", "ny", "nz",
        ])

        for seg_id, seg in built.items():
            for i, p in enumerate(seg.ports_world):
                writer.writerow([
                    seg_id,
                    i,
                    float(p.position[0]),
                    float(p.position[1]),
                    float(p.position[2]),
                    float(p.direction[0]),
                    float(p.direction[1]),
                    float(p.direction[2]),
                ])

    print("Saved:", out_vtp)
    print("Saved:", out_stl)
    print("Saved:", out_ports)


if __name__ == "__main__":
    main()
