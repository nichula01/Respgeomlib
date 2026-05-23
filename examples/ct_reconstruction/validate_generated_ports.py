from pathlib import Path
import csv
import json
import numpy as np


def load_generated_ports(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "segment_id": r["segment_id"],
                "port_index": int(r["port_index"]),
                "xyz": np.array([float(r["x"]), float(r["y"]), float(r["z"])]),
                "normal": np.array([float(r["nx"]), float(r["ny"]), float(r["nz"])]),
            })
    return rows


def load_shared_ports(path):
    data = json.loads(Path(path).read_text())
    return data["ports"]


def nearest_generated_port(target_xyz, generated):
    best = None
    best_dist = float("inf")

    for g in generated:
        d = float(np.linalg.norm(target_xyz - g["xyz"]))
        if d < best_dist:
            best_dist = d
            best = g

    return best, best_dist


if __name__ == "__main__":
    shared_path = Path("outputs/ct_reconstruction/synthetic_shared_ports.json")
    generated_path = Path("outputs/ct_reconstruction/synthetic_ct_respgeomlib_ports.csv")
    out_path = Path("outputs/ct_reconstruction/synthetic_port_validation.csv")

    shared = load_shared_ports(shared_path)
    generated = load_generated_ports(generated_path)

    print("Shared ports:", len(shared))
    print("Generated ports:", len(generated))
    print()

    rows = []

    for pid, p in shared.items():
        target_xyz = np.array(p["xyz"], dtype=float)
        target_normal = np.array(p["normal"], dtype=float)

        nearest, dist = nearest_generated_port(target_xyz, generated)

        normal_dot = float(np.dot(
            target_normal / max(np.linalg.norm(target_normal), 1e-12),
            nearest["normal"] / max(np.linalg.norm(nearest["normal"]), 1e-12),
        ))

        rows.append({
            "shared_port_id": pid,
            "shared_kind": p["kind"],
            "target_x": target_xyz[0],
            "target_y": target_xyz[1],
            "target_z": target_xyz[2],
            "nearest_segment_id": nearest["segment_id"],
            "nearest_port_index": nearest["port_index"],
            "generated_x": nearest["xyz"][0],
            "generated_y": nearest["xyz"][1],
            "generated_z": nearest["xyz"][2],
            "position_error": dist,
            "normal_dot": normal_dot,
        })

        print(pid)
        print("  kind:", p["kind"])
        print("  nearest:", nearest["segment_id"], "port", nearest["port_index"])
        print("  position_error:", round(dist, 6))
        print("  normal_dot:", round(normal_dot, 6))
        print()

    with open(out_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = [r["position_error"] for r in rows]
    print("Mean position error:", float(np.mean(errors)))
    print("Max position error:", float(np.max(errors)))
    print("Saved:", out_path)
