from pathlib import Path
import csv
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from synthetic_airway_graph_demo import build_demo_graph
from ct_respgeomlib.graph.shared_ports import build_shared_port_graph
from ct_respgeomlib.decompose.block_decomposition import build_block_decomposition
from ct_respgeomlib.export.respgeomlib_yaml import (
    block_output_port_index,
    sanitize_id,
)


def load_generated_ports(path):
    ports = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["segment_id"], int(r["port_index"]))
            ports[key] = {
                "xyz": np.array([float(r["x"]), float(r["y"]), float(r["z"])]),
                "normal": np.array([float(r["nx"]), float(r["ny"]), float(r["nz"])]),
            }
    return ports


def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


if __name__ == "__main__":
    g = build_demo_graph()
    pg = build_shared_port_graph(g, cut_radius_factor=2.5)
    dec = build_block_decomposition(g, pg)

    generated_csv = Path("outputs/ct_reconstruction/synthetic_ct_respgeomlib_ports.csv")
    out_csv = Path("outputs/ct_reconstruction/synthetic_strict_interface_validation.csv")

    generated = load_generated_ports(generated_csv)

    # Expected occurrences:
    # each CT shared port may appear once for a boundary, or twice for a block-to-block interface.
    expected = {}

    for block in dec.blocks.values():
        seg_id = sanitize_id(block.id)

        for pid in block.input_ports:
            expected.setdefault(pid, []).append((seg_id, 0, "input"))

        for pid in block.output_ports:
            idx = block_output_port_index(block, pid)
            expected.setdefault(pid, []).append((seg_id, idx, "output"))

    rows = []
    all_position_errors = []
    all_interface_direction_dots = []
    failures = []

    print("Strict shared-port interface validation")
    print("=" * 50)

    for pid, occurrences in expected.items():
        target = pg.ports[pid]
        target_xyz = np.asarray(target.xyz, dtype=float)

        print()
        print(pid)
        print("  kind:", target.kind)
        print("  expected_occurrences:", len(occurrences))

        occurrence_normals = []
        occurrence_errors = []

        for seg_id, port_index, role in occurrences:
            key = (seg_id, port_index)

            if key not in generated:
                failures.append(f"Missing generated port: {key} for shared port {pid}")
                print("  MISSING:", key)
                continue

            gen = generated[key]
            err = float(np.linalg.norm(gen["xyz"] - target_xyz))
            occurrence_errors.append(err)
            all_position_errors.append(err)
            occurrence_normals.append(unit(gen["normal"]))

            rows.append({
                "shared_port_id": pid,
                "shared_kind": target.kind,
                "segment_id": seg_id,
                "port_index": port_index,
                "role": role,
                "position_error": err,
                "generated_nx": gen["normal"][0],
                "generated_ny": gen["normal"][1],
                "generated_nz": gen["normal"][2],
            })

            print(
                "  ",
                seg_id,
                "port",
                port_index,
                role,
                "position_error=",
                round(err, 9),
            )

        # If two blocks share the same port, their local port directions should oppose.
        if len(occurrence_normals) == 2:
            dot = float(np.dot(occurrence_normals[0], occurrence_normals[1]))
            all_interface_direction_dots.append(dot)
            print("  interface_direction_dot:", round(dot, 9))

            if dot > -0.95:
                failures.append(
                    f"Interface normals are not opposite enough for {pid}: dot={dot}"
                )

        if occurrence_errors:
            max_err = max(occurrence_errors)
            print("  max_position_error:", round(max_err, 9))

            if max_err > 1e-4:
                failures.append(
                    f"Position error too high for {pid}: max_err={max_err}"
                )

    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "shared_port_id",
            "shared_kind",
            "segment_id",
            "port_index",
            "role",
            "position_error",
            "generated_nx",
            "generated_ny",
            "generated_nz",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 50)
    print("Total expected shared ports:", len(expected))
    print("Total generated port occurrences checked:", len(rows))
    print("Mean position error:", float(np.mean(all_position_errors)))
    print("Max position error:", float(np.max(all_position_errors)))

    if all_interface_direction_dots:
        print("Mean interface direction dot:", float(np.mean(all_interface_direction_dots)))
        print("Max interface direction dot:", float(np.max(all_interface_direction_dots)))

    print("Saved:", out_csv)

    if failures:
        print()
        print("VALIDATION FAILED")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)

    print()
    print("VALIDATION PASSED")
