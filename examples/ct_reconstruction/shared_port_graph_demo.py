from pathlib import Path
import sys

# Allow importing the previous demo builder from the same folder
sys.path.append(str(Path(__file__).resolve().parent))

from synthetic_airway_graph_demo import build_demo_graph
from ct_respgeomlib.graph.shared_ports import build_shared_port_graph


if __name__ == "__main__":
    g = build_demo_graph()
    pg = build_shared_port_graph(g, cut_radius_factor=2.5)

    out_dir = Path("outputs/ct_reconstruction")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "synthetic_shared_ports.json"
    pg.save_json(str(out_json))

    print("Shared ports:", len(pg.ports))
    for pid, port in pg.ports.items():
        xyz = [round(float(v), 3) for v in port.xyz]
        normal = [round(float(v), 3) for v in port.normal]
        print(
            pid,
            "| kind=", port.kind,
            "| xyz=", xyz,
            "| normal=", normal,
            "| radius=", round(port.radius, 3),
            "| diameter=", round(2 * port.radius, 3),
        )

    print("Saved:", out_json)
