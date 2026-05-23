from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from synthetic_airway_graph_demo import build_demo_graph
from ct_respgeomlib.graph.shared_ports import build_shared_port_graph
from ct_respgeomlib.decompose.block_decomposition import build_block_decomposition
from ct_respgeomlib.export.respgeomlib_yaml import export_to_respgeomlib_yaml


if __name__ == "__main__":
    g = build_demo_graph()
    pg = build_shared_port_graph(g, cut_radius_factor=2.5)
    dec = build_block_decomposition(g, pg)

    out_dir = Path("outputs/ct_reconstruction")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_yaml = out_dir / "synthetic_ct_respgeomlib.yaml"
    items = export_to_respgeomlib_yaml(dec, pg, str(out_yaml))

    print("Exported segments:", len(items))
    for item in items:
        print()
        print(item["id"])
        print("  kind:", item["kind"])
        print("  parent_id:", item["parent_id"])
        print("  parent_port_index:", item["parent_port_index"])
        print("  params:", item["params"])

    print()
    print("Saved:", out_yaml)
