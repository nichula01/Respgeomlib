from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from synthetic_airway_graph_demo import build_demo_graph
from ct_respgeomlib.graph.shared_ports import build_shared_port_graph
from ct_respgeomlib.decompose.block_decomposition import build_block_decomposition


if __name__ == "__main__":
    g = build_demo_graph()
    pg = build_shared_port_graph(g, cut_radius_factor=2.5)
    dec = build_block_decomposition(g, pg)

    out_dir = Path("outputs/ct_reconstruction")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "synthetic_block_decomposition.json"
    dec.save_json(str(out_json))

    print("Blocks:", len(dec.blocks))

    for bid, block in dec.blocks.items():
        print()
        print(bid)
        print("  type:", block.block_type)
        print("  input_ports:", block.input_ports)
        print("  output_ports:", block.output_ports)
        print("  source_edges:", block.source_edges)
        print("  source_node:", block.source_node)
        print("  label:", block.label)
        print("  params:", block.params)

    print()
    print("Saved:", out_json)
