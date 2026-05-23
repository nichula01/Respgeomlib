import numpy as np

from ct_respgeomlib.graph.airway_graph import AirwayGraph, AirwayNode, AirwayEdge


def line(p0, p1, n=20):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    t = np.linspace(0, 1, n)
    return (1 - t)[:, None] * p0[None, :] + t[:, None] * p1[None, :]


def build_demo_graph():
    g = AirwayGraph()

    g.add_node(AirwayNode("n0_inlet", np.array([0, 0, 0]), radius=0.90, generation=0, label="trachea_inlet"))
    g.add_node(AirwayNode("n1_carina", np.array([0, 0, 12]), radius=0.85, generation=1, label="carina"))
    g.add_node(AirwayNode("n2_right", np.array([3.5, -1.0, 16]), radius=0.65, generation=2, label="right_main_outlet"))
    g.add_node(AirwayNode("n3_left", np.array([-4.5, 1.0, 18]), radius=0.55, generation=2, label="left_main_outlet"))

    c0 = line([0, 0, 0], [0, 0, 12], n=30)
    r0 = np.linspace(0.90, 0.85, c0.shape[0])
    g.add_edge(AirwayEdge("e_trachea", "n0_inlet", "n1_carina", c0, r0, generation=0, label="trachea"))

    c1 = line([0, 0, 12], [3.5, -1.0, 16], n=20)
    r1 = np.linspace(0.85, 0.65, c1.shape[0])
    g.add_edge(AirwayEdge("e_right_main", "n1_carina", "n2_right", c1, r1, generation=1, label="right_main"))

    c2 = line([0, 0, 12], [-4.5, 1.0, 18], n=20)
    r2 = np.linspace(0.85, 0.55, c2.shape[0])
    g.add_edge(AirwayEdge("e_left_main", "n1_carina", "n3_left", c2, r2, generation=1, label="left_main"))

    return g


if __name__ == "__main__":
    g = build_demo_graph()

    print("Nodes:", len(g.nodes))
    print("Edges:", len(g.edges))
    print("Root nodes:", [n.id for n in g.root_nodes()])
    print("Branch nodes:", [n.id for n in g.branch_nodes()])
    print("Outlet nodes:", [n.id for n in g.outlet_nodes()])

    for edge in g.edges.values():
        print(edge.id, "length=", round(edge.length, 3), "d_in=", round(edge.d_in, 3), "d_out=", round(edge.d_out, 3))
