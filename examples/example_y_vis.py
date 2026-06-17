import numpy as np
import pyvista as pv

from respgeomlib.junctions import two_way_y_polydata


def main():
    # Y is defined in local coordinates: junction at origin, trunk along -z.
    length_trunk = 6.0
    length_child1 = 4.0
    length_child2 = 4.0
    d_trunk = d_child1 = d_child2 = 2.0
    theta1_deg, phi1_deg = 45.0, 0.0
    theta2_deg, phi2_deg = 45.0, 120.0

    mesh = two_way_y_polydata(
        length_trunk=length_trunk,
        length_child1=length_child1,
        length_child2=length_child2,
        d_trunk=d_trunk,
        d_child1=d_child1,
        d_child2=d_child2,
        theta1_deg=theta1_deg,
        phi1_deg=phi1_deg,
        theta2_deg=theta2_deg,
        phi2_deg=phi2_deg,
    )

    print("Y mesh points:", mesh.n_points)
    print("Y mesh cells:", mesh.n_cells)
    print("Bounds:", mesh.bounds)

    # Visualization/test: render the fused Y tube.
    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        color="lightblue",
        smooth_shading=True,
        show_edges=True,
    )
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error while visualizing Y junction:", e)
