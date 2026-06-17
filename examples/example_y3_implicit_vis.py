import numpy as np
import pyvista as pv

from respgeomlib.implicit_y3 import three_way_y_implicit_polydata


def main():
    # Geometry in local coords: junction at origin, trunk along -z, children set by angles.
    length_trunk = 6.0
    length_child1 = 4.0
    length_child2 = 4.0
    length_child3 = 4.0
    d_trunk = d_child1 = d_child2 = d_child3 = 2.0

    theta1_deg, phi1_deg = 45.0, 0.0
    theta2_deg, phi2_deg = 45.0, 120.0
    theta3_deg, phi3_deg = 45.0, 240.0
    blend_length = 2.0

    # Smooth blending comes from the implicit distance-field union of tubes.
    mesh = three_way_y_implicit_polydata(
        length_trunk=length_trunk,
        length_child1=length_child1,
        length_child2=length_child2,
        length_child3=length_child3,
        d_trunk=d_trunk,
        d_child1=d_child1,
        d_child2=d_child2,
        d_child3=d_child3,
        theta1_deg=theta1_deg,
        phi1_deg=phi1_deg,
        theta2_deg=theta2_deg,
        phi2_deg=phi2_deg,
        theta3_deg=theta3_deg,
        phi3_deg=phi3_deg,
        blend_length=blend_length,
    )

    print("3-way implicit Y mesh points:", mesh.n_points)
    print("3-way implicit Y mesh cells:", mesh.n_cells)
    print("Bounds:", mesh.bounds)

    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_text("Implicit 3-way Y", font_size=12)
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
        print("Error while visualizing 3-way implicit Y:", e)
